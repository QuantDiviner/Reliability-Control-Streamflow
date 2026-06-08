#!/usr/bin/env bash
# consensus_round.sh — 并行调用 Codex + Gemini 执行单轮约束共识讨论
#
# 配合 skill `constrained-consensus-discussion` 使用。
# 每轮生成两份独立报告到 docs/reports/，命名符合本项目规范。
#
# 用法:
#   ./scripts/consensus_round.sh <round_number> <prompt_file> [slug]
#
# 示例:
#   ./scripts/consensus_round.sh 1 .tmp/r1_prompt.md tie_direction
#   ./scripts/consensus_round.sh 3 .tmp/r3_prompt.md tie_direction
#
# 输出:
#   docs/reports/<YYYYMMDD_HHMMSS>_<slug>_R<N>_codex.md
#   docs/reports/<YYYYMMDD_HHMMSS>_<slug>_R<N>_gemini.md
#   .tmp/r<N>_codex_stderr.log
#   .tmp/r<N>_gemini_stderr.log
#
# Exit codes:
#   0 = both AIs succeeded (non-empty reports)
#   1 = one AI succeeded, one failed
#   2 = both AIs failed
#
# 环境变量:
#   CODEX_CMD       — Codex 可执行文件（默认 "codex"）
#   GEMINI_CMD      — Gemini 可执行文件（默认 "gemini"）
#   POLL_INTERVAL   — 轮询间隔秒数（默认 20）
#   POLL_MAX_WAIT   — 最大等待秒数（默认 1800 = 30 分钟）
#   SLUG_DEFAULT    — 默认 slug（默认 "consensus"）
#
# 注意: 本脚本仅负责"单轮"并行调用，不负责多轮编排。多轮编排由
#       skill `constrained-consensus-discussion` 的 Claude 主审层驱动。

set -euo pipefail

# ─── 参数解析 ────────────────────────────────────────────────────────

if [ $# -lt 2 ]; then
    cat >&2 <<EOF
Usage: $0 <round_number> <prompt_file> [slug]

Examples:
  $0 1 .tmp/r1_prompt.md tie_direction
  $0 3 .tmp/r3_prompt.md tie_direction

See: .claude/skills/constrained-consensus-discussion.md
EOF
    exit 64
fi

ROUND="$1"
PROMPT_FILE="$2"
SLUG="${3:-${SLUG_DEFAULT:-consensus}}"

# ─── 配置 ────────────────────────────────────────────────────────────

CODEX_CMD="${CODEX_CMD:-codex}"
GEMINI_CMD="${GEMINI_CMD:-gemini}"
POLL_INTERVAL="${POLL_INTERVAL:-20}"
POLL_MAX_WAIT="${POLL_MAX_WAIT:-1800}"

# ─── 校验 ────────────────────────────────────────────────────────────

if ! [[ "$ROUND" =~ ^[0-9]+$ ]]; then
    echo "ERROR: round_number must be a positive integer (got: $ROUND)" >&2
    exit 64
fi

if [ ! -s "$PROMPT_FILE" ]; then
    echo "ERROR: prompt file missing or empty: $PROMPT_FILE" >&2
    exit 64
fi

if ! command -v "$CODEX_CMD" &>/dev/null; then
    echo "ERROR: codex CLI not found (CODEX_CMD=$CODEX_CMD)" >&2
    exit 64
fi

if ! command -v "$GEMINI_CMD" &>/dev/null; then
    echo "ERROR: gemini CLI not found (GEMINI_CMD=$GEMINI_CMD)" >&2
    exit 64
fi

# ─── 路径 ────────────────────────────────────────────────────────────

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPORT_DIR="$PROJECT_ROOT/docs/reports"
TMP_DIR="$PROJECT_ROOT/.tmp"

mkdir -p "$REPORT_DIR" "$TMP_DIR"

TS="$(date +%Y%m%d_%H%M%S)"
R_CODEX="${REPORT_DIR}/${TS}_${SLUG}_R${ROUND}_codex.md"
R_GEMINI="${REPORT_DIR}/${TS}_${SLUG}_R${ROUND}_gemini.md"
LOG_CODEX="${TMP_DIR}/r${ROUND}_codex_stderr.log"
LOG_GEMINI="${TMP_DIR}/r${ROUND}_gemini_stderr.log"

# 用于识别轮询目标的 marker（避免匹配其他并行运行的 codex/gemini）
MARKER="${SLUG}_R${ROUND}"

# ─── 读取 prompt ─────────────────────────────────────────────────────

PROMPT="$(cat "$PROMPT_FILE")"
PROMPT_BYTES=$(wc -c < "$PROMPT_FILE")

echo "─── Consensus Round ${ROUND} ───────────────────"
echo "  Slug:       ${SLUG}"
echo "  Prompt:     ${PROMPT_FILE} (${PROMPT_BYTES} bytes)"
echo "  Codex →     ${R_CODEX}"
echo "  Gemini →    ${R_GEMINI}"
echo "  Timeout:    ${POLL_MAX_WAIT}s (poll every ${POLL_INTERVAL}s)"
echo ""

# ─── 并行启动（detached subshells） ─────────────────────────────────

cd "$PROJECT_ROOT"

(
    "$CODEX_CMD" exec --full-auto -C "$PROJECT_ROOT" -o "$R_CODEX" "$PROMPT" \
        > "$LOG_CODEX" 2>&1 &
) >/dev/null 2>&1

(
    "$GEMINI_CMD" -p "$PROMPT" -y -o text \
        > "$R_GEMINI" 2> "$LOG_GEMINI" &
) >/dev/null 2>&1

echo "Both AI calls launched in background. Polling..."

# ─── 轮询直到完成或超时 ──────────────────────────────────────────────

start_ts=$(date +%s)
while :; do
    elapsed=$(( $(date +%s) - start_ts ))

    codex_done=0
    gemini_done=0

    # Codex 完成条件: 报告非空 且 进程已退出
    if [ -s "$R_CODEX" ] && ! pgrep -f "codex.*exec.*${MARKER}_codex" >/dev/null 2>&1; then
        codex_done=1
    fi

    # Gemini 完成条件: 报告非空 且 进程已退出
    # gemini 命令行里不会包含报告文件名（用重定向），匹配 prompt 长度过于脆弱
    # 改用：进程在跑 gemini 就认为未完成
    if [ -s "$R_GEMINI" ] && ! pgrep -f "^${GEMINI_CMD} -p" >/dev/null 2>&1; then
        gemini_done=1
    fi

    if [ "$codex_done" -eq 1 ] && [ "$gemini_done" -eq 1 ]; then
        break
    fi

    if [ "$elapsed" -ge "$POLL_MAX_WAIT" ]; then
        echo "WARN: timeout after ${elapsed}s; returning whatever completed" >&2
        break
    fi

    # 可选：每 2 分钟打点一次进度
    if [ $(( elapsed % 120 )) -lt "$POLL_INTERVAL" ] && [ "$elapsed" -gt 30 ]; then
        codex_size=$(wc -c < "$R_CODEX" 2>/dev/null || echo 0)
        gemini_size=$(wc -c < "$R_GEMINI" 2>/dev/null || echo 0)
        echo "  [${elapsed}s] codex=${codex_size}B, gemini=${gemini_size}B"
    fi

    sleep "$POLL_INTERVAL"
done

# ─── 结果汇报 ────────────────────────────────────────────────────────

codex_bytes=$(wc -c < "$R_CODEX" 2>/dev/null || echo 0)
gemini_bytes=$(wc -c < "$R_GEMINI" 2>/dev/null || echo 0)
total_elapsed=$(( $(date +%s) - start_ts ))

echo ""
echo "─── Round ${ROUND} Complete (${total_elapsed}s) ─────"
echo "  Codex:  ${R_CODEX} (${codex_bytes} bytes)"
echo "  Gemini: ${R_GEMINI} (${gemini_bytes} bytes)"

# ─── Exit code 判定 ──────────────────────────────────────────────────
# 阈值: 500 bytes 视为"有实质内容"（低于此可能是 stub / error）

MIN_BYTES=500

codex_ok=0
gemini_ok=0
[ "$codex_bytes" -ge "$MIN_BYTES" ] && codex_ok=1
[ "$gemini_bytes" -ge "$MIN_BYTES" ] && gemini_ok=1

if [ "$codex_ok" -eq 1 ] && [ "$gemini_ok" -eq 1 ]; then
    echo "  Status: OK (both succeeded)"
    exit 0
elif [ "$codex_ok" -eq 1 ] || [ "$gemini_ok" -eq 1 ]; then
    echo "  Status: PARTIAL (one AI produced insufficient output)"
    exit 1
else
    echo "  Status: FAILED (both AIs produced insufficient output)"
    echo "  See stderr logs: $LOG_CODEX, $LOG_GEMINI"
    exit 2
fi
