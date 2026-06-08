#!/usr/bin/env bash
# call_codex.sh — 薄包装器：从文件读取 prompt，调用 codex exec headless
#
# 用法:
#   bash scripts/call_codex.sh <prompt_file>
#   bash scripts/call_codex.sh <prompt_file> --output-file <json_file>
#
# 两步执行模式（防止 Claude Code 在决策点意外触发 MCP 工具）:
#   Step A（Claude Code 负责，纯文件写作，无歧义）:
#     python scripts/acquire_exemplars.py --journal-slug wrr \
#       --save-prompt docs/planning/exemplar_prompt.md
#
#   Step B（本脚本负责，纯命令执行，无决策）:
#     bash scripts/call_codex.sh docs/planning/exemplar_prompt.md
#
#   或 Step B 集成版（调用 codex + 自动解析 exemplars.json）:
#     python scripts/acquire_exemplars.py --journal-slug wrr \
#       --prompt-file docs/planning/exemplar_prompt.md

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── 参数解析 ─────────────────────────────────────────────────────────────
if [[ $# -lt 1 ]]; then
    echo "用法: $0 <prompt_file> [--output-file <file>]" >&2
    exit 1
fi

PROMPT_FILE="$1"
OUTPUT_FILE=""
shift

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output-file) OUTPUT_FILE="$2"; shift 2 ;;
        *) echo "[call_codex] 未知参数: $1" >&2; exit 1 ;;
    esac
done

if [[ ! -f "$PROMPT_FILE" ]]; then
    echo "[call_codex] 错误: prompt 文件不存在: $PROMPT_FILE" >&2
    exit 1
fi

# ── ASCII 路径兼容（与 acquire_exemplars.py _get_ascii_safe_root 逻辑一致）
_get_safe_root() {
    python3 - "$1" <<'PYEOF'
import re, sys
from pathlib import Path
root = Path(sys.argv[1])
if str(root).isascii():
    print(root)
    sys.exit(0)
safe_name = re.sub(r'[^\x20-\x7e]', lambda m: f'_{ord(m.group()):04x}_', root.name)
link = Path.home() / safe_name
real = root.resolve()
if link.is_symlink():
    if link.resolve() != real:
        link.unlink()
        link.symlink_to(real)
elif not link.exists():
    link.symlink_to(real)
    print(f"[call_codex] 创建 ASCII 符号链接: {link} -> {real}", file=__import__('sys').stderr)
print(link)
PYEOF
}

SAFE_ROOT="$(_get_safe_root "$PROJECT_ROOT")"

# ── codex 可用性检查 ──────────────────────────────────────────────────────
if ! command -v codex &>/dev/null; then
    echo "[call_codex] 错误: codex CLI 未找到，请确认已安装 OpenAI Codex CLI" >&2
    exit 2
fi

# ── 读取 prompt 并调用 ────────────────────────────────────────────────────
PROMPT_CONTENT="$(cat "$PROMPT_FILE")"
PROMPT_BYTES="$(wc -c < "$PROMPT_FILE" | tr -d ' ')"
echo "[call_codex] prompt 文件: $PROMPT_FILE (${PROMPT_BYTES} 字节)" >&2
echo "[call_codex] 调用 codex exec (working dir: $SAFE_ROOT)..." >&2

if [[ -n "$OUTPUT_FILE" ]]; then
    mkdir -p "$(dirname "$OUTPUT_FILE")"
    codex exec --dangerously-bypass-approvals-and-sandbox -C "$SAFE_ROOT" "$PROMPT_CONTENT" \
        | tee "$OUTPUT_FILE"
    echo "[call_codex] ✓ 输出已保存: $OUTPUT_FILE" >&2
else
    codex exec --dangerously-bypass-approvals-and-sandbox -C "$SAFE_ROOT" "$PROMPT_CONTENT"
fi
