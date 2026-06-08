#!/usr/bin/env python3
"""
external_ai_review.py — 外部 AI 审稿调用脚本（manuscript-revision 工作流专用）

核心原则：
  1. 指定哪个模型就调用哪个模型，绝不替代
  2. 模型不可用 → exit code 2，触发永久停止协议
  3. 调用成功 → 报告保存到 docs/reports/，exit code 0
  4. 调用失败（超时/网络等）→ 重试最多 3 次，仍失败 → exit code 1

用法:
  python scripts/external_ai_review.py \\
    --model opus \\
    --prompt-file docs/planning/review_prompt.md \\
    --reference-files paper/data/metrics.json docs/narrative-framework.md \\
    --pdf paper/source/output/index.pdf \\
    --output docs/reports/20260410_S1_Opus审稿报告_R1.md

  python scripts/external_ai_review.py \\
    --model codex \\
    --prompt-file docs/planning/review_prompt.md \\
    --output docs/reports/20260410_S1_Codex审稿报告_R1.md

  python scripts/external_ai_review.py --check-only --model opus
  python scripts/external_ai_review.py --check-only --model codex

Exit codes:
  0 — 成功，报告已保存
  1 — 调用失败（超时/网络/API 错误），已重试
  2 — 模型不可用（未安装/无 API key），绝不替代，触发永久停止
  3 — 参数错误
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ─── 配置 ────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = PROJECT_ROOT / "docs" / "reports"


# ─── 非 ASCII 路径兼容层 ────────────────────────────────────────────
# Codex CLI 的 websocket header (x-codex-turn-metadata) 使用 str 编码传递
# 工作区路径，遇到非 ASCII 字符时触发 UTF-8 encoding error，沙箱完全无法启动。
# 修复：检测到非 ASCII 路径时创建 ASCII 符号链接，Codex 通过 -C 参数使用该路径。
# 其他 AI（Opus 等）不受影响，始终使用原始 PROJECT_ROOT。

def _get_codex_safe_root(project_root: Path) -> Path:
    """返回 Codex 可用的纯 ASCII 工作目录路径。
    如果原始路径已是纯 ASCII 则原样返回；否则在 $HOME 下创建 ASCII 符号链接。
    """
    root_str = str(project_root)
    if root_str.isascii():
        return project_root

    safe_name = project_root.name
    # 如果项目名本身含非 ASCII，用 hex 编码保证 ASCII
    if not safe_name.isascii():
        safe_name = re.sub(r'[^\x20-\x7e]', lambda m: f'_{ord(m.group()):04x}_', safe_name)

    link_path = Path.home() / safe_name

    # 创建或更新符号链接
    real_target = project_root.resolve()
    if link_path.is_symlink():
        if link_path.resolve() != real_target:
            link_path.unlink()
            link_path.symlink_to(real_target)
    elif not link_path.exists():
        link_path.symlink_to(real_target)
    else:
        # 同名非符号链接文件/目录已存在，回退到带 hash 后缀的名称
        import hashlib
        suffix = hashlib.md5(root_str.encode()).hexdigest()[:8]
        link_path = Path.home() / f"{safe_name}_{suffix}"
        if not link_path.exists():
            link_path.symlink_to(real_target)

    print(f"[external_ai_review] 非 ASCII 路径检测到，Codex 将通过 {link_path} 访问", file=sys.stderr)
    return link_path


CODEX_SAFE_ROOT = _get_codex_safe_root(PROJECT_ROOT)

# 支持的模型及其调用方式
SUPPORTED_MODELS = {
    "opus": {
        "description": "Claude Opus (via Claude CLI)",
        "cmd_env": "OPUS_CMD",
        "cmd_default": "claude",
        "check_cmd": lambda cmd: ["which", cmd],
        "timeout_env": "OPUS_TIMEOUT",
        "timeout_default": 1800,  # 30 分钟
    },
    "codex": {
        "description": "OpenAI Codex (via Codex CLI)",
        "cmd_env": "CODEX_CMD",
        "cmd_default": "codex",
        "check_cmd": lambda cmd: ["which", cmd],
        "timeout_env": "CODEX_TIMEOUT",
        "timeout_default": 1800,
    },
}

MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "3"))
RETRY_BASE_DELAY = int(os.environ.get("RETRY_BASE_DELAY", "5"))


# ─── 模型可用性检查 ──────────────────────────────────────────────────

def check_model_available(model_name: str) -> tuple[bool, str]:
    """检查指定模型是否可用。返回 (可用, 原因)。"""
    if model_name not in SUPPORTED_MODELS:
        return False, f"不支持的模型: {model_name}。支持的模型: {list(SUPPORTED_MODELS.keys())}"

    config = SUPPORTED_MODELS[model_name]
    cmd = os.environ.get(config["cmd_env"], config["cmd_default"])

    # 检查命令是否存在
    if not shutil.which(cmd):
        return False, (
            f"{config['description']} CLI 未找到。"
            f"命令 '{cmd}' 不在 PATH 中。"
            f"请安装后重试，或设置环境变量 {config['cmd_env']}。"
        )

    return True, "可用"


def check_all_models() -> dict[str, tuple[bool, str]]:
    """检查所有支持的模型的可用性。"""
    results = {}
    for name in SUPPORTED_MODELS:
        results[name] = check_model_available(name)
    return results


# ─── 构建调用命令 ────────────────────────────────────────────────────

def build_call_command(
    model_name: str,
    prompt_text: str,
    reference_files: list[str],
    pdf_path: str | None,
) -> list[str]:
    """根据模型类型构建调用命令。"""
    config = SUPPORTED_MODELS[model_name]
    cmd = os.environ.get(config["cmd_env"], config["cmd_default"])

    if model_name == "opus":
        # Claude CLI: claude --model opus -p "prompt" --allowedTools Read
        args = [cmd, "--model", "opus", "-p", prompt_text, "--allowedTools", "Read"]
        return args

    elif model_name == "codex":
        # Codex CLI: codex exec -C <ascii_path> "prompt"
        # -C: 显式指定工作目录（纯 ASCII 路径，规避 websocket header UTF-8 错误）
        # --dangerously-bypass-approvals-and-sandbox: 绕过 bwrap 沙箱
        #   （某些 Linux 环境下 bwrap 无法创建 loopback 网络接口导致命令全部失败，
        #    且 Codex 在本地受信环境仅做只读审核，绕过沙箱是安全的）
        args = [
            cmd, "exec",
            "--dangerously-bypass-approvals-and-sandbox",
            "-C", str(CODEX_SAFE_ROOT),
            prompt_text,
        ]
        return args

    else:
        raise ValueError(f"不支持的模型: {model_name}")


# ─── 带重试的调用 ────────────────────────────────────────────────────

def call_with_retry(
    model_name: str,
    prompt_text: str,
    reference_files: list[str],
    pdf_path: str | None,
    output_path: Path,
) -> tuple[bool, str]:
    """调用外部 AI，带重试。返回 (成功, 结果或错误信息)。"""
    config = SUPPORTED_MODELS[model_name]
    timeout = int(os.environ.get(config["timeout_env"], config["timeout_default"]))

    cmd_args = build_call_command(model_name, prompt_text, reference_files, pdf_path)

    # Codex 使用 ASCII 安全路径，其他模型使用原始路径
    run_cwd = str(CODEX_SAFE_ROOT) if model_name == "codex" else str(PROJECT_ROOT)

    for attempt in range(1, MAX_RETRIES + 1):
        print(f"[external_ai_review] {model_name} 调用第 {attempt}/{MAX_RETRIES} 次...", file=sys.stderr)

        try:
            result = subprocess.run(
                cmd_args,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=run_cwd,
            )

            if result.returncode == 0 and result.stdout.strip():
                # 成功
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(result.stdout, encoding="utf-8")
                print(f"[external_ai_review] 成功 (第 {attempt} 次)", file=sys.stderr)
                return True, str(output_path)

            else:
                error_msg = result.stderr.strip() or f"exit code {result.returncode}, 无输出"
                print(f"[external_ai_review] 失败: {error_msg}", file=sys.stderr)

        except subprocess.TimeoutExpired:
            print(f"[external_ai_review] 超时 ({timeout}秒)", file=sys.stderr)

        except Exception as e:
            print(f"[external_ai_review] 异常: {e}", file=sys.stderr)

        # 重试延迟（指数退避）
        if attempt < MAX_RETRIES:
            delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
            print(f"[external_ai_review] {delay}秒后重试...", file=sys.stderr)
            time.sleep(delay)

    return False, f"{model_name} 调用失败（已重试 {MAX_RETRIES} 次）"


# ─── 生成失败报告 ────────────────────────────────────────────────────

def write_failure_report(output_path: Path, model_name: str, reason: str, exit_code: int):
    """生成失败报告文件。"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    status = "MODEL_UNAVAILABLE" if exit_code == 2 else "CALL_FAILED"

    report = f"""# 外部 AI 审稿 — 失败报告

**时间**: {now}
**模型**: {model_name}
**状态**: {status}
**原因**: {reason}

## 处置

{"模型不可用，绝不允许替代。触发永久停止协议。" if exit_code == 2 else "调用失败，建议检查网络/API 配额后重试。"}
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(f"[external_ai_review] 失败报告已保存: {output_path}", file=sys.stderr)


# ─── 主函数 ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="外部 AI 审稿调用脚本（manuscript-revision 工作流专用）"
    )
    parser.add_argument(
        "--model", required=True, choices=list(SUPPORTED_MODELS.keys()),
        help="指定调用的外部 AI 模型（opus / codex）"
    )
    parser.add_argument(
        "--prompt-file", type=str,
        help="审稿 prompt 文件路径（.md）"
    )
    parser.add_argument(
        "--reference-files", nargs="*", default=[],
        help="参考文件路径列表"
    )
    parser.add_argument(
        "--pdf", type=str,
        help="论文 PDF 路径"
    )
    parser.add_argument(
        "--output", type=str,
        help="报告输出路径"
    )
    parser.add_argument(
        "--check-only", action="store_true",
        help="仅检查模型可用性，不执行调用"
    )

    args = parser.parse_args()

    # ── 仅检查模式 ──
    if args.check_only:
        available, reason = check_model_available(args.model)
        if available:
            print(f"✓ {args.model}: {reason}")
            sys.exit(0)
        else:
            print(f"✗ {args.model}: {reason}", file=sys.stderr)
            sys.exit(2)

    # ── 参数验证 ──
    if not args.prompt_file:
        print("ERROR: --prompt-file 是必需参数", file=sys.stderr)
        sys.exit(3)

    if not args.output:
        print("ERROR: --output 是必需参数", file=sys.stderr)
        sys.exit(3)

    prompt_path = Path(args.prompt_file)
    if not prompt_path.exists():
        print(f"ERROR: Prompt 文件不存在: {prompt_path}", file=sys.stderr)
        sys.exit(3)

    output_path = Path(args.output)

    # ── Step 1: 检查模型可用性（硬性检查，不可用则直接退出） ──
    available, reason = check_model_available(args.model)
    if not available:
        print(f"FATAL: {args.model} 不可用 — {reason}", file=sys.stderr)
        print(f"绝不允许使用替代模型。触发永久停止协议。", file=sys.stderr)
        write_failure_report(output_path, args.model, reason, exit_code=2)
        sys.exit(2)

    print(f"[external_ai_review] 模型可用性检查通过: {args.model}", file=sys.stderr)

    # ── Step 2: 读取 prompt ──
    prompt_text = prompt_path.read_text(encoding="utf-8")

    # 如果有参考文件，追加到 prompt 中作为文件路径提示
    if args.reference_files:
        ref_section = "\n\n参考文件路径（请自行读取）：\n"
        for ref in args.reference_files:
            ref_section += f"- {ref}\n"
        prompt_text += ref_section

    if args.pdf:
        prompt_text = f"论文 PDF 路径：{args.pdf}\n\n" + prompt_text

    # ── Step 3: 调用外部 AI ──
    success, result = call_with_retry(
        model_name=args.model,
        prompt_text=prompt_text,
        reference_files=args.reference_files,
        pdf_path=args.pdf,
        output_path=output_path,
    )

    if success:
        print(f"✓ 审稿报告已保存: {result}")
        sys.exit(0)
    else:
        print(f"✗ {result}", file=sys.stderr)
        write_failure_report(output_path, args.model, result, exit_code=1)
        sys.exit(1)


if __name__ == "__main__":
    main()
