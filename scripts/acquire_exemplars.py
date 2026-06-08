#!/usr/bin/env python3
"""
acquire_exemplars.py — 期刊样文自动获取脚本（paper-writing-workflow Phase 1.1c 专用）

设计模式与 external_ai_review.py 对齐：
  - 读取 .ai-models-config.yaml 确定 tool-use 模型
  - 渲染 templates/exemplar-acquisition-prompt.template.md
  - 调用 tool-use CLI（headless），获取 PDF 并返回 JSON
  - 验证下载结果，写入 exemplars.json，生成审计日志

Exit codes:
  0  成功：至少 min_exemplars 个 PDF 已验证并注册
  1  部分成功：PDF 数量低于 min_exemplars
  2  tool-use 模型不可用（CLI 未找到）
  3  参数或配置验证失败
  4  输出 JSON 验证失败
  5  tool-use AI 未返回任何候选

用法:
  python scripts/acquire_exemplars.py --journal-slug wrr --project-idea docs/idea.md
  python scripts/acquire_exemplars.py --journal-slug wrr --project-idea docs/idea.md --dry-run
  python scripts/acquire_exemplars.py --check-only --journal-slug wrr
"""

import argparse
import json
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

# ─── 路径常量 ────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DIAGNOSTICS_DIR = PROJECT_ROOT / "docs" / "diagnostics"
EXEMPLARS_ROOT = PROJECT_ROOT / "docs" / "journal-exemplars"
CONFIG_PATH = PROJECT_ROOT / ".ai-models-config.yaml"

# ASCII 路径兼容层（与 external_ai_review.py 相同逻辑）
def _get_ascii_safe_root(project_root: Path) -> Path:
    root_str = str(project_root)
    if root_str.isascii():
        return project_root
    import hashlib
    safe_name = re.sub(r'[^\x20-\x7e]', lambda m: f'_{ord(m.group()):04x}_', project_root.name)
    link_path = Path.home() / safe_name
    real_target = project_root.resolve()
    if link_path.is_symlink():
        if link_path.resolve() != real_target:
            link_path.unlink()
            link_path.symlink_to(real_target)
    elif not link_path.exists():
        link_path.symlink_to(real_target)
    return link_path

SAFE_ROOT = _get_ascii_safe_root(PROJECT_ROOT)


# ─── 配置加载 ────────────────────────────────────────────────────────

def load_config() -> dict:
    """加载 .ai-models-config.yaml，返回 dict。缺失 → exit 2。"""
    try:
        import yaml
    except ImportError:
        # yaml 未安装时用简单解析
        yaml = None

    if not CONFIG_PATH.exists():
        print(f"[acquire_exemplars] 错误: {CONFIG_PATH} 不存在。", file=sys.stderr)
        print("[acquire_exemplars] 请创建 .ai-models-config.yaml 并配置 tool_use_model。", file=sys.stderr)
        sys.exit(2)

    if yaml:
        import yaml as _yaml
        with open(CONFIG_PATH, encoding="utf-8") as f:
            return _yaml.safe_load(f)
    else:
        # 简单 YAML 回退解析（仅支持本文件需要的字段）
        return _simple_yaml_load(CONFIG_PATH)


def _simple_yaml_load(path: Path) -> dict:
    """极简 YAML 解析，支持两级嵌套 dict（不支持列表等复杂结构）。"""
    result: dict = {}
    current_top_key: str | None = None
    with open(path, encoding="utf-8") as f:
        for line in f:
            rstripped = line.rstrip()
            if not rstripped or rstripped.lstrip().startswith("#"):
                continue
            indent = len(line) - len(line.lstrip())
            if indent == 0:
                m = re.match(r'^(\w[\w_-]*):\s*(.*)', rstripped)
                if m:
                    current_top_key = m.group(1)
                    val = m.group(2).strip().strip('"').strip("'")
                    val = val.split('#')[0].strip()  # strip inline comments
                    result[current_top_key] = val if val else {}
            elif indent > 0 and current_top_key:
                m = re.match(r'^\s+(\w[\w_-]*):\s*(.*)', rstripped)
                if m:
                    key = m.group(1)
                    val = m.group(2).strip().strip('"').strip("'")
                    val = val.split('#')[0].strip()
                    if not isinstance(result.get(current_top_key), dict):
                        result[current_top_key] = {}
                    if val:
                        result[current_top_key][key] = val
    return result


def resolve_model(config: dict) -> tuple[str, dict]:
    """从 config 解析 tool_use_model.default → model_mapping。"""
    tool_use = config.get("tool_use_model", {})
    default_name = tool_use.get("default", "") if isinstance(tool_use, dict) else ""
    if not default_name:
        print("[acquire_exemplars] .ai-models-config.yaml 中未设置 tool_use_model.default", file=sys.stderr)
        sys.exit(2)

    mapping = config.get("model_mapping", {})
    if isinstance(mapping, dict) and default_name in mapping:
        return default_name, mapping[default_name]

    # 如果 yaml 解析退化，构造默认 codex mapping
    if default_name == "codex":
        return "codex", {
            "cli_id": "codex",
            "cli_mode": "exec",
            "extra_flags": ["--dangerously-bypass-approvals-and-sandbox"],
            "ascii_path_required": True,
            "timeout_seconds": 1800,
        }

    print(f"[acquire_exemplars] model_mapping 中找不到 '{default_name}'", file=sys.stderr)
    sys.exit(2)


# ─── 模型可用性检查 ──────────────────────────────────────────────────

def check_cli_available(model_cfg: dict) -> bool:
    cli = model_cfg.get("cli_id", "codex")
    return bool(shutil.which(cli))


# ─── 清单与策略加载 ──────────────────────────────────────────────────

def load_manifest(slug: str) -> dict:
    """加载或初始化 manifest.json。"""
    manifest_path = EXEMPLARS_ROOT / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as f:
            return json.load(f)
    return {
        "schema_version": "1.0",
        "scope": "project",
        "parent_manifest": None,
        "journals": {}
    }


def load_selection_policy(slug: str) -> dict:
    """加载 selection_policy.json，缺失时返回默认值。"""
    policy_path = EXEMPLARS_ROOT / slug / "selection_policy.json"
    if policy_path.exists():
        with open(policy_path, encoding="utf-8") as f:
            return json.load(f)
    return {
        "schema_version": "1.0",
        "min_exemplars": 3,
        "recommended_exemplars": 5,
        "preferred_recency_years": 5,
        "prefer_open_access_pdf": True,
    }


def save_manifest(manifest: dict) -> None:
    EXEMPLARS_ROOT.mkdir(parents=True, exist_ok=True)
    path = EXEMPLARS_ROOT / "manifest.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


# ─── 模板渲染 ────────────────────────────────────────────────────────

def find_template() -> Path:
    """在所有已知 skill 库中查找 exemplar-acquisition-prompt.template.md。"""
    candidates = [
        PROJECT_ROOT / ".claude" / "workflows" / "paper-writing" / "templates" / "exemplar-acquisition-prompt.template.md",
        PROJECT_ROOT / ".codex" / "workflows" / "paper-writing" / "templates" / "exemplar-acquisition-prompt.template.md",
    ]
    for p in candidates:
        if p.exists():
            return p
    print("[acquire_exemplars] 错误: exemplar-acquisition-prompt.template.md 未找到", file=sys.stderr)
    sys.exit(3)


def render_template(template_path: Path, values: dict) -> str:
    """渲染 {{ placeholder }} 模板。所有必需占位符必须存在。"""
    text = template_path.read_text(encoding="utf-8")
    missing = []
    for key, val in values.items():
        placeholder = "{{ " + key + " }}"
        if placeholder in text:
            text = text.replace(placeholder, str(val))
    # 检查未替换的占位符
    remaining = re.findall(r'\{\{\s*\w+\s*\}\}', text)
    if remaining:
        print(f"[acquire_exemplars] 错误: 未替换占位符: {remaining}", file=sys.stderr)
        sys.exit(3)
    return text


# ─── 调用 Tool-Use CLI ───────────────────────────────────────────────

def call_tool_use_model(model_name: str, model_cfg: dict, prompt: str,
                        timeout: int, dry_run: bool) -> str:
    """调用 headless tool-use CLI，返回 stdout 文本。"""
    cli = model_cfg.get("cli_id", "codex")
    mode = model_cfg.get("cli_mode", "exec")
    extra_flags = model_cfg.get("extra_flags", [])
    ascii_required = model_cfg.get("ascii_path_required", False)

    run_cwd = str(SAFE_ROOT) if ascii_required else str(PROJECT_ROOT)

    if model_name == "codex":
        cmd = [cli, mode] + extra_flags + ["-C", run_cwd, prompt]
    else:
        cmd = [cli, "-p", prompt]

    if dry_run:
        print("[acquire_exemplars] --dry-run: 渲染后的 prompt 如下\n" + "="*60)
        print(prompt)
        print("="*60)
        print(f"[acquire_exemplars] 调用命令: {' '.join(cmd[:4])} <prompt>")
        sys.exit(0)

    print(f"[acquire_exemplars] 调用 {model_name} (timeout={timeout}s)...", file=sys.stderr)

    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=run_cwd,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout
            else:
                err = result.stderr.strip() or f"exit {result.returncode}, no output"
                print(f"[acquire_exemplars] 第 {attempt} 次失败: {err}", file=sys.stderr)
        except subprocess.TimeoutExpired:
            print(f"[acquire_exemplars] 第 {attempt} 次超时 ({timeout}s)", file=sys.stderr)
        except Exception as e:
            print(f"[acquire_exemplars] 第 {attempt} 次异常: {e}", file=sys.stderr)

        if attempt < max_retries:
            delay = 5 * (2 ** (attempt - 1))
            time.sleep(delay)

    print("[acquire_exemplars] 所有重试失败", file=sys.stderr)
    sys.exit(1)


# ─── JSON 解析与验证 ─────────────────────────────────────────────────

def extract_json(raw_output: str) -> dict:
    """从 raw_output 中提取最外层 JSON 对象。"""
    # 先尝试直接解析
    stripped = raw_output.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    # 找到第一个 { ... }
    match = re.search(r'\{.*\}', stripped, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    print("[acquire_exemplars] 无法从 AI 输出中解析 JSON", file=sys.stderr)
    print("[acquire_exemplars] 原始输出 (前 500 字符):", raw_output[:500], file=sys.stderr)
    sys.exit(4)


def validate_pdf(pdf_path: str) -> bool:
    """验证 PDF 文件：存在、非空、以 %PDF 开头。"""
    p = Path(pdf_path)
    if not p.exists() or p.stat().st_size < 100:
        return False
    try:
        with open(p, "rb") as f:
            header = f.read(4)
        return header == b"%PDF"
    except Exception:
        return False


def validate_doi(doi: str) -> bool:
    return bool(re.match(r'^10\.\d{4,9}/[-._;()/:A-Z0-9]+$', doi, re.IGNORECASE))


# ─── exemplars.json 更新 ─────────────────────────────────────────────

def load_exemplars_json(slug: str) -> dict:
    path = EXEMPLARS_ROOT / slug / "exemplars.json"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {
        "schema_version": "1.0",
        "target_journal": "",
        "article_type": "Research Article",
        "language": "en",
        "papers": []
    }


def save_exemplars_json(slug: str, data: dict) -> None:
    path = EXEMPLARS_ROOT / slug / "exemplars.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def update_exemplars_json(slug: str, candidates: list, journal_name: str) -> int:
    """将验证通过的候选追加到 exemplars.json（按 DOI 去重）。返回新增数量。"""
    data = load_exemplars_json(slug)
    data["target_journal"] = journal_name
    existing_dois = {p["doi"] for p in data["papers"]}
    added = 0

    for c in candidates:
        if not c.get("downloaded"):
            continue
        doi = c.get("doi", "")
        if doi in existing_dois:
            continue
        pdf_path = c.get("pdf_path", "")
        if not validate_pdf(pdf_path):
            print(f"[acquire_exemplars] PDF 验证失败，跳过: {pdf_path}", file=sys.stderr)
            c["validation_status"] = "invalid"
            continue

        entry = {
            "id": re.sub(r'[^a-z0-9]', '-', doi.lower())[:40] if doi else f"unknown-{added}",
            "title": c.get("title", ""),
            "authors": c.get("authors_short", ""),
            "year": c.get("year", 0),
            "journal": c.get("journal", journal_name),
            "article_type": c.get("article_type", "Research Article"),
            "methodology_genre": "",
            "subdomain": "",
            "doi": doi,
            "source_url": c.get("source_url", ""),
            "pdf_path": pdf_path,
            "text_path": "",
            "open_access": c.get("open_access", True),
            "license": c.get("license", "unknown"),
            "pdf_committed_to_vcs": False,
            "validation_status": "valid",
        }
        data["papers"].append(entry)
        existing_dois.add(doi)
        added += 1

    save_exemplars_json(slug, data)
    return added


# ─── 审计日志 ────────────────────────────────────────────────────────

def write_audit(slug: str, model_name: str, output_data: dict, duration: float) -> None:
    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    audit = {
        "schema_version": "1.0",
        "invoked_at": ts,
        "task": "search+download",
        "abstract_model_name": model_name,
        "journal_slug": slug,
        "candidates_considered": output_data.get("candidates_considered", 0),
        "candidates_downloaded": sum(1 for c in output_data.get("candidates", []) if c.get("downloaded")),
        "duration_seconds": round(duration, 1),
    }
    path = DIAGNOSTICS_DIR / f"{ts}_tool_use_invocation.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)
    print(f"[acquire_exemplars] 审计日志: {path}", file=sys.stderr)


# ─── .gitignore ─────────────────────────────────────────────────────

def ensure_gitignore(slug: str) -> None:
    gi = EXEMPLARS_ROOT / ".gitignore"
    entry = "*/pdf/*.pdf\n"
    if gi.exists():
        if entry.strip() in gi.read_text():
            return
    with open(gi, "a", encoding="utf-8") as f:
        f.write(entry)


# ─── 主入口 ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="acquire_exemplars — 期刊样文自动获取")
    parser.add_argument("--journal-slug", default=None,
                        help="期刊 slug（如 wrr）。省略时从 .ai-models-config.yaml project.target_journal_slug 读取")
    parser.add_argument("--project-idea", default="docs/idea.md", help="项目概念文件路径")
    parser.add_argument("--max-candidates", type=int, default=None, help="覆盖 selection_policy 中的 recommended_exemplars")
    parser.add_argument("--year-window", type=int, default=None, help="覆盖 selection_policy 中的 preferred_recency_years")
    parser.add_argument("--dry-run", action="store_true", help="渲染 prompt 并打印，不调用 AI")
    parser.add_argument("--check-only", action="store_true", help="仅检查 tool-use 模型可用性")
    parser.add_argument("--save-prompt", metavar="FILE",
                        help="渲染 prompt 并保存到文件（不调用 AI）。配合 call_codex.sh 实现两步执行模式")
    parser.add_argument("--prompt-file", metavar="FILE",
                        help="从文件读取预渲染的 prompt，跳过模板渲染直接调用 AI（两步执行模式 Step B）")
    args = parser.parse_args()

    # ① 加载配置
    config = load_config()
    model_name, model_cfg = resolve_model(config)

    # 解析 journal slug：命令行 > .ai-models-config.yaml project.target_journal_slug
    slug = args.journal_slug
    if not slug:
        project_cfg = config.get("project", {})
        slug = project_cfg.get("target_journal_slug", "") if isinstance(project_cfg, dict) else ""
    if not slug:
        print("[acquire_exemplars] 错误: 未指定 --journal-slug，且 .ai-models-config.yaml 中无 project.target_journal_slug", file=sys.stderr)
        sys.exit(3)

    # ② 可用性检查
    if not check_cli_available(model_cfg):
        print(f"[acquire_exemplars] 错误: CLI '{model_cfg.get('cli_id')}' 未找到", file=sys.stderr)
        sys.exit(2)
    print(f"[acquire_exemplars] tool-use 模型: {model_name} ✓", file=sys.stderr)

    if args.check_only:
        print(f"[acquire_exemplars] --check-only: {model_name} 可用", file=sys.stderr)
        sys.exit(0)

    # ③ 加载清单与策略
    manifest = load_manifest(slug)
    policy = load_selection_policy(slug)

    # 从 manifest 读取期刊信息；回退顺序：manifest > config project section > 硬编码默认
    project_cfg = config.get("project", {}) if isinstance(config.get("project"), dict) else {}
    journals = manifest.get("journals", {})
    journal_info = journals.get(slug, {})
    journal_name = (journal_info.get("target_journal")
                    or project_cfg.get("target_journal")
                    or "Water Resources Research")
    article_type = (journal_info.get("article_type")
                    or project_cfg.get("article_type")
                    or "Research Article")
    language = (journal_info.get("language")
                or project_cfg.get("language")
                or "en")

    max_candidates = args.max_candidates or policy.get("recommended_exemplars", 5)
    year_window = args.year_window or policy.get("preferred_recency_years", 5)
    year_end = datetime.now().year
    year_start = year_end - year_window

    pdf_dir = EXEMPLARS_ROOT / slug / "pdf"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    ensure_gitignore(slug)

    # ④/⑤ 渲染 prompt（或从文件读取 — 两步执行模式 Step B）
    if args.prompt_file:
        prompt_path = Path(args.prompt_file)
        if not prompt_path.exists():
            print(f"[acquire_exemplars] 错误: --prompt-file 指定的文件不存在: {args.prompt_file}", file=sys.stderr)
            sys.exit(3)
        rendered = prompt_path.read_text(encoding="utf-8")
        print(f"[acquire_exemplars] 从文件读取 prompt: {prompt_path} ({len(rendered)} 字符)", file=sys.stderr)
    else:
        # ④ 验证 project_idea_path
        idea_path = PROJECT_ROOT / args.project_idea
        if not idea_path.exists():
            # 尝试常见备选路径
            for alt in ["PROJECT_CHARTER.md", "docs/idea.md", "docs/narrative-framework.md"]:
                alt_path = PROJECT_ROOT / alt
                if alt_path.exists():
                    idea_path = alt_path
                    break
            else:
                print(f"[acquire_exemplars] 错误: project_idea 文件不存在: {args.project_idea}", file=sys.stderr)
                sys.exit(3)

        # ⑤ 渲染模板
        template_path = find_template()
        ascii_pdf_dir = str(SAFE_ROOT / "docs" / "journal-exemplars" / slug / "pdf")
        ascii_idea_path = str(SAFE_ROOT / idea_path.relative_to(PROJECT_ROOT))

        rendered = render_template(template_path, {
            "target_journal": journal_name,
            "article_type": article_type,
            "language": language,
            "subdomain_hint": "machine learning hydrology, uncertainty quantification, streamflow prediction",
            "project_idea_path": ascii_idea_path,
            "max_candidates": max_candidates,
            "max_attempts": max_candidates * 3,
            "year_start": year_start,
            "year_end": year_end,
            "require_oa": str(policy.get("prefer_open_access_pdf", True)).lower(),
            "output_dir": ascii_pdf_dir,
            "filename_pattern": "<doi-slugified>.pdf",
        })

    # --save-prompt：保存 prompt 到文件，不调用 AI（两步执行模式 Step A）
    if args.save_prompt:
        save_path = Path(args.save_prompt)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(rendered, encoding="utf-8")
        print(f"[acquire_exemplars] ✓ prompt 已保存: {save_path} ({len(rendered)} 字符)", file=sys.stderr)
        print(f"[acquire_exemplars] 两步执行 Step B — 运行以下命令调用 AI:")
        print(f"  bash scripts/call_codex.sh {save_path}")
        print(f"  # 或使用集成方式（call_codex + 自动解析）:")
        print(f"  python scripts/acquire_exemplars.py --prompt-file {save_path}")
        sys.exit(0)

    # ⑥ 调用 AI
    timeout = int(config.get("tool_use_model", {}).get("timeout_seconds", 1800)
                  if isinstance(config.get("tool_use_model"), dict) else 1800)
    t0 = time.time()
    raw_output = call_tool_use_model(model_name, model_cfg, rendered, timeout, args.dry_run)
    duration = time.time() - t0

    # ⑦ 解析验证
    output_data = extract_json(raw_output)
    if "candidates" not in output_data:
        print("[acquire_exemplars] 输出 JSON 缺少 candidates 字段", file=sys.stderr)
        sys.exit(4)

    candidates = output_data.get("candidates", [])
    if not candidates:
        print("[acquire_exemplars] AI 未返回任何候选", file=sys.stderr)
        sys.exit(5)

    # ⑧ 更新 exemplars.json
    added = update_exemplars_json(slug, candidates, journal_name)
    write_audit(slug, model_name, output_data, duration)

    # ⑨ 更新 manifest
    if slug not in manifest.setdefault("journals", {}):
        manifest["journals"][slug] = {}
    manifest["journals"][slug].update({
        "target_journal": journal_name,
        "canonical_slug": slug,
        "language": language,
        "article_type": article_type,
        "exemplar_count": len(load_exemplars_json(slug)["papers"]),
        "pdf_dir": str(pdf_dir.relative_to(PROJECT_ROOT)),
        "exemplars_json": f"docs/journal-exemplars/{slug}/exemplars.json",
        "last_analyzed": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "degraded_mode": "compliance_only",
    })
    save_manifest(manifest)

    # ⑩ 结果报告
    downloaded = [c for c in candidates if c.get("downloaded")]
    min_needed = policy.get("min_exemplars", 3)
    print(f"[acquire_exemplars] 完成: 考察 {output_data.get('candidates_considered', len(candidates))} 篇, "
          f"下载 {len(downloaded)} 篇, 验证新增 {added} 篇", file=sys.stderr)

    if added >= min_needed:
        print(f"[acquire_exemplars] ✓ 达到 min_exemplars={min_needed} 要求", file=sys.stderr)
        sys.exit(0)
    else:
        print(f"[acquire_exemplars] ⚠ 已验证 {added} 篇 < min_exemplars={min_needed}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
