#!/usr/bin/env python
"""Render 18 LORO fold configs from experiments/exp003/template_config.yml.

Placeholders: {HUC} (zero-padded '01'..'18'), {REPO_ROOT}.

Output: experiments/exp003/configs/loro_huc{k:02d}.yml (18 files).
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = REPO_ROOT / "experiments" / "exp003" / "template_config.yml"
OUT_DIR = REPO_ROOT / "experiments" / "exp003" / "configs"


def main() -> None:
    template_text = TEMPLATE.read_text()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for k in range(1, 19):
        huc = f"{k:02d}"
        rendered = template_text.format(HUC=huc, REPO_ROOT=str(REPO_ROOT))
        out = OUT_DIR / f"loro_huc{huc}.yml"
        out.write_text(rendered)
        written.append(out)

    print(f"Wrote {len(written)} fold configs to {OUT_DIR}")
    for p in written:
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
