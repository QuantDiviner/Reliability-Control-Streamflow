#!/bin/bash
# CAMELS-GB 数据下载脚本
# 数据源: NERC EIDC, Coxon et al. 2020, DOI: 10.5285/8344e4f3-d2ea-44f5-8afa-86d2987543a9
# 许可: Open Government Licence (OGL)
# 引用要求: Coxon, G. et al. (2020). CAMELS-GB. NERC EIDC. https://doi.org/10.5285/8344e4f3-d2ea-44f5-8afa-86d2987543a9
#
# 注意: EIDC 服务器不支持 Range 请求，wget -c 断点续传无效。
#       务必在 tmux detached 会话中运行，防止 SSH 断线导致重新下载。

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RAW_DIR="$PROJECT_ROOT/data/raw/CAMELS_GB"
URL="https://data-package.ceh.ac.uk/data/8344e4f3-d2ea-44f5-8afa-86d2987543a9.zip"
ZIP_FILE="$RAW_DIR/CAMELS_GB.zip"
DONE_FLAG="$RAW_DIR/.download_done"
LOG_FILE="$RAW_DIR/download.log"

mkdir -p "$RAW_DIR"

echo "=== CAMELS-GB 下载脚本 ==="
echo "目标目录: $RAW_DIR"
echo "URL: $URL"
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

if [ -f "$DONE_FLAG" ]; then
    echo "[SKIP] 检测到 .download_done，下载已完成。如需重下，删除该文件后重跑。"
    exit 0
fi

# 通过 catalogue.ceh.ac.uk wrapper 验证 OGL 已被接受（OGL 是公开许可，无需登录）。
# 直接拉 zip。--retry 5 应对网络抖动，但服务器无 Range 故重试会从头开始。
echo "[INFO] 开始下载（约几百 MB ~ 数 GB，预计 700 KB/s ≈ 30-60 min）..."
wget --tries=5 --timeout=120 --progress=dot:giga \
     -O "$ZIP_FILE.partial" "$URL" 2>&1 | tee "$LOG_FILE"

# 验证 zip 完整性
echo ""
echo "[INFO] 验证 zip 完整性..."
if unzip -t "$ZIP_FILE.partial" > /dev/null 2>&1; then
    mv "$ZIP_FILE.partial" "$ZIP_FILE"
    touch "$DONE_FLAG"
    SIZE=$(du -sh "$ZIP_FILE" | cut -f1)
    echo "[OK] 下载完成。文件大小: $SIZE"
    echo "[OK] $DONE_FLAG 已创建。"
    echo "结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    echo "下一步: 解压并按 NeuralHydrology camels_gb dataset class 组织目录。"
    echo "  cd $RAW_DIR && unzip -q CAMELS_GB.zip"
else
    echo "[FAIL] zip 完整性验证失败，保留 .partial 供检查。"
    exit 1
fi
