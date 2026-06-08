#!/bin/bash
# 数据解压和目录整理脚本
# 在 CAMELS zip 下载完成后运行

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RAW_DIR="$PROJECT_ROOT/data/raw"
CAMELS_DIR="$RAW_DIR/CAMELS_US"
ZIP_FILE="$RAW_DIR/basin_timeseries_v1p2_metForcing_obsFlow.zip"

echo "=== CAMELS 数据解压脚本 ==="
echo "项目根目录: $PROJECT_ROOT"
echo ""

# 检查 zip 是否已完整下载
if [ ! -f "$ZIP_FILE" ]; then
    echo "错误: 找不到 $ZIP_FILE"
    echo "请先运行数据下载脚本"
    exit 1
fi

ZIP_SIZE=$(du -sh "$ZIP_FILE" | cut -f1)
echo "Zip 文件大小: $ZIP_SIZE"

# 解压到 raw/ 目录
echo ""
echo "正在解压 basin_timeseries_v1p2_metForcing_obsFlow.zip ..."
echo "（预计解压后约 14.9 GB，需要几分钟）"
cd "$RAW_DIR"
unzip -q "$ZIP_FILE" -d "$RAW_DIR/"

# NeuralHydrology 期望的目录名是 CAMELS_US
# 解压后的目录名是 basin_dataset_public_v1p2
if [ -d "$RAW_DIR/basin_dataset_public_v1p2" ]; then
    echo "重命名 basin_dataset_public_v1p2 → CAMELS_US/basin_dataset_public_v1p2"
    # 将解压内容合并到已有的 CAMELS_US 目录
    mv "$RAW_DIR/basin_dataset_public_v1p2"/* "$CAMELS_DIR/" 2>/dev/null || true
    rmdir "$RAW_DIR/basin_dataset_public_v1p2" 2>/dev/null || true
fi

echo ""
echo "=== 验证目录结构 ==="
echo "期望结构:"
echo "  CAMELS_US/"
echo "  ├── basin_mean_forcing/"
echo "  ├── usgs_streamflow/"
echo "  └── camels_attributes_v2.0/"
echo ""
echo "实际结构:"
ls "$CAMELS_DIR/"

echo ""
echo "=== 验证属性文件 ==="
ls "$CAMELS_DIR/camels_attributes_v2.0/"

echo ""
echo "=== 统计流域数量 ==="
if [ -d "$CAMELS_US_DIR/usgs_streamflow" ]; then
    BASIN_COUNT=$(ls "$CAMELS_DIR/usgs_streamflow" | wc -l)
    echo "流域数量: $BASIN_COUNT"
fi

echo ""
echo "解压完成！"
echo ""
echo "下一步: 激活 conda 环境并安装 NeuralHydrology"
echo "  conda activate hscc-hydrology"
echo "  cd libs/neuralhydrology && pip install -e . && cd ../.."
