#!/bin/bash
# Linux (Ubuntu) + NVIDIA GPU 部署脚本
# 在 Ubuntu 机器上首次部署时运行
#
# 前置条件:
#   - Ubuntu 20.04 / 22.04
#   - NVIDIA 驱动 >= 525 (支持 CUDA 12.1)
#   - 已安装 miniconda / anaconda
#   - 已克隆项目: git clone <repo_url>
#
# 运行方式:
#   bash scripts/deploy_linux_gpu.sh

set -e
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=== HSCC 项目 Linux + GPU 部署 ==="
echo ""

# 检查 NVIDIA GPU
echo "检查 NVIDIA GPU..."
if ! command -v nvidia-smi &>/dev/null; then
    echo "错误: 未检测到 nvidia-smi，请先安装 NVIDIA 驱动"
    exit 1
fi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
echo ""

# 克隆 NeuralHydrology（如果还没有）
if [ ! -d "libs/neuralhydrology" ]; then
    echo "克隆 NeuralHydrology..."
    git clone https://github.com/neuralhydrology/neuralhydrology.git libs/neuralhydrology
fi

# 创建 conda 环境
if conda env list | grep -q "hscc-hydrology"; then
    echo "conda 环境已存在，跳过创建"
else
    echo "创建 conda 环境 (CUDA 版本)..."
    conda env create -f environment-cuda.yml
fi

# 安装 NeuralHydrology
echo "安装 NeuralHydrology..."
conda run -n hscc-hydrology pip install "numpy>=1.24,<2.0" -q
conda run -n hscc-hydrology pip install -e libs/neuralhydrology/ -q

# 安装 CP 库
echo "安装 Conformal Prediction 库..."
conda run -n hscc-hydrology pip install "mapie>=0.8" "crepes>=0.6" "hydroeval" -q

# 验证
echo ""
echo "=== 验证安装 ==="
conda run -n hscc-hydrology python -c "
import torch, numpy, mapie
from neuralhydrology.utils.config import Config

print(f'PyTorch:    {torch.__version__}')
print(f'NumPy:      {numpy.__version__}')
print(f'CUDA 可用:   {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU 数量:   {torch.cuda.device_count()}')
    print(f'GPU 名称:   {torch.cuda.get_device_name(0)}')
    print(f'GPU 显存:   {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
print(f'MAPIE:      {mapie.__version__}')
print()
print('安装验证通过 ✓')
"

echo ""
echo "=== 下载 CAMELS 数据 ==="
echo "现在开始下载 CAMELS-US 数据集 (~3.2 GB 压缩包, ~14.9 GB 解压后)"
mkdir -p data/raw/CAMELS_US/camels_attributes_v2.0

BASE="https://zenodo.org/api/records/15529996/files"
ATTR_DIR="data/raw/CAMELS_US/camels_attributes_v2.0"

# 下载属性文件
echo "下载属性文件..."
for fname in camels_clim.txt camels_geol.txt camels_hydro.txt camels_name.txt camels_soil.txt camels_topo.txt camels_vege.txt; do
    curl -L -s -o "$ATTR_DIR/$fname" "$BASE/$fname/content"
    echo "  $fname OK"
done
curl -L -s -o "data/raw/CAMELS_US/readme.txt" "$BASE/readme.txt/content"

# 下载主数据（大文件）
echo ""
echo "下载主时序数据 (3.2 GB)，请耐心等待..."
curl -L --progress-bar \
    -o "data/raw/basin_timeseries_v1p2_metForcing_obsFlow.zip" \
    "$BASE/basin_timeseries_v1p2_metForcing_obsFlow.zip/content"

echo ""
echo "解压数据..."
bash scripts/setup_data.sh

echo ""
echo "=== 部署完成 ==="
echo "激活环境: conda activate hscc-hydrology"
echo "运行验证: python scripts/verify_setup.py"
