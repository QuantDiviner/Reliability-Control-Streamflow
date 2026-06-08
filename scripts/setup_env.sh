#!/bin/bash
# 跨平台环境安装脚本
# 自动检测平台，选择对应的 environment yml

set -e
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=== HSCC 项目环境安装 ==="
echo "项目根目录: $PROJECT_ROOT"
echo ""

# 检测平台
OS=$(uname -s)
echo "操作系统: $OS"

# 检测 NVIDIA GPU
if command -v nvidia-smi &>/dev/null; then
    NVIDIA_AVAILABLE=true
    CUDA_VER=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d. -f1)
    echo "NVIDIA GPU 检测: 可用 (CUDA $CUDA_VER)"
else
    NVIDIA_AVAILABLE=false
    echo "NVIDIA GPU 检测: 不可用 (使用 CPU)"
fi

echo ""

# 选择 environment 文件
if [ "$NVIDIA_AVAILABLE" = true ] && [ "$OS" = "Linux" ]; then
    ENV_FILE="environment-cuda.yml"
    echo "选择环境: $ENV_FILE (Linux + CUDA)"
else
    ENV_FILE="environment.yml"
    echo "选择环境: $ENV_FILE (CPU-only)"
fi

# 检查环境是否已存在
if conda env list | grep -q "hscc-hydrology"; then
    echo ""
    echo "环境 hscc-hydrology 已存在"
    echo "如需重建: conda env remove -n hscc-hydrology && bash scripts/setup_env.sh"
    exit 0
fi

# 创建 conda 环境
echo ""
echo "创建 conda 环境（可能需要 5-15 分钟）..."
conda env create -f "$ENV_FILE"

echo ""
echo "=== 安装 NeuralHydrology ==="
conda run -n hscc-hydrology pip install -e libs/neuralhydrology/

echo ""
echo "=== 验证安装 ==="
conda run -n hscc-hydrology python -c "
import torch
import neuralhydrology
import mapie
print(f'PyTorch: {torch.__version__}')
print(f'CUDA 可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'NeuralHydrology: OK')
print(f'MAPIE: {mapie.__version__}')
"

echo ""
echo "=== 完成 ==="
echo "激活环境: conda activate hscc-hydrology"
echo ""
echo "下一步:"
echo "  1. 等待数据下载完成: ls -lh data/raw/"
echo "  2. 解压数据: bash scripts/setup_data.sh"
echo "  3. 运行 Go/no-go 实验: conda activate hscc-hydrology && python scripts/go_nogo_experiment.py"
