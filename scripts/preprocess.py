#!/usr/bin/env python3
"""
数据预处理脚本模板

用途：将 data/raw/ 中的原始数据处理为 data/processed/ 中的训练数据
使用：python scripts/preprocess.py

输入：data/raw/
输出：data/processed/

重要：
1. 预处理步骤必须可复现（固定随机种子）
2. 处理完成后更新 data/README.md
3. 保存数据分割索引到 data/splits/
"""

import os
import json
import random
import argparse
from pathlib import Path
from datetime import datetime


# === 配置 ===

# 随机种子（确保可复现）
RANDOM_SEED = 42

# 路径配置
ROOT_DIR = Path(__file__).parent.parent
RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
SPLITS_DIR = ROOT_DIR / "data" / "splits"

# 数据分割比例
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1


# === 预处理函数 ===

def set_seed(seed: int):
    """设置随机种子确保可复现"""
    random.seed(seed)
    # 如果使用 numpy:
    # import numpy as np
    # np.random.seed(seed)


def load_raw_data():
    """
    加载原始数据

    TODO: 根据你的数据格式修改此函数

    Returns:
        加载的原始数据
    """
    print(f"从 {RAW_DIR} 加载原始数据...")

    # 示例：列出原始文件
    raw_files = list(RAW_DIR.glob("*"))
    raw_files = [f for f in raw_files if f.name != ".gitkeep"]

    if not raw_files:
        print("⚠️  data/raw/ 目录为空！")
        print("请先下载或放置原始数据。")
        return None

    print(f"找到 {len(raw_files)} 个原始文件")

    # TODO: 实现实际的数据加载逻辑
    # 示例:
    # import pandas as pd
    # data = pd.read_csv(RAW_DIR / "data.csv")
    # return data

    return raw_files


def preprocess(data):
    """
    数据预处理

    TODO: 根据你的需求实现预处理逻辑

    Args:
        data: 原始数据

    Returns:
        处理后的数据
    """
    print("执行预处理...")

    # TODO: 实现预处理逻辑
    # 常见操作:
    # - 数据清洗（处理缺失值、异常值）
    # - 特征工程
    # - 标准化/归一化
    # - 编码（标签编码、one-hot编码）

    return data


def split_data(data, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO, test_ratio=TEST_RATIO):
    """
    划分数据集

    Args:
        data: 预处理后的数据
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例

    Returns:
        (train_data, val_data, test_data)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "比例之和必须为1"

    print(f"划分数据集: train={train_ratio}, val={val_ratio}, test={test_ratio}")

    # TODO: 实现数据划分逻辑
    # 示例（如果data是列表）:
    # n = len(data)
    # indices = list(range(n))
    # random.shuffle(indices)
    #
    # train_end = int(n * train_ratio)
    # val_end = train_end + int(n * val_ratio)
    #
    # train_indices = indices[:train_end]
    # val_indices = indices[train_end:val_end]
    # test_indices = indices[val_end:]

    return None, None, None  # TODO: 返回实际数据


def save_processed_data(train_data, val_data, test_data):
    """
    保存处理后的数据

    Args:
        train_data: 训练数据
        val_data: 验证数据
        test_data: 测试数据
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"保存到 {PROCESSED_DIR}")

    # TODO: 保存处理后的数据
    # 示例:
    # train_data.to_csv(PROCESSED_DIR / "train.csv", index=False)
    # val_data.to_csv(PROCESSED_DIR / "val.csv", index=False)
    # test_data.to_csv(PROCESSED_DIR / "test.csv", index=False)

    # 保存分割信息（用于追溯）
    split_info = {
        "created_at": datetime.now().isoformat(),
        "random_seed": RANDOM_SEED,
        "ratios": {
            "train": TRAIN_RATIO,
            "val": VAL_RATIO,
            "test": TEST_RATIO,
        },
        # "train_size": len(train_data),
        # "val_size": len(val_data),
        # "test_size": len(test_data),
    }

    with open(SPLITS_DIR / "split_info.json", "w") as f:
        json.dump(split_info, f, indent=2)

    print("✓ 分割信息已保存到 data/splits/split_info.json")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="数据预处理脚本")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="随机种子")
    args = parser.parse_args()

    print("=" * 60)
    print("数据预处理脚本")
    print(f"时间: {datetime.now().isoformat()}")
    print(f"随机种子: {args.seed}")
    print("=" * 60)

    # 设置随机种子
    set_seed(args.seed)

    # 加载原始数据
    raw_data = load_raw_data()
    if raw_data is None:
        return

    # 预处理
    processed_data = preprocess(raw_data)

    # 划分数据集
    train_data, val_data, test_data = split_data(processed_data)

    # 保存
    if train_data is not None:
        save_processed_data(train_data, val_data, test_data)

    print("\n" + "=" * 60)
    print("✓ 预处理完成！")
    print("\n下一步:")
    print("1. 更新 data/README.md 记录预处理步骤")
    print("2. 运行 scripts/train.py 开始训练")


if __name__ == "__main__":
    main()
