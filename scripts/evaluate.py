#!/usr/bin/env python3
"""
评估脚本模板

用途：在测试集上评估训练好的模型
使用：python scripts/evaluate.py --experiment experiments/exp001
     python scripts/evaluate.py --checkpoint path/to/model.pt

输入：训练好的模型检查点 + 测试数据
输出：详细评估指标 → results/metrics.json

重要：
1. 评估结果必须保存到 results/metrics.json
2. 这是 collect_results.py 读取的数据源
"""

import os
import json
import argparse
from pathlib import Path
from datetime import datetime


# === 配置 ===

ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data" / "processed"


# === 评估函数 ===

def load_model(checkpoint_path: Path):
    """
    加载训练好的模型

    TODO: 根据你的模型框架修改
    """
    print(f"加载模型: {checkpoint_path}")

    # TODO: 实现模型加载
    # 示例 (PyTorch):
    # checkpoint = torch.load(checkpoint_path)
    # model = create_model(config)
    # model.load_state_dict(checkpoint["model_state_dict"])
    # model.eval()
    # return model

    return None


def load_test_data(data_dir: Path):
    """
    加载测试数据

    TODO: 根据你的数据格式修改
    """
    print(f"加载测试数据: {data_dir}")

    # TODO: 实现测试数据加载
    # test_data = pd.read_csv(data_dir / "test.csv")
    # return test_data

    return None


def evaluate(model, test_data) -> dict:
    """
    评估模型

    TODO: 实现评估逻辑

    Returns:
        dict: 评估指标
    """
    print("开始评估...")

    # TODO: 实现评估
    # 常见指标:
    # - 分类: accuracy, precision, recall, f1, confusion_matrix
    # - 回归: mse, rmse, mae, r2
    # - 其他: inference_time, memory_usage

    metrics = {
        # === 主要指标（用于论文） ===
        "accuracy": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,

        # === 效率指标 ===
        "inference_time_ms": 0.0,
        "model_params": 0,
        "model_size_mb": 0.0,

        # === 详细结果 ===
        # "per_class_accuracy": {},
        # "confusion_matrix": [],
    }

    return metrics


def save_metrics(metrics: dict, save_path: Path):
    """保存评估指标"""
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # 添加元信息
    metrics["_meta"] = {
        "evaluated_at": datetime.now().isoformat(),
        "script": "scripts/evaluate.py",
    }

    with open(save_path, "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"✓ 指标已保存: {save_path}")


def print_metrics(metrics: dict):
    """打印主要指标"""
    print("\n" + "=" * 40)
    print("评估结果")
    print("=" * 40)

    for key, value in metrics.items():
        if key.startswith("_"):
            continue
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="评估脚本")
    parser.add_argument(
        "--experiment",
        type=str,
        help="实验目录路径 (如 experiments/exp001)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="模型检查点路径（如果不使用 --experiment）"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("评估脚本")
    print(f"时间: {datetime.now().isoformat()}")
    print("=" * 60)

    # 确定检查点路径
    if args.experiment:
        experiment_dir = Path(args.experiment)
        checkpoint_dir = experiment_dir / "checkpoints"
        results_dir = experiment_dir / "results"

        # 找到最新的检查点
        checkpoints = list(checkpoint_dir.glob("*.pt")) + list(checkpoint_dir.glob("*.pth"))
        if not checkpoints:
            print(f"❌ 未找到检查点: {checkpoint_dir}")
            return
        checkpoint_path = max(checkpoints, key=lambda p: p.stat().st_mtime)
    elif args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        results_dir = checkpoint_path.parent.parent / "results"
    else:
        print("❌ 请指定 --experiment 或 --checkpoint")
        return

    print(f"检查点: {checkpoint_path}")
    print(f"结果目录: {results_dir}")

    # 加载模型
    model = load_model(checkpoint_path)

    # 加载测试数据
    test_data = load_test_data(DATA_DIR)

    # 评估
    metrics = evaluate(model, test_data)

    # 打印结果
    print_metrics(metrics)

    # 保存结果
    save_metrics(metrics, results_dir / "metrics.json")

    print("\n" + "=" * 60)
    print("✓ 评估完成！")
    print("\n下一步:")
    print("1. 查看 results/metrics.json")
    print("2. 运行 python paper/scripts/collect_results.py 收集数据")


if __name__ == "__main__":
    main()
