#!/usr/bin/env python3
"""
训练脚本模板

用途：训练模型并保存结果
使用：python scripts/train.py --config configs/default.yaml
     python scripts/train.py --config experiments/exp001/config.yaml

输入：data/processed/ (训练数据)
输出：experiments/expXXX/ (模型检查点、日志、指标)

重要：
1. 必须固定随机种子
2. 训练完成后自动保存 results/metrics.json
3. 所有超参数从配置文件读取
"""

import os
import json
import random
import argparse
from pathlib import Path
from datetime import datetime


# === 配置 ===

ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data" / "processed"
DEFAULT_CONFIG = ROOT_DIR / "configs" / "default.yaml"


# === 工具函数 ===

def set_seed(seed: int):
    """设置所有随机种子"""
    random.seed(seed)

    # 如果使用 numpy
    # import numpy as np
    # np.random.seed(seed)

    # 如果使用 PyTorch
    # import torch
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    # 如果使用 TensorFlow
    # import tensorflow as tf
    # tf.random.set_seed(seed)


def load_config(config_path: Path) -> dict:
    """加载配置文件"""
    import yaml  # 需要 pip install pyyaml

    with open(config_path) as f:
        config = yaml.safe_load(f)

    print(f"配置已加载: {config_path}")
    return config


def load_data(data_dir: Path):
    """
    加载训练数据

    TODO: 根据你的数据格式修改
    """
    print(f"从 {data_dir} 加载数据...")

    # TODO: 实现数据加载
    # 示例:
    # train_data = pd.read_csv(data_dir / "train.csv")
    # val_data = pd.read_csv(data_dir / "val.csv")
    # return train_data, val_data

    return None, None


def create_model(config: dict):
    """
    创建模型

    TODO: 根据你的模型修改
    """
    print("创建模型...")

    # TODO: 实现模型创建
    # 示例:
    # from src.models import MyModel
    # model = MyModel(
    #     hidden_dim=config["model"]["hidden_dim"],
    #     num_layers=config["model"]["num_layers"],
    # )
    # return model

    return None


def train_epoch(model, train_data, config):
    """
    训练一个 epoch

    TODO: 实现训练逻辑
    """
    # TODO: 实现训练循环
    pass


def validate(model, val_data, config):
    """
    验证模型

    TODO: 实现验证逻辑

    Returns:
        dict: 验证指标
    """
    # TODO: 实现验证
    return {"val_loss": 0.0, "val_accuracy": 0.0}


def save_checkpoint(model, optimizer, epoch, metrics, save_dir: Path):
    """保存检查点"""
    save_dir.mkdir(parents=True, exist_ok=True)

    # TODO: 保存模型
    # 示例 (PyTorch):
    # torch.save({
    #     "epoch": epoch,
    #     "model_state_dict": model.state_dict(),
    #     "optimizer_state_dict": optimizer.state_dict(),
    #     "metrics": metrics,
    # }, save_dir / f"checkpoint_epoch_{epoch}.pt")

    print(f"检查点已保存: epoch {epoch}")


def save_metrics(metrics: dict, save_path: Path):
    """
    保存最终指标到 metrics.json

    这是 collect_results.py 读取的格式
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # 添加元信息
    metrics["_meta"] = {
        "saved_at": datetime.now().isoformat(),
        "script": "scripts/train.py",
    }

    with open(save_path, "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"✓ 指标已保存: {save_path}")


def train(config: dict, experiment_dir: Path):
    """主训练函数"""

    # 设置随机种子
    seed = config.get("seed", 42)
    set_seed(seed)
    print(f"随机种子: {seed}")

    # 创建输出目录
    logs_dir = experiment_dir / "logs"
    checkpoints_dir = experiment_dir / "checkpoints"
    results_dir = experiment_dir / "results"

    logs_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # 加载数据
    train_data, val_data = load_data(DATA_DIR)

    # 创建模型
    model = create_model(config)

    # 训练参数
    epochs = config.get("training", {}).get("epochs", 100)

    print(f"\n开始训练: {epochs} epochs")
    print("-" * 40)

    best_metrics = {}

    for epoch in range(1, epochs + 1):
        # 训练
        train_epoch(model, train_data, config)

        # 验证
        metrics = validate(model, val_data, config)

        print(f"Epoch {epoch}/{epochs}: {metrics}")

        # 保存最佳模型（示例：基于 val_loss）
        # if not best_metrics or metrics["val_loss"] < best_metrics.get("val_loss", float("inf")):
        #     best_metrics = metrics.copy()
        #     best_metrics["best_epoch"] = epoch
        #     save_checkpoint(model, None, epoch, metrics, checkpoints_dir)

    # 保存最终指标
    final_metrics = {
        "accuracy": 0.0,  # TODO: 替换为实际指标
        "loss": 0.0,
        "epochs_trained": epochs,
        "seed": seed,
        # 添加更多你需要在论文中引用的指标
    }

    save_metrics(final_metrics, results_dir / "metrics.json")

    return final_metrics


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="训练脚本")
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG),
        help="配置文件路径"
    )
    parser.add_argument(
        "--experiment-dir",
        type=str,
        default=None,
        help="实验输出目录（默认从配置文件推断）"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("训练脚本")
    print(f"时间: {datetime.now().isoformat()}")
    print("=" * 60)

    # 加载配置
    config_path = Path(args.config)
    config = load_config(config_path)

    # 确定实验目录
    if args.experiment_dir:
        experiment_dir = Path(args.experiment_dir)
    elif config_path.parent.name.startswith("exp"):
        # 配置在实验目录中
        experiment_dir = config_path.parent
    else:
        # 使用默认配置，创建临时实验目录
        experiment_dir = ROOT_DIR / "experiments" / "_scratch" / datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"实验目录: {experiment_dir}")

    # 训练
    metrics = train(config, experiment_dir)

    print("\n" + "=" * 60)
    print("✓ 训练完成！")
    print(f"结果保存在: {experiment_dir}")
    print("\n下一步:")
    print("1. 查看 results/metrics.json")
    print("2. 在 EXPERIMENT_LOG.md 中记录本次实验")
    print("3. 运行 scripts/evaluate.py 进行详细评估")


if __name__ == "__main__":
    main()
