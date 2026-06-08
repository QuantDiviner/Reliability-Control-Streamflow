#!/usr/bin/env python3
"""
数据下载脚本模板

用途：下载原始数据到 data/raw/ 目录
使用：python scripts/download_data.py

重要：
1. 下载的数据应保存到 data/raw/
2. 下载后在 data/README.md 中记录数据来源
3. 大文件（>50MB）考虑使用 Git LFS 或不纳入版本控制
"""

import os
import hashlib
import urllib.request
from pathlib import Path
from datetime import datetime


# === 配置 ===

# 数据源定义
# 格式: {名称: {url: 下载地址, filename: 保存文件名, sha256: 校验和(可选)}}
DATA_SOURCES = {
    # 示例：
    # "dataset_train": {
    #     "url": "https://example.com/data/train.csv",
    #     "filename": "train.csv",
    #     "sha256": "abc123...",  # 可选，用于验证完整性
    # },
    # "dataset_test": {
    #     "url": "https://example.com/data/test.csv",
    #     "filename": "test.csv",
    # },
}

# 输出目录
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "raw"


# === 工具函数 ===

def calculate_sha256(filepath: Path) -> str:
    """计算文件的 SHA256 校验和"""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def download_file(url: str, dest: Path, expected_sha256: str = None) -> bool:
    """
    下载文件并可选验证校验和

    Args:
        url: 下载地址
        dest: 保存路径
        expected_sha256: 期望的 SHA256 校验和（可选）

    Returns:
        bool: 下载是否成功
    """
    print(f"下载: {url}")
    print(f"保存到: {dest}")

    try:
        # 下载
        urllib.request.urlretrieve(url, dest)

        # 验证校验和（如果提供）
        if expected_sha256:
            actual_sha256 = calculate_sha256(dest)
            if actual_sha256 != expected_sha256:
                print(f"❌ 校验和不匹配!")
                print(f"   期望: {expected_sha256}")
                print(f"   实际: {actual_sha256}")
                return False
            print(f"✓ 校验和验证通过")

        print(f"✓ 下载完成: {dest.name}")
        return True

    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return False


def main():
    """主函数：下载所有数据"""

    print("=" * 60)
    print("数据下载脚本")
    print(f"时间: {datetime.now().isoformat()}")
    print(f"输出目录: {OUTPUT_DIR}")
    print("=" * 60)

    # 检查数据源配置
    if not DATA_SOURCES:
        print("\n⚠️  DATA_SOURCES 为空！")
        print("请在脚本中配置数据源后再运行。")
        print("\n示例配置:")
        print('''
DATA_SOURCES = {
    "my_dataset": {
        "url": "https://example.com/data.csv",
        "filename": "data.csv",
    },
}
''')
        return

    # 创建输出目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 下载统计
    success_count = 0
    fail_count = 0

    # 下载每个数据源
    for name, config in DATA_SOURCES.items():
        print(f"\n--- {name} ---")

        dest = OUTPUT_DIR / config["filename"]

        # 检查是否已存在
        if dest.exists():
            print(f"⏭️  文件已存在，跳过: {dest.name}")
            success_count += 1
            continue

        # 下载
        success = download_file(
            url=config["url"],
            dest=dest,
            expected_sha256=config.get("sha256")
        )

        if success:
            success_count += 1
        else:
            fail_count += 1

    # 总结
    print("\n" + "=" * 60)
    print(f"下载完成: {success_count} 成功, {fail_count} 失败")

    if fail_count == 0:
        print("\n✓ 所有数据下载成功!")
        print("\n下一步:")
        print("1. 在 data/README.md 中记录数据来源和版本")
        print("2. 运行预处理脚本（如有）")
    else:
        print("\n❌ 部分数据下载失败，请检查网络连接后重试")


if __name__ == "__main__":
    main()
