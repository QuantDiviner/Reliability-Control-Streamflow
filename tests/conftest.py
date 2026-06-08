"""
Pytest 配置和共享 fixtures

在此定义所有测试共享的 fixtures 和配置。
"""

import pytest
import sys
from pathlib import Path

# 将项目根目录添加到 Python 路径
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))


# === Fixtures ===

@pytest.fixture
def project_root():
    """返回项目根目录"""
    return ROOT_DIR


@pytest.fixture
def data_dir(project_root):
    """返回数据目录"""
    return project_root / "data"


@pytest.fixture
def sample_config():
    """返回示例配置"""
    return {
        "seed": 42,
        "batch_size": 32,
        "learning_rate": 0.001,
    }


# === 示例 fixtures（根据项目需要修改） ===

# @pytest.fixture
# def sample_data():
#     """加载测试用的样本数据"""
#     import pandas as pd
#     return pd.DataFrame({
#         "feature": [1, 2, 3],
#         "target": [0, 1, 0],
#     })


# @pytest.fixture
# def trained_model(sample_data):
#     """返回训练好的模型"""
#     from src.models import MyModel
#     model = MyModel()
#     model.fit(sample_data)
#     return model


# === Pytest 配置 ===

def pytest_configure(config):
    """Pytest 启动时的配置"""
    # 注册自定义标记
    config.addinivalue_line("markers", "slow: 标记为慢速测试")
    config.addinivalue_line("markers", "integration: 标记为集成测试")
