# ============================================================================
# 学术研究项目工作流
# ============================================================================
#
# 使用: make help
#
# 核心命令:
#   make paper   - 生成论文（收集数据→图表→检查→渲染）
#   make check   - 运行检查（新鲜度 + 硬编码）
#
# ============================================================================

.PHONY: help all clean check paper train evaluate collect figures render

.DEFAULT_GOAL := help

PYTHON := python
PAPER_DIR := paper

# ============================================================================
# 帮助
# ============================================================================

help:
	@echo "学术研究项目 Makefile"
	@echo ""
	@echo "论文:"
	@echo "  make collect   - 收集实验结果"
	@echo "  make figures   - 生成图表"
	@echo "  make render    - 渲染论文"
	@echo "  make paper     - 完整论文流程"
	@echo ""
	@echo "检查:"
	@echo "  make check     - 运行所有检查"
	@echo "  make check-fresh   - 数据新鲜度"
	@echo "  make check-numbers - 硬编码数字"
	@echo ""
	@echo "实验:"
	@echo "  make train     - 训练"
	@echo "  make evaluate  - 评估"
	@echo ""
	@echo "其他:"
	@echo "  make clean     - 清理"

# ============================================================================
# 数据准备
# ============================================================================

download:
	$(PYTHON) scripts/download_data.py

preprocess:
	$(PYTHON) scripts/preprocess.py

# ============================================================================
# 实验
# ============================================================================

train:
	$(PYTHON) scripts/train.py

evaluate:
	$(PYTHON) scripts/evaluate.py

# ============================================================================
# 论文
# ============================================================================

collect:
	$(PYTHON) $(PAPER_DIR)/scripts/collect_results.py

figures:
	$(PYTHON) $(PAPER_DIR)/scripts/generate_figures.py

render:
	cd $(PAPER_DIR)/source && quarto render

# 完整论文流程
paper: collect figures check render
	@echo "✓ 论文生成完成"

# ============================================================================
# 检查
# ============================================================================

check-fresh:
	$(PYTHON) scripts/check_data_freshness.py

check-numbers:
	$(PYTHON) $(PAPER_DIR)/scripts/check_hardcoded_numbers.py

check: check-fresh check-numbers

# ============================================================================
# 清理
# ============================================================================

clean:
	rm -rf $(PAPER_DIR)/source/_site/
	rm -rf $(PAPER_DIR)/source/.quarto/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
