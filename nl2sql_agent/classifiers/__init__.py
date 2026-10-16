"""
分类器模块 —— 难点B 的难度分级 + Few-shot 选择。

包含:
  - DifficultyClassifier: 轻量 LLM 预分类（带规则兜底）
  - FewShotManager: 示例池管理 + 动态选择
"""
from nl2sql_agent.interfaces.difficulty_classifier import (
    IDifficultyClassifier,
    LLMDifficultyClassifier,
    RuleBasedDifficultyClassifier,
)
from nl2sql_agent.interfaces.fewshot_selector import (
    IFewShotSelector,
    TagBasedFewShotSelector,
)

__all__ = [
    "IDifficultyClassifier", "LLMDifficultyClassifier",
    "RuleBasedDifficultyClassifier",
    "IFewShotSelector", "TagBasedFewShotSelector",
]
