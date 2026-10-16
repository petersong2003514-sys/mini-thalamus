"""
接口 4: IFewShotSelector —— Few-shot 示例选择器（难点B）

根据难度分级结果 + 技能标签，从示例池中动态选择最相关的 Few-shot 示例。

设计依据（你的决策）:
  - 不是静态 10 条全注入 → 而是根据 DifficultyClassification 动态选 3 条
  - 每条示例都有 tags（difficulty / skills / tables）
  - 选择策略: 同难度优先 + 技能标签匹配 + 表名重叠加分
  - 选择器做成接口，后续可替换为 Embedding 相似度选择
"""

from abc import ABC, abstractmethod
from nl2sql_agent.schemas import FewShotExample, DifficultyClassification
from nl2sql_agent.config import Difficulty, SQLSkill


class IFewShotSelector(ABC):
    """
    Few-shot 示例选择器抽象接口。

    调用时机: Agent 循环的 Prompt 构造阶段
    输入: DifficultyClassification（来自 IDifficultyClassifier）
          + 用户原始问题
    输出: 选中的 Top-K 条 FewShotExample
    """

    @abstractmethod
    def select(self, question: str,
               classification: DifficultyClassification,
               top_k: int = 3) -> list[FewShotExample]:
        """
        根据难度分级结果，从示例池中选择最匹配的 Top-K 条示例。

        NOTE: `question` 参数当前未参与打分逻辑（仅基于 classification 标签匹配），
        预留供未来 embedding-based 语义相似度选择使用。
        """
        ...

    @abstractmethod
    def get_all_examples(self) -> list[FewShotExample]:
        """返回所有示例（供调试和评测用）"""
        ...

    @abstractmethod
    def add_example(self, example: FewShotExample) -> None:
        """动态添加一条示例（热插拔）"""
        ...


# ═══════════════════════════════════════════════════════════════
#  TagBasedFewShotSelector — 基于标签匹配的选择器
# ═══════════════════════════════════════════════════════════════

class TagBasedFewShotSelector(IFewShotSelector):
    """
    基于标签匹配的 Few-shot 选择器。

    打分规则:
      - difficulty 相同:            +3 分
      - 每匹配一个 skill 标签:       +2 分
      - 每匹配一个 table 名:         +1 分
      - difficulty 相邻级别（easy↔medium or medium↔hard）: +1 分

    选 Top-K 得分最高的示例。
    """

    def __init__(self, example_pool: list[FewShotExample] | None = None):
        self._pool: list[FewShotExample] = example_pool or []

    def select(self, question: str,
               classification: DifficultyClassification,
               top_k: int = 3) -> list[FewShotExample]:
        if not self._pool:
            return []

        scored = []
        for ex in self._pool:
            score = 0

            # 难度匹配
            if ex.difficulty == classification.difficulty:
                score += 3
            elif self._adjacent_difficulty(ex.difficulty, classification.difficulty):
                score += 1

            # Skill 标签匹配
            ex_skill_set = set(s.value for s in ex.skills)
            q_skill_set = set(s.value for s in classification.skills)
            score += len(ex_skill_set & q_skill_set) * 2

            # 表名重叠
            ex_table_set = set(ex.tables)
            q_table_set = set(classification.tables_needed)
            score += len(ex_table_set & q_table_set) * 1

            scored.append((score, ex))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [ex for _, ex in scored[:top_k]]

    def get_all_examples(self) -> list[FewShotExample]:
        return self._pool

    def add_example(self, example: FewShotExample) -> None:
        self._pool.append(example)

    @staticmethod
    def _adjacent_difficulty(a: Difficulty, b: Difficulty) -> bool:
        """判断两个难度是否相邻"""
        order = {Difficulty.EASY: 0, Difficulty.MEDIUM: 1, Difficulty.HARD: 2}
        return abs(order.get(a, 0) - order.get(b, 0)) == 1
