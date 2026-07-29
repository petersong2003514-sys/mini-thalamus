"""
接口 8: ITaskDecomposer —— 复杂任务拆解器

当单句 SQL 生成失败（如 CTE+LAG 超过 LLM 能力），
将问题拆解为 2-3 个独立子任务，逐步执行后汇总。

设计依据:
  复杂 SQL 失败根因 = LLM 认知负载过高
  拆解后每步子任务≤2表 JOIN → 每步独立可执行
  这和 Plan-Execute 模式一致
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from nl2sql_agent.agent.prompts import safe_format


@dataclass
class SubTask:
    """单个子任务"""
    step: int
    description: str          # 这步做什么
    question: str             # 简化的子问题
    sql: str = ""             # 生成的 SQL
    result: dict | None = None  # 执行结果


@dataclass
class DecomposedPlan:
    """拆解后的完整执行计划"""
    original_question: str
    subtasks: list[SubTask] = field(default_factory=list)
    reasoning: str = ""       # LLM 的拆解推理
    problem_type: str = "parallel"  # "sequential" | "parallel"
    cte_hint: str = ""              # CTE 链的 SQL 伪代码提示


class ITaskDecomposer(ABC):
    """任务拆解器抽象接口"""

    @abstractmethod
    def decompose(self, question: str, schema_ddl: str,
                  failed_attempts: list[str] | None = None) -> DecomposedPlan:
        """
        将复杂问题拆解为可独立执行的子任务。

        Args:
            question: 原始问题
            schema_ddl: 数据库 Schema
            failed_attempts: 之前失败的 SQL 尝试（供 LLM 分析哪部分太复杂）

        Returns:
            DecomposedPlan: 包含有序子任务列表
        """
        ...


class LLMTaskDecomposer(ITaskDecomposer):
    """
    使用 LLM 将复杂分析问题拆解为逐步执行的子任务。

    Prompt 设计要点:
      - 要求 LLM 先给出整体思路(reasoning)
      - 每步子任务必须是单表或双表 JOIN 可完成的
      - 子问题之间可以有依赖（后一步使用前一步的结果）
    """

    DECOMPOSE_PROMPT = """你是数据分析专家。以下问题太复杂，无法用一句SQL完成。
请判断问题类型，选择拆解策略:

## 类型判断
- **顺序依赖型**: 后一步需要用前一步的结果数据做精确计算（如: 队列留存、环比增长率、排名后筛选）
  → 用 CTE 链: 一个 SQL 里用 WITH 串联多步，最后 SELECT 输出
- **独立汇总型**: 各步独立查询不同维度，最后汇总对比（如: 多维指标对比、分组统计）
  → 用子任务: 拆为独立 SQL，各步结果文本汇总

## 数据库 Schema
{schema}

## 原始问题
{question}

{failed_hint}

## 拆解规则
{strategy_rules}

## 输出格式
{{
  "reasoning": "整体思路（一句话）",
  "type": "sequential" 或 "parallel",
  "cte_sql_hint": "如果是sequential，给出一条带WITH的完整CTE SQL思路（伪代码即可）",
  "subtasks": [
    {{
      "step": 1,
      "description": "这一步做什么",
      "question": "具体的子问题",
      "sql_hint": "建议的SQL写法（可选）"
    }}
  ]
}}

只输出JSON。"""

    PARALLEL_RULES = """1. 每步子任务只涉及1-2张表，一句简单SQL即可完成
2. 子任务之间独立，不依赖前面步骤的结果数据
3. 每个子任务的问题描述要具体、完整
4. 最后一步是"汇总以上结果，回答原始问题" """

    SEQUENTIAL_RULES = """1. 这个问题需要顺序计算——后一步依赖前一步的精确数据
2. 用 WITH 子句串联: WITH step1 AS (...), step2 AS (...), step3 AS (...) SELECT ... FROM step3
3. 每步 CTE 只做一件事，结果列命名清晰
4. 最后的 SELECT 输出最终答案
5. 每个 CTE 都可以独立执行和调试"""

    SYNTHESIS_PROMPT = """你是数据分析师。以下是逐步执行的结果，请汇总回答原始问题。

## 原始问题
{question}

## 逐步结果
{subtask_results}

## 要求
用1-2句话给出结论，包含关键数据。"""

    def __init__(self, llm):
        self._llm = llm

    def decompose(self, question: str, schema_ddl: str,
                  failed_attempts: list[str] | None = None) -> DecomposedPlan:
        import json

        failed_hint = ""
        if failed_attempts:
            failed_hint = "## 之前失败的SQL（说明单句SQL太复杂，需要拆解）\n"
            for i, sql in enumerate(failed_attempts[-2:], 1):
                failed_hint += f"失败{i}: {sql[:200]}\n"

        # 先用默认策略(parallel)尝试
        prompt = safe_format(self.DECOMPOSE_PROMPT,
            schema=schema_ddl,
            question=question,
            failed_hint=failed_hint,
            strategy_rules=self.PARALLEL_RULES,
        )

        messages = [
            {"role": "system", "content": "你是数据分析专家，擅长拆解复杂问题。只输出JSON。"},
            {"role": "user", "content": prompt},
        ]
        response = self._llm.generate(messages)

        try:
            text = response.strip()
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            data = json.loads(text)
        except json.JSONDecodeError:
            return DecomposedPlan(original_question=question, reasoning="拆解失败")

        subtasks = []
        for st in data.get("subtasks", []):
            subtasks.append(SubTask(
                step=st.get("step", 0),
                description=st.get("description", ""),
                question=st.get("question", ""),
            ))

        plan = DecomposedPlan(
            original_question=question,
            subtasks=sorted(subtasks, key=lambda s: s.step),
            reasoning=data.get("reasoning", ""),
            problem_type=data.get("type", "parallel"),
            cte_hint=data.get("cte_sql_hint", ""),
        )

        return plan

    def synthesize(self, question: str, results: list[dict]) -> str:
        """汇总子任务结果，生成最终答案"""
        result_text = ""
        for r in results:
            result_text += f"Step {r['step']}: {r['description']}\n"
            result_text += f"SQL: {r.get('sql', '')}\n"
            result_text += f"结果: {r.get('result_summary', '')}\n\n"

        prompt = safe_format(self.SYNTHESIS_PROMPT,
            question=question,
            subtask_results=result_text,
        )
        messages = [
            {"role": "system", "content": "你是数据分析师。"},
            {"role": "user", "content": prompt},
        ]
        return self._llm.generate(messages)
