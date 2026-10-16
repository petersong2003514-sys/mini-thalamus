"""
接口 3: IDifficultyClassifier —— 难度分级器（难点B）

在 Agent 循环开始前，用一个轻量 LLM 对问题进行难度分级和技能标签预测。
分级结果驱动后续的 Few-shot 选择策略。

设计依据（你的决策）:
  - 不用全量 LLM（主模型），节省 Token 和延迟
  - 用轻量模型只做分类任务，输出结构化标签
  - 标签包括: difficulty / skills / tables_needed
  - 分类结果注入 Agent 上下文，指导 Few-shot 选择和纠错策略
"""

from abc import ABC, abstractmethod
from nl2sql_agent.schemas import DifficultyClassification
from nl2sql_agent.config import Difficulty, SQLSkill
from nl2sql_agent.agent.prompts import safe_format


class IDifficultyClassifier(ABC):
    """
    难度分级器抽象接口。

    调用时机: Agent 循环开始前（Step 0）
    输入: 用户自然语言问题
    输出: DifficultyClassification（难度 + 技能标签 + 所需表名）
    """

    @abstractmethod
    def classify(self, question: str) -> DifficultyClassification:
        """
        对问题进行难度分级和技能标签预测。

        返回的 DifficultyClassification 包含:
          - difficulty: EASY / MEDIUM / HARD
          - skills: 需要哪些 SQL 技能（JOIN / GROUP_BY / SUBQUERY...）
          - tables_needed: 预估需要哪些表
          - reasoning: 判定理由（可解释性，报告用）
        """
        ...


# ═══════════════════════════════════════════════════════════════
#  LLMDifficultyClassifier — 用轻量模型做分类
# ═══════════════════════════════════════════════════════════════

class LLMDifficultyClassifier(IDifficultyClassifier):
    """
    基于轻量 LLM 的难度分级器。

    用一个简短的 Prompt 让 LLM 做结构化分类（非生成 SQL），
    Token 消耗极小（<200 input tokens），可以使用更便宜的模型。
    """

    CLASSIFY_PROMPT = """你是一个 SQL 难度评估专家。分析以下用户问题，输出结构化分类。

## 难度定义
- easy: 单表查询，简单 WHERE 条件，无聚合/排序
- medium: 单表聚合(GROUP BY/HAVING/ORDER BY/LIMIT/日期过滤)，或两表JOIN
- hard: 三表及以上 JOIN，或嵌套子查询，或多步推理

## 技能标签(可多选)
single_table / join / aggregation / group_by / order_by /
subquery / having / date_filter / like / limit

## 输出格式
{{
  "difficulty": "easy|medium|hard",
  "skills": ["join", "aggregation"],
  "tables_needed": ["users", "orders"],
  "reasoning": "一句话说明判定理由"
}}

## 用户问题
{question}

## 你的分类（只输出JSON，不要其他内容）"""

    _SKILL_MAP = {s.value: s for s in SQLSkill}

    def __init__(self, llm):
        """
        Args:
            llm: 轻量 LLM 实例（如 gpt-3.5-turbo / deepseek-chat 的 low-tier）
                 Phase 5 先复用主模型，后续可替换为更便宜的模型
        """
        self._llm = llm

    def classify(self, question: str) -> DifficultyClassification:
        import json

        prompt = safe_format(self.CLASSIFY_PROMPT, question=question)
        messages = [
            {"role": "system", "content": "你是 SQL 难度评估专家。只输出 JSON。"},
            {"role": "user", "content": prompt},
        ]
        response = self._llm.generate(messages)

        # 从 LLM 输出中提取 JSON
        try:
            # 去掉可能的 markdown 围栏
            text = response.strip()
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            data = json.loads(text)
        except json.JSONDecodeError:
            # 兜底：默认为 medium
            return DifficultyClassification(
                difficulty=Difficulty.MEDIUM,
                skills=[SQLSkill.SINGLE_TABLE],
                tables_needed=[],
                reasoning="JSON 解析失败，使用默认分类",
            )

        # 解析 skill 字符串 → SQLSkill 枚举
        skills = []
        for s in data.get("skills", []):
            if s in self._SKILL_MAP:
                skills.append(self._SKILL_MAP[s])

        # 解析 difficulty
        diff_map = {d.value: d for d in Difficulty}
        difficulty = diff_map.get(data.get("difficulty", "medium"), Difficulty.MEDIUM)

        return DifficultyClassification(
            difficulty=difficulty,
            skills=skills,
            tables_needed=data.get("tables_needed", []),
            reasoning=data.get("reasoning", ""),
        )


# ═══════════════════════════════════════════════════════════════
#  预留: 规则兜底分类器（不依赖 LLM，但不够智能）
# ═══════════════════════════════════════════════════════════════

class RuleBasedDifficultyClassifier(IDifficultyClassifier):
    """
    基于关键词规则的兜底分类器。

    在 LLM 调用失败或不需要 LLM 的低成本场景下使用。
    规则覆盖不如 LLM 精准，但零 Token 消耗、零延迟。
    """

    # 多表信号词（命中 → HARD）
    _MULTI_TABLE_SIGNALS = [
        '还买过', '同时买', '和.*一起', '既.*又',
        '关联', '对应', '所属',
    ]
    # 聚合信号词（命中 → MEDIUM+）
    _AGG_SIGNALS = [
        '每个', '每类', '每种', '平均', '总共', '一共',
        '多少', '几个', '排名', '最多', '最少', '最高', '最低',
        '总和', '总计', '数量', '人数', '个数',
    ]
    # 子查询信号词（命中 → HARD）
    _SUBQUERY_SIGNALS = [
        '从未', '没有.*过', '高于平均', '低于平均',
        '超过平均', '大于平均',
    ]

    def classify(self, question: str) -> DifficultyClassification:
        import re

        difficulty = Difficulty.EASY
        skills = []
        tables_needed = []

        # ① 子查询检测
        for signal in self._SUBQUERY_SIGNALS:
            if re.search(signal, question):
                difficulty = Difficulty.HARD
                skills.append(SQLSkill.SUBQUERY)
                break

        # ② 多表检测
        for signal in self._MULTI_TABLE_SIGNALS:
            if re.search(signal, question):
                difficulty = Difficulty.HARD
                skills.append(SQLSkill.JOIN)
                break

        # ③ 聚合检测
        has_agg = False
        for signal in self._AGG_SIGNALS:
            if re.search(signal, question):
                has_agg = True
                skills.append(SQLSkill.AGGREGATION)
                skills.append(SQLSkill.GROUP_BY)
                break

        if has_agg and difficulty == Difficulty.EASY:
            difficulty = Difficulty.MEDIUM

        # ④ 排序/LIMIT 检测
        if any(w in question for w in ['排名', '前', 'TOP', '排第', '最大', '最小']):
            skills.append(SQLSkill.ORDER_BY)
            skills.append(SQLSkill.LIMIT)

        if any(w in question for w in ['排序', '从高到低', '从低到高', '按']):
            skills.append(SQLSkill.ORDER_BY)

        # ⑤ 日期检测
        if any(w in question for w in ['年', '月', '日', '今天', '昨天', '最近']):
            skills.append(SQLSkill.DATE_FILTER)

        # ⑥ 模糊匹配检测
        if any(w in question for w in ['包含', '含有', '模糊', '搜索']):
            skills.append(SQLSkill.LIKE)

        # ⑦ 推测涉及的表（多个城市名也检测）
        table_keywords = {
            '商品': 'products', '产品': 'products', '价格': 'products',
            '品牌': 'products', '品类': 'products', '库存': 'products',
            '用户': 'users', '客户': 'users', '会员': 'users',
            '城市': 'users', '北京': 'users', '上海': 'users',
            '广州': 'users', '深圳': 'users', '杭州': 'users',
            '订单': 'orders', '买了': 'orders', '买得': 'orders', '买': 'orders',
            '购买': 'orders', '下单': 'orders',
            '销售额': 'order_items', '销量': 'order_items', '售出': 'order_items',
            '浏览': 'user_behaviors', '加购': 'user_behaviors',
            '收藏': 'user_behaviors', '搜索': 'user_behaviors',
            '购物车': 'user_behaviors', '行为': 'user_behaviors',
        }
        for kw, table in table_keywords.items():
            if kw in question and table not in tables_needed:
                tables_needed.append(table)

        # 如果涉及 3+ 张不同的表 → HARD
        if len(tables_needed) >= 3 and difficulty != Difficulty.HARD:
            difficulty = Difficulty.HARD
            if SQLSkill.JOIN not in skills:
                skills.append(SQLSkill.JOIN)

        # 去重 + 默认给 skills
        skills = list(set(skills))
        if not skills:
            skills = [SQLSkill.SINGLE_TABLE]

        return DifficultyClassification(
            difficulty=difficulty,
            skills=skills,
            tables_needed=tables_needed[:5],
            reasoning=f"规则分类: {' | '.join(s.value for s in skills)}",
        )
