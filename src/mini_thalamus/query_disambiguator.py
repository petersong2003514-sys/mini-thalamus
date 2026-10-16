"""SmartDisambiguator — V2  Schema 感知 + V3 LLM 消歧 混合版

V2: 读真实 DB schema → 匹配问题关键词和列的中文描述 → 1:1 匹配则直接消歧
V3: V2 产生多个候选 → 轻量 LLM 调用 → 选最佳列 → 追加排序提示到 query
"""

from openai import OpenAI
from mini_thalamus.config import get_llm_config

# ─── Schema-aware mapping (V2) ───

# 模糊词 → 可能对应的数据库列名 + 中文描述
AMBIGUITY_MAP = {
    "最高":  ["total_amount", "subtotal", "price", "unit_price", "quantity", "cost"],
    "最好":  ["total_amount", "subtotal", "price", "quantity", "stock"],
    "最多":  ["quantity", "stock", "total_amount", "subtotal"],
    "最低":  ["price", "cost", "unit_price", "discount_amount"],
    "增长":  ["total_amount", "subtotal", "quantity"],
    "快":    ["total_amount", "subtotal", "quantity"],
    "大":    ["total_amount", "subtotal", "quantity", "stock"],
    "贵":    ["price", "unit_price", "total_amount"],
    "便宜":  ["price", "unit_price", "cost"],
    "收入":  ["total_amount", "subtotal"],
    "利润":  ["subtotal", "cost"],
    "库存":  ["stock"],
    "销量":  ["quantity"],
}

# 数据库列 → 用户友好的解释
COLUMN_MEANING = {
    "quantity":        "按销售件数排序",
    "subtotal":        "按销售金额排序",
    "total_amount":    "按订单金额排序",
    "price":           "按商品单价排序",
    "unit_price":      "按商品单价排序",
    "cost":            "按成本价排序",
    "stock":           "按库存数量排序",
    "discount_amount": "按折扣金额排序",
    "duration_sec":    "按浏览时长排序",
}


class SmartDisambiguator:
    """V2 Schema 感知 + V3 LLM 消歧"""

    def __init__(self, db_connector, llm_client=None):
        self.db = db_connector

        llm_config = get_llm_config()

        self.llm = llm_client or OpenAI(
            api_key=llm_config["api_key"],
            base_url=llm_config["base_url"],
        )
        self.model = llm_config["model"]

        # 缓存：同一 query 不同次调用保持一致
        self._cache = {}

    def disambiguate(self, question: str) -> dict:
        # 缓存命中
        cache_key = question.strip()
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Step 1: 检测模糊词
        matched_term = None
        for term in AMBIGUITY_MAP:
            if term in question:
                matched_term = term
                break

        if not matched_term:
            result = {"action": "pass_through", "query": question}
            self._cache[cache_key] = result
            return result

        # Step 2: V2 — Schema 感知匹配
        candidates = AMBIGUITY_MAP[matched_term]
        schema_columns = self._get_numeric_columns()
        matched_columns = [c for c in candidates if c in schema_columns]

        if not matched_columns:
            # 没有匹配的 schema 列 → 跳过消歧
            result = {"action": "pass_through", "query": question}
            self._cache[cache_key] = result
            return result

        if len(matched_columns) == 1:
            # 只有一个候选 → V2 直接消歧
            col = matched_columns[0]
            hint = COLUMN_MEANING.get(col, f"按{col}排序")
            result = {
                "action": "enhance",
                "enhanced_query": f"{question}（请{hint}）",
                "resolved": hint,
                "method": "V2_schema",
                "matched_term": matched_term,
                "column": col,
            }
            self._cache[cache_key] = result
            return result

        # Step 3: V3 — 多个候选 → LLM 消歧（有缓存）
        col = self._llm_pick(question, matched_term, matched_columns)
        hint = COLUMN_MEANING.get(col, f"按{col}排序")
        result = {
            "action": "enhance",
            "enhanced_query": f"{question}（请{hint}）",
            "resolved": hint,
            "method": "V3_llm",
            "matched_term": matched_term,
            "column": col,
        }
        self._cache[cache_key] = result
        return result

    # ─── V2: 读真实 Schema ───

    def _get_numeric_columns(self) -> set:
        """从数据库提取所有数值列名"""
        try:
            info = self.db.get_schema_info()
            cols = set()
            for table in info.tables:
                for col in table.columns:
                    if col.data_type in ("INTEGER", "REAL", "FLOAT", "NUMERIC"):
                        cols.add(col.name)
            return cols
        except Exception:
            return set()  # empty: disambiguation falls through cleanly

    # ─── V3: LLM 消歧 ───

    def _llm_pick(self, question: str, term: str, candidates: list) -> str:
        """轻量 LLM — 从多个候选中选最合适的列"""
        clues = "\n".join(f"  - {c}: {COLUMN_MEANING.get(c, c)}" for c in candidates)
        prompt = f"""用户的查询中有模糊词"{term}"。这个模糊词可能对应数据库中的以下指标:

{clues}

用户查询: "{question}"

请选择最符合用户意图的指标。只输出指标名（如 quantity），不要解释。"""

        try:
            r = self.llm.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=20,
            )
            choice = r.choices[0].message.content.strip().lower()
            if choice in candidates:
                return choice
        except Exception:
            pass

        # LLM 失败 → 第一个候选（最常用的）
        return candidates[0]
