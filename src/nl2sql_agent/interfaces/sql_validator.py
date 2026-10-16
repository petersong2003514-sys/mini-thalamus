"""
接口 5: ISQLValidator —— SQL 智能预校验器（难点C）

在 SQL 发送到数据库执行之前，进行智能预校验。
不是简单的正则匹配——而是用轻量 LLM 做语义层面的列名/表名合规检查。

设计依据（你的决策）:
  正则校验太硬——"商业品"写成"商品表"就会被拦截，本质是同一个东西。
  需要 LLM 级别的语义理解: 识别用户列名是否和 Schema 中的列名语义一致，
  如果不一致，给出修正建议而非直接拒绝。

校验层次:
  Layer 1 (硬规则):  黑名单检查 / SELECT 检查 / 基本语法（<1ms）
  Layer 2 (白名单):  列名正则提取 + Schema 比对（<1ms）
  Layer 3 (LLM语义): 语义不一致时的模糊匹配 + 修正建议（仅 Layer 2 失败时触发）
"""

from abc import ABC, abstractmethod
from nl2sql_agent.schemas import ValidationResult, SchemaInfo


class ISQLValidator(ABC):
    """
    SQL 预校验器抽象接口。

    三层防御:
      Layer 1 — 硬规则（必过，不通过直接拒绝）
      Layer 2 — 白名单精确匹配
      Layer 3 — LLM 语义模糊匹配（Layer 2 失败时触发）
    """

    @abstractmethod
    def validate(self, sql: str, schema_info: SchemaInfo) -> ValidationResult:
        """
        对生成的 SQL 进行预校验。

        Args:
            sql: LLM 生成的 SQL 语句
            schema_info: 当前数据库的完整 Schema 信息

        Returns:
            ValidationResult:
              - is_valid=True: 通过校验，可以执行
              - is_valid=False: 未通过，包含 error + suggestion
        """
        ...


# ═══════════════════════════════════════════════════════════════
#  LayeredSQLValidator — 三层防御实现
# ═══════════════════════════════════════════════════════════════

class LayeredSQLValidator(ISQLValidator):
    """
    三层防御的 SQL 预校验器。

    流程:
      1. 硬规则检查（黑白名单 + SELECT 开头）
      2. 列名/表名白名单精确匹配
      3. 如果②失败 → 用轻量 LLM 做语义模糊匹配
    """

    def __init__(self, light_llm=None):
        """
        Args:
            light_llm: 轻量 LLM 实例，用于 Layer 3 语义匹配。
                       如果为 None，Layer 3 退化为纯正则匹配。
        """
        self._light_llm = light_llm

    def validate(self, sql: str, schema_info: SchemaInfo) -> ValidationResult:
        import re

        # ── Layer 1: 硬规则检查 ──
        sql_upper = sql.strip().upper()
        sql_clean = '\n'.join(
            line for line in sql_upper.split('\n')
            if not line.strip().startswith('--')
        )

        # 允许 SELECT 和 WITH (CTE) —— 都是只读查询
        if not (sql_clean.startswith("SELECT") or sql_clean.startswith("WITH")):
            return ValidationResult(
                is_valid=False,
                error="只允许 SELECT 或 WITH (CTE) 查询",
                suggestion="请生成 SELECT 或 WITH ... SELECT 语句",
            )

        # 黑名单检查
        from nl2sql_agent.config import SQL_BLACKLIST
        for keyword in SQL_BLACKLIST:
            if re.search(r'\b' + keyword + r'\b', sql_clean):
                return ValidationResult(
                    is_valid=False,
                    error=f"检测到禁止的操作: {keyword}",
                    suggestion=f"仅支持 SELECT 查询，请移除 {keyword} 语句",
                )

        # ── Layer 2: 白名单匹配 ──
        whitelist = schema_info.get_column_whitelist()  # {table: {col, ...}}

        # 方法: 提取 SQL 中所有 word.word 模式（table.column / alias.column）
        # 这比上下文正则更鲁棒——不依赖关键字位置
        all_dot_refs = re.findall(
            r'\b([a-zA-Z_]\w*)\.([a-zA-Z_]\w*)\b', sql_clean
        )

        # 收集所有合法的 {table}.{column} 组合
        all_qualified = set()
        all_bare_columns = set()
        for table_name, cols in whitelist.items():
            for col in cols:
                all_qualified.add(f"{table_name.lower()}.{col.lower()}")
                all_bare_columns.add(col.lower())

        unknown_refs = []
        seen_rejected = set()  # 去重

        for prefix, col_name in all_dot_refs:
            prefix_lower = prefix.lower()
            col_lower = col_name.lower()

            # 跳过已知的 SQL 关键字被误当表名
            if prefix_lower in {'sql', 'sqlite'}:
                continue

            # 检查 table.column 是否在合法集合中
            ref = f"{prefix_lower}.{col_lower}"
            if ref not in all_qualified:
                # prefix 是已知表名 → 严格校验：列是否属于该表
                if prefix_lower in whitelist:
                    if col_lower not in {c.lower() for c in whitelist[prefix_lower]}:
                        if ref not in seen_rejected:
                            unknown_refs.append(f"{prefix}.{col_name}")
                            seen_rejected.add(ref)
                else:
                    # 可能是表别名 → 宽松处理，只检查列是否存在
                    if col_lower not in all_bare_columns and ref not in seen_rejected:
                        unknown_refs.append(f"{prefix}.{col_name}")
                        seen_rejected.add(ref)

        if not unknown_refs:
            return ValidationResult(is_valid=True)

        # ── Layer 3: LLM 语义模糊匹配 ──
        if self._light_llm:
            return self._semantic_validate(sql, unknown_refs, schema_info)
        else:
            # 无 LLM → 直接拒绝 + 给最接近的列名建议
            suggestions = []
            for ref in unknown_refs:
                closest = self._fuzzy_match(ref, all_bare_columns)
                suggestions.append(f"'{ref}' 不存在，是否指 '{closest}'？")

            return ValidationResult(
                is_valid=False,
                error=f"未知的列引用: {', '.join(unknown_refs)}",
                suggestion="; ".join(suggestions),
            )

    def _semantic_validate(self, sql: str, unknown_refs: list[str],
                           schema_info: SchemaInfo) -> ValidationResult:
        """
        使用轻量 LLM 做语义级别的列名校验。

        关键思路（难点C）:
          "商业品" 不是正则能解决的——需要 LLM 理解"商业品"= "商品"= "products"
          LLM 判断: 这个列引用的语义意图是什么？Schema 中有语义相同的列吗？
        """
        whitelist = schema_info.get_column_whitelist()

        # 构建可用列清单
        available_cols = []
        for table, cols in whitelist.items():
            for col in cols:
                available_cols.append(f"{table}.{col}")

        prompt = f"""你是数据库专家。以下 SQL 中包含不存在于 Schema 中的列引用。

## SQL
{sql}

## 未知的列引用
{', '.join(unknown_refs)}

## 可用的列（表名.列名）
{', '.join(available_cols[:100])}

## 任务
对于每个未知列引用，判断:
1. 它是否是可用列的别名/别称/同义词/手误？
2. 如果是，给出正确的列名
3. 如果不是，标记为 "unknown"

## 输出格式
{{
  "corrections": {{
    "错误的列引用": "正确的 表.列",
    ...
  }},
  "unknown": ["完全无法匹配的列"]
}}

只输出 JSON，不要其他内容。"""

        messages = [
            {"role": "system", "content": "你是数据库列名语义匹配专家。只输出 JSON。"},
            {"role": "user", "content": prompt},
        ]
        response = self._light_llm.generate(messages)

        import json
        try:
            text = response.strip()
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            data = json.loads(text)
        except json.JSONDecodeError:
            # LLM 返回不可解析 → 退回模糊匹配
            return self._fallback_validate(unknown_refs, schema_info)

        corrections = data.get("corrections", {})
        unknowns = data.get("unknown", [])

        if unknowns and not corrections:
            return ValidationResult(
                is_valid=False,
                error=f"列引用不存在且无法语义匹配: {', '.join(unknowns)}",
                suggestion="请确认列名是否正确，或使用可用列清单中的列名",
            )

        if corrections:
            return ValidationResult(
                is_valid=False,
                error=f"列名需要修正",
                suggestion=f"建议修正: {corrections}",
                corrected_columns=corrections,
            )

        return ValidationResult(is_valid=True)

    def _fallback_validate(self, unknown_refs: list[str],
                           schema_info: SchemaInfo) -> ValidationResult:
        """无 LLM 时的模糊匹配回退"""
        whitelist = schema_info.get_column_whitelist()
        all_bare = set()
        for cols in whitelist.values():
            all_bare.update(c.lower() for c in cols)

        suggestions = []
        for ref in unknown_refs:
            closest = self._fuzzy_match(ref.split('.')[-1].lower(), all_bare)
            suggestions.append(f"'{ref}' 不存在，最接近的列: '{closest}'")

        return ValidationResult(
            is_valid=False,
            error=f"未知列引用: {', '.join(unknown_refs)}",
            suggestion="; ".join(suggestions),
        )

    @staticmethod
    def _fuzzy_match(target: str, candidates: set[str], max_distance: int = 3) -> str:
        """
        简单的编辑距离模糊匹配。
        用于 Layer 2 失败时的回退建议。
        """
        target = target.lower().strip("`\"'[]")
        best = None
        best_dist = float('inf')

        for cand in candidates:
            dist = LayeredSQLValidator._edit_distance(target, cand)
            if dist < best_dist:
                best_dist = dist
                best = cand

        return best if best and best_dist <= max_distance else target

    @staticmethod
    def _edit_distance(s1: str, s2: str) -> int:
        """Levenshtein 编辑距离"""
        if len(s1) < len(s2):
            return LayeredSQLValidator._edit_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)

        prev = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            curr = [i + 1]
            for j, c2 in enumerate(s2):
                curr.append(min(
                    curr[j] + 1,
                    prev[j + 1] + 1,
                    prev[j] + (0 if c1 == c2 else 1),
                ))
            prev = curr

        return prev[-1]
