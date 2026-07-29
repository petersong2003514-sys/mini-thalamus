"""
接口 1: ISchemaProvider —— Schema 提供者

根据用户问题返回相关的数据库 Schema 信息。
核心设计（难点A）：
  - 小库(≤10表): FullSchemaProvider     — 全量 DDL 注入，零检索开销
  - 中库(10-100表): KeywordSchemaProvider — 关键词匹配，轻量快速
  - 大库(100+表):  HybridSchemaProvider   — Keyword粗筛 + Embedding精排

  Agent 只依赖 ISchemaProvider 接口，不感知策略。
  后续加 Embedding 方案不改 Agent 一行代码。
"""

from abc import ABC, abstractmethod
from nl2sql_agent.schemas import SchemaInfo, TableMeta, ColumnMeta, SchemaRetrievalResult
from nl2sql_agent.agent.prompts import safe_format


class ISchemaProvider(ABC):
    """
    Schema 提供者抽象接口。

    三种策略通过同一接口调用:
      provider = FullSchemaProvider(schema_info)
      provider = KeywordSchemaProvider(schema_info)
      provider = HybridSchemaProvider(schema_info, embedding_client=None)  # 预留
      ddl = provider.retrieve("销售额最高的商品是哪个？")
    """

    @abstractmethod
    def retrieve(self, question: str, top_k_tables: int = 5,
                 top_k_columns: int = 15) -> SchemaRetrievalResult:
        """
        根据用户问题返回相关的 Schema 信息。

        Args:
            question: 用户自然语言问题
            top_k_tables: 最多返回几张表
            top_k_columns: 最多返回几个列

        Returns:
            SchemaRetrievalResult: 包含 DDL 文本 + 匹配的表/列列表 + 策略名
        """
        ...

    @abstractmethod
    def search_columns(self, keyword: str, top_k: int = 10) -> list[ColumnMeta]:
        """
        按关键词搜索列 —— 供工具层的 search_column 工具使用。

        Args:
            keyword: 搜索关键词（中文或英文）
            top_k: 返回前 K 个结果
        """
        ...

    @abstractmethod
    def get_table_ddl(self, table_name: str) -> str:
        """返回单张表的完整 DDL"""
        ...

    @abstractmethod
    def get_all_tables(self) -> list[str]:
        """返回所有表名"""
        ...

    @property
    @abstractmethod
    def strategy_name(self) -> str:
        """策略名称: 'full' | 'keyword' | 'hybrid'"""
        ...

    @property
    @abstractmethod
    def schema_info(self) -> SchemaInfo:
        """返回底层的完整 SchemaInfo（供高级功能使用）"""
        ...


# ═══════════════════════════════════════════════════════════════
#  FullSchemaProvider — Demo 默认实现（≤10 表）
# ═══════════════════════════════════════════════════════════════

class FullSchemaProvider(ISchemaProvider):
    """
    全量 Schema 注入策略。

    对所有问题返回全部表的 DDL（含列注释+外键+示例值+跨表提示）。
    适合小规模数据库（≤10 张表）。
    优点: 零检索开销、不丢信息、Schema 变更即时生效。
    缺点: Token 消耗随表数线性增长。
    """

    def __init__(self, schema_info: SchemaInfo):
        self._schema_info = schema_info
        self._full_ddl = schema_info.to_ddl()  # 含完整注释的 DDL

    def retrieve(self, question: str, top_k_tables: int = 5,
                 top_k_columns: int = 15) -> SchemaRetrievalResult:
        all_tables = [t.name for t in self._schema_info.tables]
        all_columns = [
            f"{t.name}.{c.name}"
            for t in self._schema_info.tables
            for c in t.columns
        ]
        return SchemaRetrievalResult(
            ddl_text=self._full_ddl,
            matched_tables=all_tables,
            matched_columns=all_columns,
            retrieval_method="full",
        )

    def search_columns(self, keyword: str, top_k: int = 10) -> list[ColumnMeta]:
        keyword_lower = keyword.lower()
        results = []
        for table in self._schema_info.tables:
            for col in table.columns:
                score = 0
                if keyword_lower in col.name.lower():
                    score += 2
                if keyword_lower in col.description.lower():
                    score += 1
                if score > 0:
                    results.append((score, col))
        results.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in results[:top_k]]

    def get_table_ddl(self, table_name: str) -> str:
        for t in self._schema_info.tables:
            if t.name == table_name:
                return self._schema_info._table_ddl(t)
        return f"Table '{table_name}' not found"

    def get_all_tables(self) -> list[str]:
        return [t.name for t in self._schema_info.tables]

    @property
    def strategy_name(self) -> str:
        return "full"

    @property
    def schema_info(self) -> SchemaInfo:
        return self._schema_info


# ═══════════════════════════════════════════════════════════════
#  预留: KeywordSchemaProvider + HybridSchemaProvider
#  (Phase 5 只定义接口和签名，具体实现在后续 Phase 完成)
# ═══════════════════════════════════════════════════════════════

class KeywordSchemaProvider(ISchemaProvider):
    """
    关键词匹配策略（10-100 张表时使用）。

    将 question 分词 → 和每张表/每列的 description 做关键词匹配打分
    → 返回 top-K 相关表和列的 DDL。

    优点: 零外部依赖、速度快(<1ms)。
    缺点: 同义词盲区（"货"≠"商品"）、需要维护高质量的 description。
    """

    # 停用词（高频但无信息量的词）
    _STOP_WORDS = {'的', '是', '了', '在', '有', '和', '与', '或', '及',
                   '请', '帮', '我', '你', '把', '被', '给', '从', '到',
                   '哪个', '哪些', '什么', '怎么', '如何', '多少', '吗',
                   '呢', '吧', '啊', '哦', '嗯', '这', '那', '一个', '一下'}

    def __init__(self, schema_info: SchemaInfo):
        self._schema_info = schema_info

    def _tokenize(self, text: str) -> list[str]:
        """中文分词——简单版：字符 2-gram + 英文单词"""
        import re

        tokens = []
        # 提取英文单词
        eng_words = re.findall(r'[a-zA-Z_]\w*', text)
        tokens.extend(w.lower() for w in eng_words)

        # 中文: 提取连续的汉字，生成 2-gram
        chinese = re.findall(r'[一-鿿]+', text)
        for seg in chinese:
            # 2-gram（覆盖面更好）
            for i in range(len(seg) - 1):
                bigram = seg[i:i+2]
                if bigram not in self._STOP_WORDS:
                    tokens.append(bigram)
            # 单字（兜底: 品类/城市名可能是单字常用）
            for ch in seg:
                if ch not in self._STOP_WORDS:
                    tokens.append(ch)

        return tokens

    def retrieve(self, question: str, top_k_tables: int = 5,
                 top_k_columns: int = 15) -> SchemaRetrievalResult:
        """关键词匹配检索"""
        tokens = self._tokenize(question)
        if not tokens:
            return SchemaRetrievalResult(
                ddl_text=self._schema_info.to_ddl(),
                matched_tables=[t.name for t in self._schema_info.tables],
                matched_columns=[],
                retrieval_method="keyword",
            )

        # 为每张表打分
        table_scores = []
        for table in self._schema_info.tables:
            score = 0
            matched_cols = []

            # 表名匹配（权重最高——表名是强信号）
            for token in tokens:
                if token in table.name.lower():
                    score += 5
                if table.description and token in table.description:
                    score += 3  # 中文描述匹配（如"用户信息表"↔"用户"）

            # 列名匹配
            for col in table.columns:
                col_score = 0
                for token in tokens:
                    if token in col.name.lower():
                        col_score += 3
                    if col.description and token in col.description:
                        col_score += 1  # 列描述匹配（如"用户ID"↔"用户"）
                if col_score > 0:
                    matched_cols.append((col_score, col.name))

            if matched_cols:
                score += sum(cs for cs, _ in matched_cols)

            # 高分列也间接说明表相关 → 加分，但不让列分主导
            if score > 0:
                table_scores.append((score, table, matched_cols))

        # 按分数降序排序
        table_scores.sort(key=lambda x: x[0], reverse=True)

        if not table_scores:
            # 无匹配 → 返回全部（兜底）
            return SchemaRetrievalResult(
                ddl_text=self._schema_info.to_ddl(),
                matched_tables=[t.name for t in self._schema_info.tables],
                matched_columns=[],
                retrieval_method="keyword",
            )

        # Top-K 表
        top_tables = table_scores[:top_k_tables]

        # 构建精简 DDL: 只包含匹配到的表 + 匹配到的列
        ddl_parts = [f"-- 关键词匹配策略 | 查询词: {question}", ""]
        all_matched_cols = []

        for _, table, matched_cols in top_tables:
            ddl_parts.append(self._keyword_table_ddl(table, matched_cols, top_k_columns))
            ddl_parts.append("")
            for cs, cname in matched_cols[:top_k_columns]:
                all_matched_cols.append(f"{table.name}.{cname}")

        ddl_text = "\n".join(ddl_parts)

        return SchemaRetrievalResult(
            ddl_text=ddl_text,
            matched_tables=[t.name for _, t, _ in top_tables],
            matched_columns=all_matched_cols[:top_k_columns],
            retrieval_method="keyword",
        )

    def _keyword_table_ddl(self, table: TableMeta,
                           matched_cols: list, max_cols: int) -> str:
        """生成单张表的精简 DDL——只展示匹配到的列 + 全部列名概览"""
        matched_set = {mc[1] for mc in matched_cols[:max_cols]}

        lines = []
        if table.description:
            lines.append(f"-- {table.description}")
        lines.append(f"-- 行数: {table.row_count}")
        lines.append(f"CREATE TABLE {table.name} (")

        col_lines = []
        for col in table.columns:
            markers = []
            comment = ""
            if col.name in matched_set:
                markers.append("★")  # 匹配标记
            if col.description:
                comment = f"  -- {col.description}"
            marker = " ".join(markers)
            col_lines.append(f"  {col.name} {col.data_type}{' ' + marker if marker else ''}{comment}")

        lines.append(",\n".join(col_lines))
        lines.append(");")

        # 所有列概览（帮助 LLM 了解全貌）
        all_cols = ", ".join(c.name for c in table.columns)
        lines.append(f"-- 全部列: {all_cols}")

        hints = self._schema_info._usage_hints(table.name)
        if hints:
            lines.append(f"-- [提示] {hints}")

        return "\n".join(lines)

    def search_columns(self, keyword: str, top_k: int = 10) -> list[ColumnMeta]:
        """按关键词搜索列名"""
        tokens = self._tokenize(keyword)
        results = []

        for table in self._schema_info.tables:
            for col in table.columns:
                score = 0
                for token in tokens:
                    if token in col.name.lower():
                        score += 3
                    if col.description and token in col.description:
                        score += 2
                if score > 0:
                    results.append((score, col))

        results.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in results[:top_k]]

    def get_table_ddl(self, table_name: str) -> str:
        """返回单张表的完整 DDL（使用 SchemaInfo 的丰富格式）"""
        for t in self._schema_info.tables:
            if t.name == table_name:
                return self._schema_info._table_ddl(t)
        return f"Table '{table_name}' not found"

    def get_all_tables(self) -> list[str]:
        return [t.name for t in self._schema_info.tables]

    @property
    def strategy_name(self) -> str:
        return "keyword"

    @property
    def schema_info(self) -> SchemaInfo:
        return self._schema_info


class HybridSchemaProvider(ISchemaProvider):
    """
    混合检索策略（100+ 张表时使用）。

    两阶段:
      Stage 1 — Keyword 粗筛: 关键词匹配快速过滤，高召回（2x top_k）
      Stage 2 — LLM 语义精排: 单次 LLM 调用对候选表打分排序，高精度

    优点: 兼顾召回率和精度，对同义词鲁棒。
          纯 LLM 实现，零外部 Embedding 依赖。
    """

    # LLM 精排 Prompt
    RERANK_PROMPT = """你是数据库专家。用户要查询数据，请评估每张候选表与问题的相关度。

## 用户问题
{question}

## 候选表及其说明
{table_descriptions}

## 任务
对每张表的相关度打分(1-10)，考虑:
- 表名/描述是否与问题中的概念直接相关
- 查询该问题是否需要用到这张表的列
- 10分=必需, 5分=可能有关, 1分=无关

## 输出格式
{{"scores": {{"表名1": 分数, "表名2": 分数, ...}}, "reasoning": "一句话理由"}}
只输出JSON。"""

    def __init__(self, schema_info: SchemaInfo, llm=None):
        self._schema_info = schema_info
        self._llm = llm  # LLM instance for reranking
        self._keyword_provider = KeywordSchemaProvider(schema_info)

    def retrieve(self, question: str, top_k_tables: int = 5,
                 top_k_columns: int = 15) -> SchemaRetrievalResult:
        # Stage 1: Keyword 粗筛（高召回，取 2x 候选）
        coarse = self._keyword_provider.retrieve(
            question, top_k_tables=top_k_tables * 2, top_k_columns=top_k_columns * 2
        )

        # 候选少 → 不需要精排
        if len(coarse.matched_tables) <= top_k_tables or self._llm is None:
            return SchemaRetrievalResult(
                ddl_text=coarse.ddl_text,
                matched_tables=coarse.matched_tables[:top_k_tables],
                matched_columns=coarse.matched_columns[:top_k_columns],
                retrieval_method="hybrid (keyword only, no LLM)",
            )

        # Stage 2: LLM 语义精排
        return self._llm_rerank(question, coarse, top_k_tables, top_k_columns)

    def _llm_rerank(self, question: str, coarse: SchemaRetrievalResult,
                    top_k_tables: int, top_k_columns: int) -> SchemaRetrievalResult:
        """用 LLM 对候选表做语义级重排序"""
        import json

        # 构建候选表描述
        table_descs = []
        candidate_tables = {}
        for t in self._schema_info.tables:
            if t.name in coarse.matched_tables:
                desc = f"{t.name}"
                if t.description:
                    desc += f" ({t.description})"
                desc += f" [{t.row_count}行]"
                table_descs.append(desc)
                candidate_tables[t.name] = t

        prompt = safe_format(self.RERANK_PROMPT,
            question=question,
            table_descriptions="\n".join(table_descs),
        )
        messages = [
            {"role": "system", "content": "你是数据库专家。只输出JSON。"},
            {"role": "user", "content": prompt},
        ]
        try:
            response = self._llm.generate(messages)
        except Exception:
            # LLM 调用失败 → 回退到 keyword-only 粗筛结果
            coarse.retrieval_method = "hybrid (rerank failed, fallback to keyword)"
            return coarse

        # 解析 LLM 评分
        try:
            text = response.strip()
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            data = json.loads(text)
            scores = data.get("scores", {})
            reasoning = data.get("reasoning", "")
        except json.JSONDecodeError:
            # LLM 输出不可解析 → 退回 keyword 结果
            coarse.retrieval_method = "hybrid (rerank failed, fallback to keyword)"
            return coarse

        # 按 LLM 分数排序
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_names = [name for name, _ in ranked[:top_k_tables] if name in candidate_tables]

        # 生成精排 DDL
        ddl_parts = [
            f"-- 混合检索 (Keyword粗筛 + LLM精排) | 查询: {question}",
            f"-- LLM判断: {reasoning}",
            "",
        ]
        for name in top_names:
            table = candidate_tables[name]
            ddl_parts.append(self._schema_info._table_ddl(table))
            ddl_parts.append("")

        # 精排后的匹配列
        all_matched_cols = []
        for name in top_names:
            for col_name in coarse.matched_columns:
                if col_name.startswith(name + "."):
                    all_matched_cols.append(col_name)

        return SchemaRetrievalResult(
            ddl_text="\n".join(ddl_parts),
            matched_tables=top_names,
            matched_columns=all_matched_cols[:top_k_columns],
            retrieval_method="hybrid",
        )

    def search_columns(self, keyword: str, top_k: int = 10) -> list:
        return self._keyword_provider.search_columns(keyword, top_k)

    def get_table_ddl(self, table_name: str) -> str:
        for t in self._schema_info.tables:
            if t.name == table_name:
                return self._schema_info._table_ddl(t)
        return f"Table '{table_name}' not found"

    def get_all_tables(self) -> list[str]:
        return [t.name for t in self._schema_info.tables]

    @property
    def strategy_name(self) -> str:
        return "hybrid"

    @property
    def schema_info(self) -> SchemaInfo:
        return self._schema_info


# ═══════════════════════════════════════════════════════════════
#  工具函数
# ═══════════════════════════════════════════════════════════════
# _table_to_ddl 已移除 —— 统一使用 SchemaInfo._table_ddl()（含外键+行数+提示）
