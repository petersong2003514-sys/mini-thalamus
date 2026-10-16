"""
NL2SQL Agent 主循环 —— 将所有接口串联为完整的 Agent 工作流。

数据流（对应 Phase 3 的完整分析）:

  User Question
      │
      ▼
  ┌─────────────────────────┐
  │ Step 0: 难度分级         │  ← IDifficultyClassifier (难点B)
  │ 轻量 LLM → difficulty    │
  │ + skills + tables_needed │
  └───────────┬─────────────┘
              │
              ▼
  ┌─────────────────────────┐
  │ Step 1: Schema 检索      │  ← ISchemaProvider (难点A)
  │ 策略: full | keyword |   │
  │       hybrid             │
  └───────────┬─────────────┘
              │
              ▼
  ┌─────────────────────────┐
  │ Step 2: Few-shot 选择    │  ← IFewShotSelector (难点B)
  │ 按难度+技能标签选 Top-3  │
  └───────────┬─────────────┘
              │
              ▼
  ┌─────────────────────────┐
  │ Step 3: Prompt 构造      │  ← NL2SQL_SYSTEM_PROMPT + fewshot
  └───────────┬─────────────┘
              │
              ▼
  ╔═════════════════════════╗
  ║  Agent 纠错循环 (≤3轮)   ║
  ║                         ║
  ║  Step 4: LLM 生成 SQL   ║
  ║     ↓                   ║
  ║  Step 5: SQL 预校验      ║ ← ISQLValidator (难点C)
  ║     ↓                   ║
  ║  Step 6: SQL 执行        ║ ← IDatabaseConnector (难点D)
  ║     ↓                   ║
  ║  Step 7: 停止判断        ║ ← IStopJudge (难点F)
  ║     ↓                   ║
  ║  Step 8: 错误增强        ║ ← IErrorEnhancer (难点E)
  ║     → 回到 Step 4       ║
  ╚═════════════════════════╝
              │
              ▼
        AgentResult
"""

import re
import time
from nl2sql_agent.config import MAX_ROUNDS, MAX_RESULT_ROWS
from nl2sql_agent.schemas import (
    AgentResult, RoundTrace, DifficultyClassification,
    SQLResult, ValidationResult, StopJudgment,
    Difficulty, StopReason, ErrorType,
)
from nl2sql_agent.core.semantic_postcondition import (
    PostconditionResult,
    SemanticPostconditionChecker,
)
from nl2sql_agent.agent.prompts import safe_format, COT_CORRECTION_PROMPT, CORRECTION_PROMPT


class NL2SQLAgent:
    """
    NL2SQL Agent —— 核心编排器。

    依赖 7 个接口（全部可插拔）:
      - schema_provider:    Schema 检索策略
      - db_connector:       数据库连接
      - difficulty_classifier: 难度分级（可选，None=跳过）
      - fewshot_selector:   Few-shot 选择（可选，None=全量注入）
      - sql_validator:      SQL 预校验
      - error_enhancer:     错误增强
      - stop_judge:         停止判断
    """

    def __init__(
        self,
        schema_provider,       # ISchemaProvider
        db_connector,          # IDatabaseConnector
        llm,                   # LLMBase (core.llm)
        *,
        difficulty_classifier=None,   # IDifficultyClassifier | None
        fewshot_selector=None,        # IFewShotSelector | None
        sql_validator=None,           # ISQLValidator | None
        error_enhancer=None,          # IErrorEnhancer | None
        stop_judge=None,              # IStopJudge | None
        light_llm=None,               # 轻量 LLM（用于分类/校验/停止判断）
        task_decomposer=None,         # ITaskDecomposer | None
        semantic_postcondition=None,  # 语义后置条件检查器（None=禁用；默认规则版）
        max_rounds: int = MAX_ROUNDS,
    ):
        self.schema_provider = schema_provider
        self.db_connector = db_connector
        self.llm = llm
        self.difficulty_classifier = difficulty_classifier
        self.fewshot_selector = fewshot_selector
        self.sql_validator = sql_validator
        self.error_enhancer = error_enhancer
        self.stop_judge = stop_judge
        self.light_llm = light_llm or llm
        self.max_rounds = max_rounds
        self.task_decomposer = task_decomposer
        # 语义后置条件：None=默认开启（规则版）；False=显式禁用；对象=自定义
        if semantic_postcondition is False:
            self.semantic_postcondition = None
        elif semantic_postcondition is None:
            self.semantic_postcondition = SemanticPostconditionChecker()
        else:
            self.semantic_postcondition = semantic_postcondition

        # Token 计数
        self._total_tokens = 0

    # ═══════════════════════════════════════════════════════════
    #  主入口
    # ═══════════════════════════════════════════════════════════

    def run(self, question: str, mode: str = "auto") -> AgentResult:
        """
        处理一条自然语言查询，返回 AgentResult。

        mode:
          - "auto":   默认，先尝试直接生成SQL，失败时自动 Plan-Execute 兜底
          - "simple": 只走直接生成，不拆解（最快）
          - "plan":   直接拆解执行 Plan-Execute。
                      但如果难度分级为 easy/medium，自动回退到 simple
        """
        # ── 安全: Prompt 注入防御 + 查询意图校验 ──
        from nl2sql_agent.interfaces.prompt_guard import create_prompt_guard
        guard = create_prompt_guard()

        # 注入检测
        is_safe, block_reason = guard.check(question)
        if not is_safe:
            return AgentResult(
                success=False,
                question=question,
                final_sql="",
                stop_reason=StopReason.ERROR_UNRECOVERABLE,
                rounds_attempted=1,
                result=SQLResult(success=False, error=f"输入被安全拦截: {block_reason}"),
                total_time_ms=0,
                total_tokens=0,
            )

        # 查询意图校验
        has_intent, intent_msg = guard.has_query_intent(question)
        if not has_intent:
            return AgentResult(
                success=False,
                question=question,
                final_sql="",
                stop_reason=StopReason.ERROR_UNRECOVERABLE,
                rounds_attempted=1,
                result=SQLResult(success=False, error=intent_msg),
                total_time_ms=0,
                total_tokens=0,
            )

        start_time = time.perf_counter()
        self._total_tokens = 0

        # 重置错误增强器的状态（新查询 → 清空追踪）
        if self.error_enhancer:
            if hasattr(self.error_enhancer, 'reset'):
                self.error_enhancer.reset()

        # ── Step 0: 难度分级（难点B）──
        classification = self._classify_difficulty(question)

        # ── Plan模式智能回退: easy/medium 不需要拆解 ──
        # easy(单表):    100%直接成功, Plan会慢5x
        # medium(聚合/2表): 100%直接成功, Plan会慢5x
        # hard(3表+/子查询): 才值得拆解
        if mode == "plan" and classification is not None:
            if classification.difficulty in (Difficulty.EASY, Difficulty.MEDIUM):
                mode = "simple"  # 自动回退

        # ── Step 1: Schema 检索（难点A）──
        schema_result = self.schema_provider.retrieve(question)

        # ── Step 2: Few-shot 选择（难点B）──
        examples = self._select_fewshot(question, classification)
        examples_text = self._format_examples(examples)

        # ── Step 3: Prompt 构造 ──
        messages = self._build_initial_messages(
            question, schema_result.ddl_text, examples_text
        )

        # ── Plan 模式: 直接拆解，不走单句SQL生成 ──
        if mode == "plan" and self.task_decomposer is not None:
            decomposed = self._try_decompose(question, [], schema_result.ddl_text)
            if decomposed is not None:
                elapsed = (time.perf_counter() - start_time) * 1000
                return AgentResult(
                    success=True,
                    question=question,
                    final_sql=decomposed["sql"],
                    result=decomposed["result"],
                    stop_reason=StopReason.SUCCESS,
                    rounds_attempted=1,
                    difficulty=classification.difficulty if classification else None,
                    trace=[],
                    total_time_ms=elapsed,
                    total_tokens=self._total_tokens,
                    nl_summary=decomposed["result"].rows[0][0] if decomposed["result"].rows else "",
                )

        # ── Agent 纠错循环 ──
        trace = []
        final_result = None
        final_sql = ""
        prev_sql = None
        prev_error_type = None
        stop_reason = StopReason.SUCCESS
        semantic_violation = False   # 是否出现"能跑但语义不合"的结果

        for round_idx in range(1, self.max_rounds + 1):
            round_trace = RoundTrace(round_idx=round_idx)

            # Step 4: LLM 生成 SQL
            sql = self._generate_sql(messages)
            round_trace.sql = sql

            # Step 5: SQL 预校验（难点C）
            validation = self._validate_sql(sql)
            round_trace.validation = validation

            if validation and not validation.is_valid:
                # 预校验失败 → 判断是否可恢复
                if self.stop_judge:
                    judgment = self.stop_judge.judge(
                        round_idx=round_idx, sql=sql,
                        validation=validation,
                        prev_sql=prev_sql,
                        prev_error_type=prev_error_type,
                    )
                    round_trace.stop_judgment = judgment
                    if judgment.should_stop:
                        stop_reason = judgment.stop_reason
                        final_sql = sql
                        round_trace.sql = sql
                        trace.append(round_trace)
                        break

                # 预校验失败但可恢复 → 构造修正 prompt
                messages = self._append_validation_feedback(
                    messages, sql, validation
                )
                prev_sql = sql
                trace.append(round_trace)
                continue

            # Step 6: SQL 执行
            result = self.db_connector.execute(sql)
            round_trace.result = result

            if result.success:
                # ── 语义后置条件（"能跑 ≠ 正确"）──
                spc = self._check_semantic_postcondition(question, sql, result)
                if spc is not None and not spc.ok:
                    semantic_violation = True
                    round_trace.semantic_postcondition = spc.reason
                    messages = self._append_semantic_feedback(
                        messages, sql, spc.reason
                    )
                    prev_sql = sql
                    prev_error_type = ErrorType.OTHER
                    trace.append(round_trace)
                    continue

                # ── 执行成功 → 质量检查 ──
                quality_ok, warning = self._check_result_quality(result, question)

                if quality_ok:
                    stop_reason = StopReason.SUCCESS
                    final_result = result
                    final_sql = sql
                    trace.append(round_trace)
                    break
                else:
                    # 质量问题（如结果太多）
                    stop_reason = StopReason.SUCCESS_WITH_WARNING
                    final_result = result
                    final_sql = sql
                    trace.append(round_trace)
                    break
            else:
                # ── 执行失败 → 停止判断（难点F）──
                if self.stop_judge:
                    judgment = self.stop_judge.judge(
                        round_idx=round_idx,
                        sql=sql,
                        result=result,
                        prev_sql=prev_sql,
                        prev_error_type=prev_error_type,
                    )
                    round_trace.stop_judgment = judgment

                    if judgment.should_stop:
                        stop_reason = judgment.stop_reason
                        final_result = result
                        final_sql = sql
                        trace.append(round_trace)
                        break

                # ── 错误增强 + 构造纠错 Prompt（难点E）──
                error_feedback = self._enhance_error(
                    sql, result, round_idx, prev_error_type
                )
                round_trace.error_enhanced = error_feedback

                messages = self._append_error_feedback(
                    messages, sql, error_feedback,
                    schema_ddl=schema_result.ddl_text,
                )

                prev_sql = sql
                prev_error_type = result.error_type
                trace.append(round_trace)

        else:
            # 循环结束仍未成功（MAX_ROUNDS）
            stop_reason = StopReason.MAX_ROUNDS
            if semantic_violation:
                # 最后几轮全被语义后置条件拦下：执行"成功"但结果不符意图，
                # 不能当作成功返回 → 置为失败，交给 Plan-Execute / 错误路径。
                final_result = None
                final_sql = trace[-1].sql if trace else ""
            else:
                final_result = trace[-1].result if trace else None
                final_sql = trace[-1].sql if trace else ""

        # ── 兜底: 任务拆解 (Plan-Execute) ──
        # "auto": 失败即拆解
        # "simple": 失败后自动升级尝试拆解（逃生舱）
        # "plan": 已在前面直接拆解，不会到这里
        if (not (final_result and final_result.success)
                and self.task_decomposer is not None
                and stop_reason in (StopReason.NO_PROGRESS, StopReason.MAX_ROUNDS,
                                    StopReason.STAGNATION)):

            decomposed = self._try_decompose(question, trace, schema_result.ddl_text)
            if decomposed is not None:
                # 用拆解结果替换失败的原始结果
                final_result = decomposed["result"]
                final_sql = decomposed["sql"]
                stop_reason = StopReason.SUCCESS
                if mode == "simple":
                    trace.append(RoundTrace(
                        round_idx=len(trace) + 1,
                        sql="[AUTO-UPGRADE] simple多次失败, 自动升级到Plan-Execute",
                    ))
                trace.append(RoundTrace(
                    round_idx=len(trace) + 1,
                    sql="[DECOMPOSED] " + decomposed["plan"],
                ))

        # ── 构建最终结果 ──
        elapsed = (time.perf_counter() - start_time) * 1000

        # 自然语言总结（成功时生成）
        nl_summary = ""
        if final_result and final_result.success and final_result.row_count > 0:
            nl_summary = self._generate_nl_summary(question, final_sql, final_result)

        return AgentResult(
            success=(final_result is not None and final_result.success),
            question=question,
            final_sql=final_sql,
            result=final_result,
            stop_reason=stop_reason,
            rounds_attempted=len(trace),
            difficulty=classification.difficulty if classification else None,
            trace=trace,
            total_time_ms=elapsed,
            total_tokens=self._total_tokens,
            nl_summary=nl_summary,
        )

    def _generate_nl_summary(self, question: str, sql: str, result) -> str:
        """
        SQL 执行成功后，让 LLM 用一句话总结查询结果。

        这就是"数据获取周期压缩"的最后一环——不是返回裸表格，
        而是返回人可以直接读的结论。
        """
        # 格式化结果供 LLM 阅读
        result_text = f"字段: {', '.join(result.columns)}\n"
        for i, row in enumerate(result.rows[:5]):
            result_text += f"  [{i+1}] " + " | ".join(str(v) for v in row) + "\n"

        prompt = f"""你是数据分析助手。根据查询结果用一句简洁的话总结，直接回答用户问题。

用户问题: {question}

查询SQL: {sql}

查询结果 (共{result.row_count}行):
{result_text}

请用一句话总结，直接给出结论（≤50字）。不要重复问题，不要解释SQL。"""

        try:
            messages = [
                {"role": "system", "content": "你是数据分析助手，用一句话总结查询结果。"},
                {"role": "user", "content": prompt},
            ]
            summary = self.light_llm.generate(messages, timeout=15)
            self._total_tokens += self._estimate_tokens(messages, summary)
            return summary.strip()
        except Exception:
            return ""  # 总结失败不影响主流程

    def _try_decompose(self, question: str, trace: list,
                       schema_ddl: str) -> dict | None:
        """
        复杂查询兜底——任务拆解 (Plan-Execute)。

        当单句 SQL 连续失败时:
          1. LLM 拆解为 2-3 个子任务（包含 SQL hint）
          2. 每个子任务直接生成 SQL + 执行（不走完整 Agent 循环，避免递归）
          3. 汇总子结果 → LLM 合成最终答案
        """
        failed_sqls = [t.sql for t in trace if t.sql and not (t.result and t.result.success)]
        plan = self.task_decomposer.decompose(question, schema_ddl, failed_sqls)

        if not plan.subtasks:
            return None

        # ── 顺序依赖型: 优先尝试 CTE 链（一次执行，避免文本传递丢失精度）──
        problem_type = getattr(plan, '_problem_type', 'parallel')
        if problem_type == 'sequential':
            cte_result = self._try_cte_chain(question, schema_ddl, plan)
            if cte_result is not None:
                return cte_result
            # CTE链失败 → 退回到独立子任务模式

        # ── 独立汇总型: 逐个子任务生成SQL + 纠错 + 执行 ──
        subtask_results = []
        SUB_MAX_RETRIES = 2  # 子任务最多修正 2 次

        for st in plan.subtasks:
            sub_sql = ""
            sub_result = None
            subtask_summary = ""

            for sub_round in range(1, SUB_MAX_RETRIES + 1):
                if sub_round == 1:
                    # 首次: 从子问题生成 SQL
                    sub_prompt = (
                        f"数据库 Schema:\n{schema_ddl}\n\n"
                        f"## 规则\n"
                        f"1. 只生成 SELECT/WITH 只读查询。禁止 INSERT/UPDATE/DELETE/DROP\n"
                        f"2. 使用 SQLite 语法。字符串用单引号。\n"
                        f"3. 日期用 strftime() 或 date() 函数。\n"
                        f"4. 表名和列名必须与 Schema 严格一致，不要臆造列名。\n\n"
                        f"## 子问题\n{st.question}\n\n"
                        f"只输出 SQL，用```sql包裹。"
                    )
                else:
                    # 重试: 带上错误信息修正
                    sub_prompt = (
                        f"你的 SQL 执行报错: {sub_result.error}\n"
                        f"请修正 SQL。只输出修正后的 SQL，用```sql包裹。"
                    )

                sub_messages = [
                    {"role": "system", "content": "你是 SQL 专家。只输出 SQL。"},
                    {"role": "user", "content": sub_prompt},
                ]
                sub_response = self.llm.generate(sub_messages)
                self._total_tokens += self._estimate_tokens(sub_messages, sub_response)
                sub_sql = self._extract_sql(sub_response)

                # 执行
                sub_result = self.db_connector.execute(sub_sql)
                if sub_result.success:
                    break  # 成功，退出重试

            # 收集结果
            if sub_result.success and sub_result.rows:
                rows = sub_result.rows[:5]
                cols = sub_result.columns
                subtask_summary = f"cols={cols}, rows={rows}"
                if sub_result.row_count > 5:
                    subtask_summary += f" (+{sub_result.row_count - 5} more)"
            elif not sub_result.success:
                subtask_summary = f"FAILED after {SUB_MAX_RETRIES} retries: {sub_result.error[:150]}"

            subtask_results.append({
                "step": st.step,
                "description": st.description,
                "sql": sub_sql,
                "result_summary": subtask_summary or sub_result.error,
                "success": sub_result.success,
            })

        # LLM 汇总
        synthesis = self.task_decomposer.synthesize(question, subtask_results)
        self._total_tokens += self._estimate_tokens(
            [{"role": "user", "content": synthesis}], synthesis
        )

        from nl2sql_agent.schemas import SQLResult
        combined_result = SQLResult(
            success=True,
            columns=["分析结论"],
            rows=[(synthesis.strip(),)],
            row_count=1,
        )

        combined_sql = "-- [Plan-Execute] " + plan.reasoning + "\n"
        for sr in subtask_results:
            combined_sql += f"\n-- Step {sr['step']}: {sr['description']}\n{sr['sql']}\n"

        return {
            "result": combined_result,
            "sql": combined_sql,
            "plan": plan.reasoning,
            "subtask_results": subtask_results,
        }

    def _try_cte_chain(self, question: str, schema_ddl: str, plan) -> dict | None:
        """
        顺序依赖型问题: 生成一条带 WITH 的 CTE 链 SQL。

        和拆解为独立子任务的区别:
          - 独立子任务: 每步结果文本传递 → 丢失表格精度
          - CTE 链:      一个 SQL 内用 WITH 串联 → 保留完整数据流

        适用: 队列留存、环比增长、排名后筛选等后步依赖前步精确数据的场景。
        """
        cte_hint = getattr(plan, '_cte_hint', '')
        sub_descriptions = "\n".join(
            f"Step {s.step}: {s.description}" for s in plan.subtasks
        )

        cte_prompt = (
            f"数据库 Schema:\n{schema_ddl}\n\n"
            f"## 问题\n{question}\n\n"
            f"## 拆解思路\n{plan.reasoning}\n\n"
            f"## 子步骤\n{sub_descriptions}\n\n"
            f"{'## CTE 提示\n' + cte_hint if cte_hint else ''}\n"
            f"## 规则\n"
            f"1. 用 WITH 子句串联多步计算: WITH step1 AS (...), step2 AS (...) SELECT ...\n"
            f"2. 每步 CTE 只做一件事，中间列命名清晰\n"
            f"3. 最后 SELECT 输出问题的最终答案\n"
            f"4. 使用 SQLite 语法。只输出 SQL，用```sql包裹。\n"
        )

        messages = [
            {"role": "system", "content": "你是 SQL 专家。生成带 WITH 的 CTE 链查询。"},
            {"role": "user", "content": cte_prompt},
        ]

        # CTE 链重试最多 2 次
        for retry in range(2):
            try:
                response = self.llm.generate(messages)
            except Exception as e:
                # LLM 调用失败 → 不继续重试，退出 CTE 链
                break
            self._total_tokens += self._estimate_tokens(messages, response)
            sql = self._extract_sql(response)

            result = self.db_connector.execute(sql)
            if result.success:
                from nl2sql_agent.schemas import SQLResult
                combined = SQLResult(
                    success=True,
                    columns=result.columns,
                    rows=result.rows,
                    row_count=result.row_count,
                    execution_time_ms=result.execution_time_ms,
                )
                return {
                    "result": combined,
                    "sql": "-- [Plan-Execute CTE链] " + plan.reasoning + "\n" + sql,
                    "plan": plan.reasoning,
                }

            # 失败 → 带错误重试
            messages.append({"role": "assistant", "content": f"```sql\n{sql}\n```"})
            messages.append({"role": "user", "content":
                f"SQL 执行报错: {result.error}\n请修正。只输出修正后的 SQL。"})
            self._total_tokens += self._estimate_tokens(messages, "")

        return None  # CTE 链 2 次都失败 → 回退到独立子任务

    # ═══════════════════════════════════════════════════════════
    #  Step 0: 难度分级
    # ═══════════════════════════════════════════════════════════

    def _classify_difficulty(self, question: str) -> DifficultyClassification | None:
        """使用轻量 LLM 对问题进行难度分级，失败时降级到规则分类器（难点B）"""
        if self.difficulty_classifier is None:
            return None
        try:
            return self.difficulty_classifier.classify(question)
        except Exception:
            # LLM 分类失败 → 降级到规则兜底
            from nl2sql_agent.interfaces.difficulty_classifier import RuleBasedDifficultyClassifier
            try:
                fallback = RuleBasedDifficultyClassifier()
                return fallback.classify(question)
            except Exception:
                return None  # 连规则也失败了才放弃

    # ═══════════════════════════════════════════════════════════
    #  Step 2: Few-shot 选择
    # ═══════════════════════════════════════════════════════════

    def _select_fewshot(self, question: str,
                        classification: DifficultyClassification | None):
        """根据难度分级动态选择 Few-shot 示例（难点B）"""
        if self.fewshot_selector is None:
            return []
        if classification is None:
            # 无分级 → 返回难度为 MEDIUM 的默认 3 条
            from nl2sql_agent.agent.fewshot_examples import FEWSHOT_POOL
            return [ex for ex in FEWSHOT_POOL
                    if ex.difficulty == Difficulty.MEDIUM][:3]
        return self.fewshot_selector.select(question, classification)

    @staticmethod
    def _format_examples(examples: list) -> str:
        from nl2sql_agent.agent.prompts import format_fewshot
        return format_fewshot(examples)

    # ═══════════════════════════════════════════════════════════
    #  Step 3: Prompt 构造
    # ═══════════════════════════════════════════════════════════

    def _build_initial_messages(self, question: str, schema_ddl: str,
                                examples_text: str) -> list[dict]:
        from nl2sql_agent.agent.prompts import NL2SQL_SYSTEM_PROMPT

        system_content = safe_format(NL2SQL_SYSTEM_PROMPT,
            schema=schema_ddl,
            examples=examples_text,
            question=question,
        )

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": question},
        ]

    # ═══════════════════════════════════════════════════════════
    #  Step 4: LLM 生成 SQL
    # ═══════════════════════════════════════════════════════════

    def _generate_sql(self, messages: list[dict]) -> str:
        """调用 LLM 生成 SQL，并从响应中提取 SQL"""
        response = self.llm.generate(messages)
        self._total_tokens += self._estimate_tokens(messages, response)
        return self._extract_sql(response)

    @staticmethod
    def _estimate_tokens(messages: list[dict], response: str = "") -> int:
        """估算 Token 消耗（中文: ~1.5 字/token, 英文: ~4 字/token）"""
        total_chars = sum(len(m.get("content", "")) for m in messages)
        total_chars += len(response)
        return max(1, total_chars // 2)  # 保守估计: 2 字符 ≈ 1 token

    @staticmethod
    def _extract_sql(text: str) -> str:
        """从 LLM 响应中提取 SQL 语句"""
        # 优先提取 ```sql 代码块
        match = re.search(r'```sql\s*\n?(.*?)```', text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()

        # 尝试提取普通代码块
        match = re.search(r'```\s*\n?(.*?)```', text, re.DOTALL)
        if match:
            return match.group(1).strip()

        # 查找 SELECT 语句
        match = re.search(r'(SELECT\s+.+?;)', text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()

        # 兜底：返回原始文本（去掉明显的非 SQL 内容）
        lines = text.strip().split('\n')
        sql_lines = [
            l for l in lines
            if not l.strip().startswith(('Thought:', 'Action:', '```', '解释', '说明'))
        ]
        return '\n'.join(sql_lines).strip()

    # ═══════════════════════════════════════════════════════════
    #  Step 5: SQL 预校验
    # ═══════════════════════════════════════════════════════════

    def _validate_sql(self, sql: str) -> ValidationResult | None:
        """对生成的 SQL 进行预校验（难点C）"""
        if self.sql_validator is None:
            return None
        return self.sql_validator.validate(sql, self.schema_provider.schema_info)

    # ═══════════════════════════════════════════════════════════
    #  Step 6: SQL 执行（直接委托给 db_connector）
    #  Step 7: 停止判断（委托给 stop_judge，在 run() 中内联调用）
    # ═══════════════════════════════════════════════════════════

    # ═══════════════════════════════════════════════════════════
    #  Step 8: 错误增强 + 反馈注入
    # ═══════════════════════════════════════════════════════════

    def _enhance_error(self, sql: str, result: SQLResult,
                       round_idx: int, prev_error_type: ErrorType | None) -> str:
        """使用错误增强器生成纠错反馈（难点E）"""
        if self.error_enhancer:
            return self.error_enhancer.enhance(sql, result, round_idx, prev_error_type)

        # 无增强器 → 原始错误直传
        return f"SQL 执行报错: {result.error}\n请修正 SQL。"

    def _append_error_feedback(self, messages: list[dict],
                               sql: str, error_feedback: str,
                               schema_ddl: str = "") -> list[dict]:
        """将错误反馈追加到消息历史中

        自动检测 CoT 模式：当 error_enhancer 返回了 [ANALYSIS] 标记时，
        切换为 COT_CORRECTION_PROMPT（两步推理），否则用普通 CORRECTION_PROMPT。
        """
        # 找到原始的 user 问题
        user_question = ""
        for m in messages:
            if m["role"] == "user":
                user_question = m["content"]
                break

        messages.append({"role": "assistant", "content": f"```sql\n{sql}\n```"})

        # 检测 CoT 模式
        is_cot = any(marker in error_feedback for marker in
                     ["[ANALYSIS]", "Step 1", "[/ANALYSIS]", "[/SQL]"])

        if is_cot:
            # CoT 提示本身已包含步骤引导，直接作为 user message 注入
            correction = safe_format(COT_CORRECTION_PROMPT,
                question=user_question,
                schema=schema_ddl,
                sql=sql,
                error_feedback=error_feedback,
            )
        else:
            correction = safe_format(CORRECTION_PROMPT,
                question=user_question,
                sql=sql,
                error_feedback=error_feedback,
            )

        messages.append({"role": "user", "content": correction})
        return messages

    def _append_validation_feedback(self, messages: list[dict],
                                    sql: str, validation: ValidationResult) -> list[dict]:
        """将预校验失败反馈追加到消息历史中"""
        feedback = (
            f"SQL 预校验未通过: {validation.error}\n"
            f"修正建议: {validation.suggestion}\n"
            f"请根据建议修正 SQL，只输出修正后的 SQL。"
        )
        messages.append({"role": "assistant", "content": f"```sql\n{sql}\n```"})
        messages.append({"role": "user", "content": feedback})
        return messages

    # ═══════════════════════════════════════════════════════════
    #  语义后置条件（"能跑 ≠ 正确"）
    # ═══════════════════════════════════════════════════════════

    def _check_semantic_postcondition(
        self, question: str, sql: str, result,
    ) -> PostconditionResult | None:
        """SQL 执行成功后，检查结果是否符合查询的语义意图。

        返回 PostconditionResult；未启用检查器（None）时返回 None 跳过。
        """
        if self.semantic_postcondition is None:
            return None
        return self.semantic_postcondition.check(question, sql, result)

    def _append_semantic_feedback(
        self, messages: list[dict], sql: str, reason: str,
    ) -> list[dict]:
        """将语义后置条件违规喂回模型，触发修正轮（预算内收敛）。"""
        messages.append({"role": "assistant", "content": f"```sql\n{sql}\n```"})
        feedback = (
            f"SQL 执行成功，但结果不满足查询的语义要求。{reason}\n"
            f"请根据这条语义后置条件修正 SQL，只输出修正后的 SQL。"
        )
        messages.append({"role": "user", "content": feedback})
        return messages

    # ═══════════════════════════════════════════════════════════
    #  结果质量检查
    # ═══════════════════════════════════════════════════════════

    def _check_result_quality(self, result: SQLResult,
                              question: str) -> tuple[bool, str]:
        """
        对执行成功的结果做质量检查。

        Returns:
            (is_ok, warning_message)
        """
        from nl2sql_agent.config import ROW_COUNT_WARN_THRESHOLD

        warnings = []

        # 检查1: 空结果
        if result.row_count == 0:
            warnings.append(
                "查询返回 0 条结果。请确认查询条件是否正确——"
                "可能是 WHERE 条件过严、JOIN 条件错误或表选择不当。"
            )

        # 检查2: 结果过多
        if result.row_count > ROW_COUNT_WARN_THRESHOLD:
            warnings.append(
                f"查询返回 {result.row_count} 行结果（超过 {ROW_COUNT_WARN_THRESHOLD}）。"
                f"如果用户不是要求'全部/所有'数据，可能缺少 LIMIT 或聚合操作。"
            )

        if warnings:
            return False, "; ".join(warnings)
        return True, ""


# ═══════════════════════════════════════════════════════════════
#  便捷函数：一行创建 + 运行
# ═══════════════════════════════════════════════════════════════

def run_nl2sql(question: str, agent: NL2SQLAgent) -> AgentResult:
    """便捷入口：运行 NL2SQL Agent 并返回结果"""
    return agent.run(question)
