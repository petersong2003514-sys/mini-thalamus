"""
接口 7: IStopJudge —— 停止条件判断器（难点F）

Agent 循环的终极裁判——判断当前是否应该停止以及为什么。

设计依据（你的决策）:
  不能只靠正则匹配（太硬）——需要 LLM 级别的语义理解。
  用轻量 LLM 判断: SQL 的执行结果/错误是否本质上和上一轮一样？
  是否已经无法进一步修复？

判断流程:
  1. 硬规则预判（MAX_ROUNDS / 相同 SQL 文本归一化比较）→ 快速路由
  2. LLM 语义判断（错误本质是否相同 / 是否有修复空间）→ 硬规则不确定时触发
"""

from abc import ABC, abstractmethod
from nl2sql_agent.schemas import StopJudgment, SQLResult, ValidationResult
from nl2sql_agent.config import StopReason, ErrorType, MAX_ROUNDS
from nl2sql_agent.agent.prompts import safe_format


class IStopJudge(ABC):
    """
    停止条件判断器抽象接口。

    输入: 当前轮次状态（round_idx / sql / result / previous sql / previous error）
    输出: StopJudgment（是否停止 + 原因 + 推理）
    """

    @abstractmethod
    def judge(self, round_idx: int,
              sql: str,
              result: SQLResult | None = None,
              validation: ValidationResult | None = None,
              prev_sql: str | None = None,
              prev_error_type: ErrorType | None = None,
              ) -> StopJudgment:
        """
        综合判断是否应停止循环。

        返回 StopJudgment:
          - should_stop: True → 停止 / False → 继续纠错
          - stop_reason: 停止原因
          - reasoning: 判断依据（透明可解释）
          - is_recoverable: 是否可恢复（True=继续纠错 / False=无法修复）
        """
        ...


# ═══════════════════════════════════════════════════════════════
#  HybridStopJudge — 硬规则预判 + LLM 语义兜底
# ═══════════════════════════════════════════════════════════════

class HybridStopJudge(IStopJudge):
    """
    混合停止判断器。

    层① 硬规则预判（0 Token 消耗）:
      - MAX_ROUNDS: round_idx >= 3 → 直接停止
      - SUCCESS: result.success=True → 直接成功
      - NO_PROGRESS: 归一化 SQL 完全相同 → 直接停止
      - UNRECOVERABLE: PERMISSION_DENIED → 直接停止

    层② LLM 语义判断（层①不确定时触发）:
      - STAGNATION: 错误看起来不同但本质相同？
      - 当前错误是否有修复空间？
    """

    # LLM 语义判断用的 Prompt
    STAGNATION_PROMPT = """你是 SQL 纠错专家。Agent 在尝试修复 SQL 错误，但连续几轮都失败了。
请判断: 这两轮的错误本质是否相同？（即使具体错误文本不同）

## Round {prev_round} 的错误
{prev_error}

## Round {curr_round} 的 SQL
{curr_sql}

## Round {curr_round} 的错误
{curr_error}

## 判断标准
- 如果错误根源相同（比如都是因为用错了列名，只是每次写错的列名不同）→ same_essence: true
- 如果错误根源不同（比如第一轮是语法错，第二轮是表名错）→ same_essence: false
- 如果当前错误有明确的修复方向 → fixable: true
- 如果当前错误 LLM 无法自行修复（如需要的表不存在）→ fixable: false

## 输出格式
{{"same_essence": true/false, "fixable": true/false, "reasoning": "一句话理由"}}
只输出 JSON。"""

    def __init__(self, light_llm=None, max_rounds=MAX_ROUNDS):
        """
        Args:
            light_llm: 轻量 LLM 实例，用于语义层面的 STAGNATION 判断。
                       如果为 None，仅使用硬规则判断。
            max_rounds: 最大重试轮次，默认从 config.MAX_ROUNDS 读取。
        """
        self._light_llm = light_llm
        self._max_rounds = max_rounds
        # 同一错误类型重复次数计数器（避免第二次就误判 STAGNATION）
        self._same_error_count = 0
        self._last_error_type = None

    def judge(self, round_idx: int,
              sql: str,
              result: SQLResult | None = None,
              validation: ValidationResult | None = None,
              prev_sql: str | None = None,
              prev_error_type: ErrorType | None = None,
              ) -> StopJudgment:

        # ── 层①: 硬规则预判 ──

        # 成功
        if result and result.success:
            return StopJudgment(
                should_stop=True,
                stop_reason=StopReason.SUCCESS,
                reasoning="SQL 执行成功",
                is_recoverable=False,
            )

        # 超轮次
        if round_idx >= self._max_rounds:
            return StopJudgment(
                should_stop=True,
                stop_reason=StopReason.MAX_ROUNDS,
                reasoning=f"达到最大重试轮次 {self._max_rounds}",
                is_recoverable=False,
            )

        # 不可恢复错误
        if result and result.error_type == ErrorType.PERMISSION_DENIED:
            return StopJudgment(
                should_stop=True,
                stop_reason=StopReason.ERROR_UNRECOVERABLE,
                reasoning="权限被拒绝，无法恢复",
                is_recoverable=False,
            )

        # SQL 完全没变
        if prev_sql and self._normalize_sql(sql) == self._normalize_sql(prev_sql):
            return StopJudgment(
                should_stop=True,
                stop_reason=StopReason.NO_PROGRESS,
                reasoning="连续两轮生成的 SQL 完全相同，再试无意义",
                is_recoverable=False,
            )

        # ── 层②: LLM 语义判断 STAGNATION ──
        if result and not result.success and prev_error_type is not None:
            curr_error_type = result.error_type

            # 错误类型追踪: 只有同类型错误出现 3+ 次才触发 STAGNATION
            if curr_error_type == self._last_error_type:
                self._same_error_count += 1
            else:
                self._last_error_type = curr_error_type
                self._same_error_count = 1

            if self._same_error_count >= 3:
                if self._light_llm:
                    return self._llm_stagnation_judge(
                        round_idx, sql, result, prev_sql, prev_error_type
                    )
                else:
                    # 无 LLM → 按硬规则：3+ 次相同错误直接 STAGNATION
                    return StopJudgment(
                        should_stop=True,
                        stop_reason=StopReason.STAGNATION,
                        reasoning=f"相同错误类型重复 {self._same_error_count} 次: {curr_error_type.value}",
                        is_recoverable=False,
                    )

        # 可继续纠错
        return StopJudgment(
            should_stop=False,
            stop_reason=StopReason.SUCCESS,  # placeholder
            reasoning="可以继续尝试修正",
            is_recoverable=True,
        )

    def _llm_stagnation_judge(self, round_idx: int,
                               sql: str, result: SQLResult,
                               prev_sql: str | None,
                               prev_error_type: ErrorType) -> StopJudgment:
        """使用轻量 LLM 判断错误是否本质相同（避免硬规则误判）"""
        import json

        prompt = safe_format(self.STAGNATION_PROMPT,
            prev_round=round_idx - 1,
            prev_error=prev_error_type.value,
            curr_round=round_idx,
            curr_sql=sql,
            curr_error=f"{result.error_type.value}: {result.error}" if result else "无错误",
        )

        messages = [
            {"role": "system", "content": "你是 SQL 纠错专家。只输出 JSON。"},
            {"role": "user", "content": prompt},
        ]
        response = self._light_llm.generate(messages)

        try:
            text = response.strip()
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            data = json.loads(text)
        except json.JSONDecodeError:
            # 解析失败 → 保守处理：继续给一次机会
            return StopJudgment(
                should_stop=False,
                stop_reason=StopReason.SUCCESS,
                reasoning="LLM 判断失败，继续尝试",
                is_recoverable=True,
            )

        same_essence = data.get("same_essence", False)
        fixable = data.get("fixable", True)
        reasoning = data.get("reasoning", "")

        if same_essence and not fixable:
            return StopJudgment(
                should_stop=True,
                stop_reason=StopReason.STAGNATION,
                reasoning=f"错误本质相同且无法修复: {reasoning}",
                is_recoverable=False,
            )

        if same_essence and fixable:
            return StopJudgment(
                should_stop=False,
                stop_reason=StopReason.SUCCESS,
                reasoning=f"错误本质相同但仍有修复空间: {reasoning}",
                is_recoverable=True,
            )

        # 错误本质不同 → 继续（可能是新的错误，值得再试）
        return StopJudgment(
            should_stop=False,
            stop_reason=StopReason.SUCCESS,
            reasoning=f"错误类型发生变化，继续尝试: {reasoning}",
            is_recoverable=True,
        )

    @staticmethod
    def _normalize_sql(sql: str) -> str:
        """
        SQL 归一化：去掉空白差异、统一大小写。
        用于判断两轮 SQL 是否"完全一样"。
        """
        return ' '.join(sql.upper().split())
