"""
NL2SQL Agent — Web UI (Gradio)

启动: python app.py
访问: http://localhost:7860
"""

import sys
import os
import time
import gradio as gr

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

from nl2sql_agent.agent_factory import build_agent, load_dotenv

# ─── 全局 Agent（启动时初始化一次）───
_agent = None
_db_connector = None
_schema_info = None
_init_cooldown_until = 0.0  # 初始化失败后的冷静期时间戳


def init_agent(schema_strategy="full"):
    """初始化 Agent"""
    global _agent, _db_connector, _schema_info
    load_dotenv()
    _agent, _db_connector, _ = build_agent(schema_strategy)
    _schema_info = _db_connector.get_schema_info()
    return _agent


def query_nl2sql(question, schema_strategy, mode):
    """处理自然语言查询"""
    global _agent

    if not question.strip():
        return "", "请输入问题", "", "", ""

    # 安全拦截
    from nl2sql_agent.interfaces.prompt_guard import create_prompt_guard
    guard = create_prompt_guard()
    is_safe, reason = guard.check(question)
    if not is_safe:
        return "", f"[安全拦截] {reason}", "", "", ""

    # 查询意图校验
    has_intent, intent_msg = guard.has_query_intent(question)
    if not has_intent:
        return "", intent_msg, "", "", ""

    # 如果策略变了，重新初始化
    if _agent is None or _agent.schema_provider.strategy_name != schema_strategy:
        now = time.time()
        if now < _init_cooldown_until:
            return "", f"Agent 初始化冷却中，请 {_init_cooldown_until - now:.0f}s 后重试", "", "", ""
        try:
            init_agent(schema_strategy)
        except Exception as e:
            _init_cooldown_until = time.time() + 30  # 失败后 30s 冷却
            return "", f"Agent 初始化失败: {e}", "", "", ""

    start = time.perf_counter()
    result = _agent.run(question, mode=mode)
    elapsed = time.perf_counter() - start

    # ── 构建返回内容 ──

    # 状态信息
    if result.success:
        status = f"✅ 成功 | 难度: {result.difficulty.value if result.difficulty else 'N/A'} | "
        status += f"轮次: {result.rounds_attempted} | "
        status += f"耗时: {result.total_time_ms:.0f}ms | "
        status += f"停止: {result.stop_reason.value}"
    else:
        status = f"❌ 失败 | {result.stop_reason.value} | 轮次: {result.rounds_attempted}"

    # SQL
    sql_display = result.final_sql if result.final_sql else "（未生成 SQL）"

    # 结果表格
    if result.result and result.result.success:
        if result.result.row_count > 0:
            result_table = _format_table(result.result.columns, result.result.rows)
        else:
            result_table = "查询返回 0 行。"
    elif result.result:
        result_table = f"错误: {result.result.error}"
    else:
        result_table = "（无结果）"

    # NL 总结
    nl_summary = getattr(result, 'nl_summary', '') or ""

    # 纠错追踪
    if result.rounds_attempted > 1:
        trace_lines = ["### 纠错追踪"]
        for t in result.trace:
            icon = "✅" if (t.result and t.result.success) else "🔄"
            err = ""
            if t.result and not t.result.success:
                err = f" → {t.result.error_type.value if t.result.error_type else 'error'}"
            trace_lines.append(f"- Round {t.round_idx} {icon}{err}")
        trace_text = "\n".join(trace_lines)
    else:
        trace_text = ""

    return sql_display, status, result_table, nl_summary, trace_text


def _format_table(columns, rows):
    """格式化为 Markdown 表格（最多 20 行）"""
    if not columns:
        return "（无数据）"

    lines = ["| " + " | ".join(columns) + " |"]
    lines.append("|" + "|".join([" --- " for _ in columns]) + "|")

    for row in rows[:20]:
        lines.append("| " + " | ".join(str(v) for v in row) + " |")

    if len(rows) > 20:
        lines.append(f"\n*（仅显示前 20 行，共 {len(rows)} 行）*")

    return "\n".join(lines)


def show_schema(strategy):
    """显示当前 Schema"""
    global _schema_info
    if _schema_info is None:
        return "请先启动 Agent"

    tables_info = []
    for t in _schema_info.tables:
        cols = ", ".join(f"{c.name} ({c.data_type})" for c in t.columns)
        desc = f" — {t.description}" if t.description else ""
        tables_info.append(f"- **{t.name}** ({t.row_count} 行){desc}")
        tables_info.append(f"  `{cols}`")

    return "\n".join(tables_info)


# ─── Gradio UI ───

def create_ui():
    theme = gr.themes.Soft(primary_hue="blue")

    with gr.Blocks(title="NL2SQL Agent") as demo:
        gr.Markdown("""
        # NL2SQL Agent — 算法工程师的数据助手

        ### 用自然语言完成数据分析查询，将数据获取周期从**小时级压缩到秒级**。

        支持**漏斗分析、用户分层、转化归因、留存分析、A/B指标对比**等算法工程师日常分析场景。
        输入问题 → Agent 自动生成 SQL → 返回结果 + 一句话总结。
        """)

        with gr.Row():
            with gr.Column(scale=3):
                question = gr.Textbox(
                    label="输入你的问题",
                    placeholder="例如: 统计每月GMV总额及环比增长率",
                    lines=2,
                )

            with gr.Column(scale=1):
                mode = gr.Dropdown(
                    label="执行模式",
                    choices=["auto", "simple", "plan"],
                    value="auto",
                    info="auto=智能 | simple=直接SQL | plan=拆解执行",
                )
                strategy = gr.Dropdown(
                    label="Schema 策略",
                    choices=["full", "keyword", "hybrid"],
                    value="full",
                    info="full=全量 | keyword=关键词 | hybrid=混合精排",
                )
                submit_btn = gr.Button("查询", variant="primary", size="lg")
                schema_btn = gr.Button("查看 Schema", variant="secondary", size="sm")

        with gr.Row():
            status = gr.Markdown("就绪，等待查询...")

        with gr.Accordion("SQL 语句", open=True):
            sql_output = gr.Code(language="sql", label="生成的 SQL")

        with gr.Accordion("查询结果", open=True):
            result_output = gr.Markdown("（等待查询）")

        with gr.Accordion("自然语言总结", open=True):
            summary_output = gr.Markdown("")

        with gr.Accordion("纠错追踪（多轮时显示）", open=False):
            trace_output = gr.Markdown("")

        # ─── 事件 ───
        submit_btn.click(
            fn=query_nl2sql,
            inputs=[question, strategy, mode],
            outputs=[sql_output, status, result_output, summary_output, trace_output],
        )

        schema_btn.click(
            fn=show_schema,
            inputs=[strategy],
            outputs=[result_output],
        )

        # ─── 示例（算法工程师日常分析场景）───
        gr.Examples(
            examples=[
                ["加购转化率最高的品类是哪个？加购转化率=下单用户数/加购用户数"],
                ["不同会员等级的复购率对比（复购=下单≥2次的用户占比）"],
                ["最近30天新用户中，7天内完成首单的比例？"],
                ["用户从首次浏览到首单的平均间隔是多少天？"],
                ["周末和工作日的订单量和客单价差异？"],
                ["只浏览不购买的用户主要看了哪些品类？浏览量Top5"],
                ["高频用户(>5单)和低频用户(≤2单)的品类偏好差异？"],
                ["购买过电子产品的用户还买过哪些品类？交叉购买分析"],
            ],
            inputs=[question],
            label="算法工程师分析场景（点击直接查询）",
        )

    return demo


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", help="生成公网链接分享给他人测试")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    print("正在初始化 NL2SQL Agent...")
    load_dotenv()
    init_agent("full")
    print(f"Agent 就绪 | 数据库: {len(_schema_info.tables)} 张表")

    if args.share:
        print("[SHARE] 正在生成公网链接，72小时内有效...")

    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        inbrowser=not args.share,
    )
