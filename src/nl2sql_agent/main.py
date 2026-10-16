#!/usr/bin/env python3
"""
NL2SQL Agent — CLI 入口

用法:
  # 单次查询
  python main.py --question "北京用户买得最多的品类是什么？"

  # 交互模式（逐条问答）
  python main.py --interactive

  # 指定 Schema 策略 + 最大轮次
  python main.py --question "..." --schema full --max-rounds 5

  # 列出可用表
  python main.py --list-tables
"""

import sys
import os
import argparse
import time
import json

# Windows 控制台 UTF-8 编码
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

from nl2sql_agent.config import (
    DB_PATH, DB_READONLY, SCHEMA_STRATEGY, MAX_ROUNDS,
    PRIMARY_API_KEY, PRIMARY_MODEL, PRIMARY_BASE_URL,
)
from nl2sql_agent.agent_factory import build_agent, load_dotenv


# 多数据源注册
def load_sources():
    """加载数据源配置文件"""
    sources_file = os.path.join(PROJECT_ROOT, "data", "sources.json")
    if os.path.exists(sources_file):
        with open(sources_file, "r", encoding="utf-8") as f:
            return json.load(f).get("sources", [])
    return []


# ═══════════════════════════════════════════════════════════════
#  输出美化
# ═══════════════════════════════════════════════════════════════

class Colors:
    """终端颜色（ANSI）"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    END = '\033[0m'

    @staticmethod
    def supports():
        return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def c(color, text):
    """条件着色"""
    if Colors.supports():
        return f"{color}{text}{Colors.END}"
    return text


def print_result(agent_result):
    """美化打印 AgentResult"""
    r = agent_result

    print()
    print(c(Colors.BOLD, "═" * 70))

    # 状态行
    if r.success:
        status = c(Colors.GREEN + Colors.BOLD, "✅ SUCCESS")
    else:
        status = c(Colors.RED + Colors.BOLD, "❌ FAILED")
    # Built as a plain variable: a nested f-string reusing the outer quote
    # character is only legal from Python 3.12 (PEP 701).
    difficulty = r.difficulty.value if r.difficulty else "N/A"
    detail = (
        f"轮次: {r.rounds_attempted}  |  "
        f"耗时: {r.total_time_ms:.0f}ms  |  "
        f"难度: {difficulty}"
    )
    print(f"  {status}  {c(Colors.DIM, r.stop_reason.value)}")
    print(f"  {c(Colors.DIM, detail)}")

    # SQL
    if r.final_sql:
        print()
        is_decomposed = "[Plan-Execute]" in r.final_sql
        label = "── 拆解执行 (Plan-Execute) ──" if is_decomposed else "── SQL ──"
        print(c(Colors.CYAN + Colors.BOLD, f"  {label}"))
        for line in r.final_sql.strip().split('\n')[:30]:
            print(f"  {c(Colors.DIM, line)}")

    # 结果
    if r.result and r.result.success:
        print()
        print(c(Colors.GREEN + Colors.BOLD, "  ── 结果 ──"))
        shape = (
            f"{r.result.row_count} 行, {len(r.result.columns)} 列, "
            f"{r.result.execution_time_ms:.1f}ms"
        )
        print(f"  {c(Colors.DIM, shape)}")
        if r.result.columns:
            # 打印列头
            header = "  " + " │ ".join(r.result.columns)
            print(c(Colors.BOLD, header))
            print(c(Colors.DIM, "  " + "─" * (len(header) - 2)))
            # 打印数据行
            for row in r.result.rows[:20]:
                print("  " + " │ ".join(str(v) for v in row))
            if r.result.row_count > 20:
                print(c(Colors.DIM, f"  ... (仅显示前 20 行，共 {r.result.row_count} 行)"))

        # NL 总结
        if getattr(r, 'nl_summary', ''):
            print()
            print(c(Colors.GREEN, f"  [总结] {r.nl_summary}"))

    elif r.result and not r.result.success:
        print()
        print(c(Colors.RED + Colors.BOLD, "  ── 错误 ──"))
        print(f"  {c(Colors.RED, r.result.error)}")

    # 追踪
    if len(r.trace) > 1:
        print()
        print(c(Colors.YELLOW + Colors.BOLD, "  ── 纠错追踪 ──"))
        for t in r.trace:
            status_icon = "✅" if (t.result and t.result.success) else "🔄"
            error_info = ""
            if t.result and not t.result.success:
                error_info = f" → {t.result.error_type.value if t.result.error_type else 'error'}"
            elif t.validation and not t.validation.is_valid:
                error_info = " → validation failed"
            print(f"  Round {t.round_idx} {status_icon}{error_info}")

    print(c(Colors.BOLD, "═" * 70))


# ═══════════════════════════════════════════════════════════════
#  交互模式
# ═══════════════════════════════════════════════════════════════

def interactive_mode(agent):
    """交互式 NL2SQL 查询"""
    print()
    print(c(Colors.BOLD + Colors.CYAN, "═" * 60))
    print(c(Colors.BOLD + Colors.CYAN, "  NL2SQL Agent — 交互模式"))
    print(c(Colors.BOLD + Colors.CYAN, "═" * 60))
    print()
    print("  输入自然语言问题，Agent 自动生成 SQL 并查询数据库。")
    print("  /mode auto|simple|plan — 切换执行模式    当前: auto")
    print("  tables — 查看所有表  |  schema <表名> — 查看表结构")
    print("  quit — 退出")
    print()

    interactive_state = {"mode": "auto"}

    while True:
        try:
            question = input(c(Colors.BOLD, "SQL> ")).strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not question:
            continue

        if question.lower() in ("quit", "exit", "q"):
            print("再见！")
            break

        if question.lower().startswith("/mode "):
            new_mode = question[6:].strip()
            if new_mode in ("auto", "simple", "plan"):
                interactive_state["mode"] = new_mode
                print(f"  模式切换: {new_mode}")
                if new_mode == "plan":
                    print("  (复杂问题直接拆解执行，简单问题自动回退到simple)")
            else:
                print(f"  无效模式: {new_mode}。可用: auto / simple / plan")
            print()
            continue

        if question.lower() == "tables":
            tables = agent.schema_provider.get_all_tables()
            print(f"\n  共 {len(tables)} 张表:")
            for t in tables:
                print(f"    • {t}")
            print()
            continue

        if question.lower().startswith("schema "):
            table_name = question[7:].strip()
            ddl = agent.schema_provider.get_table_ddl(table_name)
            print(f"\n{ddl}\n")
            continue

        # 执行查询
        print(c(Colors.DIM, f"  处理中..."))
        start = time.perf_counter()
        result = agent.run(question, mode=interactive_state["mode"])
        print_result(result)


# ═══════════════════════════════════════════════════════════════
#  CLI 入口
# ═══════════════════════════════════════════════════════════════

def main():
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="NL2SQL Agent — 用自然语言查询数据库",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py --question "北京用户买得最多的品类？"
  python main.py --interactive
  python main.py --list-tables
  python main.py --question "每个品类的平均价格" --max-rounds 5
        """,
    )
    parser.add_argument("--question", "-q", type=str, default=None,
                        help="自然语言查询问题")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="交互模式（逐条问答）")
    parser.add_argument("--schema", "-s", type=str, default=SCHEMA_STRATEGY,
                        choices=["full", "keyword", "hybrid"],
                        help=f"Schema 检索策略（默认: {SCHEMA_STRATEGY})")
    parser.add_argument("--max-rounds", "-r", type=int, default=MAX_ROUNDS,
                        help=f"最大纠错轮次（默认: {MAX_ROUNDS})")
    parser.add_argument("--list-tables", "-l", action="store_true",
                        help="列出所有表名")
    parser.add_argument("--list-sources", action="store_true",
                        help="列出所有注册的数据源")
    parser.add_argument("--source", type=str, default=None,
                        help="指定数据源名称（如 '电商主库'），默认使用 config.py 中的 DB_PATH")
    parser.add_argument("--mode", "-m", type=str, default="auto",
                        choices=["auto", "simple", "plan"],
                        help="执行模式: auto=智能选择, simple=直接SQL, plan=Plan-Execute拆解")
    args = parser.parse_args()

    # ── 快速操作: 列出数据源 ──
    if args.list_sources:
        sources = load_sources()
        print(f"\n注册数据源: {len(sources)} 个\n")
        for src in sources:
            print(f"  [{src['name']}]")
            print(f"    类型: {src['db_type']}")
            print(f"    说明: {src['description']}")
            print(f"    表: {src['tables_hint']}")
            print()
        return

    # ── 快速操作: 列出表名 ──
    if args.list_tables:
        from nl2sql_agent.interfaces.database_connector import SQLiteConnector
        conn = SQLiteConnector(DB_PATH, readonly=True)
        if not conn.test_connection():
            print(f"[ERROR] 无法连接数据库: {DB_PATH}")
            sys.exit(1)
        schema_info = conn.get_schema_info()
        print(f"\n数据库: {DB_PATH}")
        print(f"共 {len(schema_info.tables)} 张表:\n")
        for table in schema_info.tables:
            cols = ", ".join(f"{c.name}({c.data_type})" for c in table.columns[:5])
            more = f"... (+{len(table.columns)-5})" if len(table.columns) > 5 else ""
            print(f"  {table.name} ({table.row_count} 行)")
            print(f"    列: {cols}{more}")
            if table.description:
                print(f"    说明: {table.description}")
            print()
        return

    # ── 解析数据源 ──
    db_path = DB_PATH
    if args.source:
        sources = load_sources()
        matched = [s for s in sources if s["name"] == args.source]
        if not matched:
            names = ", ".join(s["name"] for s in sources)
            print(f"[ERROR] 数据源 '{args.source}' 未找到。可用: {names}")
            sys.exit(1)
        db_path = os.path.join(PROJECT_ROOT, matched[0]["db_path"])
        print(f"数据源: {args.source} → {db_path}")

    # ── 初始化 Agent ──
    print(c(Colors.BOLD + Colors.HEADER, "\n初始化 NL2SQL Agent...\n"))
    agent, _, _ = build_agent(args.schema, args.max_rounds, db_path=db_path)
    print(c(Colors.GREEN, "\n初始化完成！\n"))

    # ── 路由 ──
    if args.interactive:
        interactive_mode(agent)
    elif args.question:
        print(f"  {c(Colors.BOLD, '问题:')} {args.question}")
        print(c(Colors.DIM, f"  处理中..."))
        result = agent.run(args.question, mode=args.mode)
        print_result(result)
    else:
        parser.print_help()
        print(f"\n{c(Colors.YELLOW, '提示: 使用 --question 或 --interactive 开始查询')}")


if __name__ == "__main__":
    main()
