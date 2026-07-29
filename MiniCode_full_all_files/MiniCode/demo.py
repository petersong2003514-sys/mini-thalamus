"""ShuShu Agent 交互式 Demo — 真实 NL2SQL + 模拟 RAG + 真实 Code Agent"""
import sys, os, time, json, io
_MINICODE_ROOT = os.path.dirname(os.path.abspath(__file__))
_BUNDLE_ROOT = os.path.dirname(_MINICODE_ROOT)
if _MINICODE_ROOT not in sys.path:
    sys.path.insert(0, _MINICODE_ROOT)
if _BUNDLE_ROOT not in sys.path:
    sys.path.append(_BUNDLE_ROOT)

# Windows 终端 UTF-8
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from router import Router
from session_manager import SessionManager
from error_handler import ErrorHandler, AgentError
from agents.code_agent import CodeAgent
from agent_runtime import get_runtime


# ===== 加载真实 Agent（可选，加载失败降级为 Mock）=====
REAL_AGENTS = {}

# NL2SQL
try:
    print("[ShuShu Agent] Loading NL2SQL Agent (real)...")
    REAL_AGENTS["nl2sql"] = get_runtime().get_agent("nl2sql")
    print("[ShuShu Agent] NL2SQL Agent OK — ecommerce.db, 5 tables, DeepSeek API")
except Exception as e:
    print(f"[ShuShu Agent] NL2SQL load failed: {e}")
    print("[ShuShu Agent] Using Mock NL2SQL instead")
    from demo_mock import MockNL2SQL
    REAL_AGENTS["nl2sql"] = MockNL2SQL()

# RAG（暂用 Mock，后续替换）
try:
    REAL_AGENTS["rag"] = get_runtime().get_agent("rag")
    print("[ShuShu Agent] RAG Agent OK")
except Exception:
    from demo_mock import MockRAG
    REAL_AGENTS["rag"] = MockRAG()
    print("[ShuShu Agent] RAG Agent — Mock mode")

# Code
REAL_AGENTS["code"] = CodeAgent()
print("[ShuShu Agent] Code Agent OK (sandbox)")
print()


# ===== ShuShu Agent 交互循环 =====
def run():
    router = Router()
    session_mgr = SessionManager()
    code_agent = REAL_AGENTS["code"]
    agents = REAL_AGENTS

    session = session_mgr.create("demo_user")

    print("+--------------------------------------------------+")
    print("|         ShuShu Agent — Multi-Agent Platform           |")
    print(f"|  NL2SQL: {'REAL' if agents['nl2sql'].__class__.__name__ == 'NL2SQLPeerAgent' else 'MOCK'} | RAG: READY | Code: REAL        |")
    print("|  Type 'quit' to exit                             |")
    print("+--------------------------------------------------+")
    print(f"  Session: {session.session_id}")
    print()

    while True:
        try:
            msg = input("You > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not msg:
            continue
        if msg.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break

        # Step 1: Route
        route = router.route(msg, session.get_context_for("router"))
        agent_type = route["agent"]
        print(f"  [Router] -> {agent_type} (confidence: {route.get('confidence', '?')})")

        # Step 2: Execute
        if agent_type == "general":
            print(f"  [General] 试试: '查一下北京销量Top 5' 或 '帮我分析销售数据'")
            continue

        agent = agents.get(agent_type)
        if not agent:
            print(f"  [Error] Agent '{agent_type}' not available")
            continue

        if agent_type == "code":
            # Code Agent: 自动生成并执行分析代码
            code = _gen_code(msg, session)
            print(f"  [Code] executing...")
            c_result = code_agent.run(code)
            output = c_result.get("stdout", c_result.get("stderr", ""))
            print(f"  [Code] output:\n{output}")
            session.add_agent_trace("code", "executed", output[:200])
        else:
            # NL2SQL / RAG
            print(f"  [{agent_type.upper()}] thinking...")
            result = agent.run(msg)
            status = result.get("status", "?")
            answer = result.get("answer", result.get("summary", ""))

            if status == "success":
                print(f"  [{agent_type.upper()}] {answer}")
            else:
                print(f"  [{agent_type.upper()}] FAILED: {answer}")
                # 降级
                err = AgentError(agent_type, "exec_failed", answer)
                fb = ErrorHandler().handle(agent_type, err, session)
                if fb["action"] == "retry":
                    print(f"  [ErrorHandler] retrying...")
                    result = agent.run(msg)
                    print(f"  [{agent_type.upper()}] {result.get('answer','')}")

            session.add_agent_trace(agent_type, "完成", result.get("summary", ""))
            session.set_last_result(result)

            # 自动串联：NL2SQL 查完了 -> Code 分析
            if agent_type == "nl2sql" and _needs_code(msg):
                print("  [Orchestrator] NL2SQL done, auto-generating analysis...")
                code = _gen_code(msg, session)
                c_result = code_agent.run(code)
                output = c_result.get("stdout", c_result.get("stderr", ""))
                print(f"  [Code] output:\n{output}")

        print()


def _needs_code(msg: str) -> bool:
    return any(kw in msg for kw in ["分析", "趋势", "画图", "图表", "可视化", "代码"])


def _gen_code(question: str, session) -> str:
    """根据 NL2SQL 返回的真实数据生成分析代码"""
    last = session.last_result
    if not last:
        return "print('请先查询数据，如: 查一下北京销量Top 5')"

    data = last.get("data", {})
    rows = data.get("rows", [])
    sql = data.get("sql", "")

    if not rows:
        return "print('查询结果为空，无法分析')"

    # 把真实数据转成 Python 变量
    rows_json = json.dumps([list(r) for r in rows], ensure_ascii=False)
    return f'''# Auto-generated analysis code
# SQL: {sql[:80]}...
import json

rows = {rows_json}

print(f"共 {{len(rows)}} 条记录:")
for i, row in enumerate(rows, 1):
    print(f"  {{i}}. {{row}}")

# 如果有数值列，做简单统计
try:
    # 尝试把最后一列当数值
    values = []
    for r in rows:
        vals = [v for v in r if isinstance(v, (int, float))]
        if vals:
            values.append(vals[-1])
    if values:
        print(f"\\n最大值: {{max(values)}}")
        print(f"最小值: {{min(values)}}")
        print(f"平均值: {{sum(values)/len(values):.1f}}")
except Exception as e:
    print(f"统计失败: {{e}}")
'''


if __name__ == "__main__":
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        router = Router()
        session_mgr = SessionManager()
        session = session_mgr.create("cli_user")
        agents = REAL_AGENTS

        print(f"You > {query}")
        route = router.route(query)
        print(f"[Router] -> {route['agent']} (confidence: {route.get('confidence', '?')})")

        agent = agents.get(route["agent"])
        if agent and route["agent"] != "general":
            result = agent.run(query)
            print(f"[{route['agent'].upper()}] {result.get('answer', result.get('summary', ''))}")

            if route["agent"] == "nl2sql" and _needs_code(query):
                session.set_last_result(result)
                print("\n[Orchestrator] Auto-generating analysis code...")
                code = _gen_code(query, session)
                c_result = agents["code"].run(code)
                print(f"[Code]\n{c_result.get('stdout', c_result.get('stderr', ''))}")
        else:
            print("[General] 试试: '查一下北京销量Top 5'")
    else:
        run()
