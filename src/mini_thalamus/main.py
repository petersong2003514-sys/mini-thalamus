#!/usr/bin/env python
"""Mini Thalamus —— 统一多 Agent 平台入口

用法:
  python main.py                          # 交互模式
  python main.py "查一下上周销售Top 5"      # 单次查询
  python main.py --multi "分析销售趋势"     # 多步编排模式

架构:
  User → Router → NL2SQL / RAG / Code Agent
              ↕
  Tool Registry + Session Manager + Error Handler
"""

import sys, os, json

from mini_thalamus.router import Router
from mini_thalamus.session_manager import SessionManager
from mini_thalamus.error_handler import ErrorHandler
from mini_thalamus.orchestrator import Orchestrator, MultiStepOrchestrator
from mini_thalamus.config import ROUTER_AGENTS
from mini_thalamus.agent_runtime import get_runtime


def create_agents(verbose=True):
    """Load every available Agent through the shared runtime."""
    runtime = get_runtime()
    agents = runtime.load_all()
    if verbose:
        for agent_id in runtime.list_agent_ids(include_unavailable=True):
            entry = runtime.registry.get(agent_id)
            if agent_id in agents:
                print(f"[Mini Thalamus] {agent_id} Agent 已加载")
            elif entry and entry.get("status") == "unavailable":
                print(
                    f"[Mini Thalamus] {agent_id} Agent 加载失败: "
                    f"{entry.get('error', 'unknown error')}"
                )
    return agents


def interactive_mode(orchestrator, session_mgr):
    """交互式对话模式"""
    print("\n" + "=" * 60)
    print("  Mini Thalamus — 多 Agent 统一平台")
    print("  可用 Agent: " + ", ".join(ROUTER_AGENTS.keys()))
    print("  输入 'quit' 退出, 'status' 查看当前会话")
    print("=" * 60 + "\n")

    session = session_mgr.create("interactive_user")
    print(f"会话: {session.session_id}")

    while True:
        try:
            msg = input("\nYou > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not msg:
            continue
        if msg.lower() in ("quit", "exit", "q"):
            print("再见！")
            break
        if msg.lower() == "status":
            print(f"会话 ID: {session.session_id}")
            print(f"对话轮次: {len(session.messages)}")
            print(f"Agent 历史: {' → '.join(session.agent_history) if session.agent_history else '无'}")
            continue

        print(f"\n[Mini Thalamus] 分析中...")
        result = orchestrator.process(
            msg,
            user_id="interactive_user",
            session_id=session.session_id,
        )

        print(f"[Router] → {result['agent_used']} Agent (confidence: {result['trace'][0].get('confidence', '?')})")
        print(f"\nMini Thalamus > {result['answer']}")


def single_query_mode(orchestrator, question):
    """单次查询模式"""
    result = orchestrator.process(question)
    print(f"[Router → {result['agent_used']}] {result['answer']}")
    if result.get("data"):
        print(f"[数据] {json.dumps(result['data'], ensure_ascii=False, indent=2)}")


def multi_step_mode(orchestrator, question):
    """多步编排模式"""
    if not isinstance(orchestrator, MultiStepOrchestrator):
        orchestrator = MultiStepOrchestrator(
            agents=orchestrator.agents,
            session_mgr=orchestrator.session_mgr,
        )
    result = orchestrator.process_multi(question)
    print(f"[MultiStep] {result['answer']}")


def main(argv=None) -> None:
    """命令行入口。

    argv 默认取 sys.argv[1:]，便于 console script 与测试直接调用。
    """
    argv = list(sys.argv[1:] if argv is None else argv)

    runtime = get_runtime()
    agents = create_agents(verbose=True)

    if not agents:
        print("[Mini Thalamus] 注意: 没有 Agent 成功加载。系统将在 general 模式下运行。")

    session_mgr = SessionManager()
    orchestrator = MultiStepOrchestrator(
        agents=agents,
        session_mgr=session_mgr,
        registry=runtime.registry,
    )

    if argv:
        question = " ".join(argv)
        if "--multi" in argv:
            question = question.replace("--multi", "").strip()
            multi_step_mode(orchestrator, question)
        else:
            single_query_mode(orchestrator, question)
    else:
        interactive_mode(orchestrator, session_mgr)


if __name__ == "__main__":
    main()
