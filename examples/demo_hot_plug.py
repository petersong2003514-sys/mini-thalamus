#!/usr/bin/env python3
"""演示：进程不重启，运行时完成子 Agent 的 新增 / 替换 / 移除。

全程只用一个 AgentRegistry 对象，中间不重新构造、不重启进程。
运行: python demo_hot_plug.py
"""

import os
import sys
import tempfile



import mini_thalamus.agent_registry as agent_registry
from mini_thalamus.agent_registry import AgentLoadError, AgentRegistry


AGENT_SOURCE = '''\
AGENT_SPEC = {{
    "id": "{agent_id}",
    "name": "{agent_id} Agent",
    "description": "版本 {version}",
    "factory": "agents.{agent_id}_agent:create_agent",
}}

VERSION = "{version}"


class ProbeAgent:
    def __init__(self):
        self.capabilities = {{}}

    def run(self, query, context=None):
        return {{
            "answer": f"{{VERSION}} 处理了「{{query}}」",
            "status": "success",
            "data": {{}},
        }}


def create_agent():
    return ProbeAgent()
'''


def banner(title):
    print(f"\n{'=' * 62}\n{title}\n{'=' * 62}")


def write_agent(directory, agent_id, version, broken=False):
    source = AGENT_SOURCE.format(agent_id=agent_id, version=version)
    if broken:
        source += "\n这里故意写一行语法错误的 ( python\n"
    path = os.path.join(directory, f"{agent_id}_agent.py")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(source)
    return path


def ask(registry, agent_id, query):
    instance = registry.get_or_create(agent_id)
    return instance.run(query)["answer"]


def main():
    seen = []
    registry = AgentRegistry()
    with tempfile.TemporaryDirectory() as agents_dir:
        registry._agents_dir = agents_dir  # 演示用临时目录
        # Router 就是这么被通知的：注册/移除都会回调它重新读取实时配置。
        registry.on_register(
            lambda agent_id, meta: seen.append(
                (agent_id, "移除" if meta is None else "注册/更新")
            )
        )

        banner("① 运行时新增：把文件丢进 agents/，调 rescan()，无需重启")
        write_agent(agents_dir, "probe", "v1")
        write_agent(agents_dir, "helper", "v1")
        print("rescan ->", registry.rescan())
        print("当前可用 Agent:", sorted(registry.list_all()))
        print("probe 返回:", ask(registry, "probe", "测试"))

        banner("② 运行时替换：改磁盘上的实现，热替换掉正在跑的版本")
        in_flight = registry.get_or_create("probe")  # 一个"在途请求"持有的实例
        write_agent(agents_dir, "probe", "v2")
        print("reload ->", registry.reload_agent("probe"))
        print("新请求  probe 返回:", ask(registry, "probe", "测试"))
        print("在途实例 probe 返回:", in_flight.run("测试")["answer"],
              "   <- 全程未被打断，仍跑它启动时的那版代码")
        print("Router 看到的描述:", registry.get_router_config()["probe"]["description"])

        banner("③ 失败的替换：改坏了只报错，旧版本原样继续服务")
        write_agent(agents_dir, "probe", "v3", broken=True)
        try:
            registry.reload_agent("probe")
        except AgentLoadError as exc:
            print("reload 被拒绝 ->", exc)
        print("probe 仍然可用:", ask(registry, "probe", "测试"))

        banner("④ 运行时移除：删掉源文件，rescan 自动注销")
        os.remove(os.path.join(agents_dir, "helper_agent.py"))
        print("rescan ->", registry.rescan())
        print("当前可用 Agent:", sorted(registry.list_all()))
        print("注意：probe 的源文件还在磁盘上（只是解析不了），所以它没有被注销——")
        print("      一次坏编辑不该把一个还在正常服务的 Agent 干掉。")

        banner("⑤ 订阅者收到的事件（Router 靠它自动刷新路由表）")
        for agent_id, event in seen:
            print(f"  {event:<8} {agent_id}")

        banner("⑥ 内容没变：不重建，热实例保留")
        write_agent(agents_dir, "probe", "v2")  # 把源文件恢复到与当前运行版本一致
        warm = registry.get_or_create("probe")
        print("rescan ->", registry.rescan())
        print("实例仍是同一个对象:", registry.get_or_create("probe") is warm)

        banner("⑦ 故障 → 熔断 → 移出候选队列 → 冷却 → 半开探测 → 恢复")
        for _ in range(agent_registry.CIRCUIT_FAILURE_THRESHOLD):
            registry.record_failure("probe", "数据库连接超时")
        print("健康度        ->", registry.health_of("probe"))
        print("还在候选队列吗 ->", "probe" in registry.list_all())
        print("Router 还选它吗 ->", "probe" in registry.get_router_config())
        print("条目还在吗     ->", registry.get("probe") is not None,
              "  <- 隐藏，不是删除")

        original = agent_registry.CIRCUIT_COOLDOWN_SECONDS
        agent_registry.CIRCUIT_COOLDOWN_SECONDS = 0.0
        print("\n冷却结束后：")
        print("重新进候选队列 ->", "probe" in registry.list_all())
        registry.get_or_create("probe")            # 放一个半开探测请求
        print("探测中的状态   ->", registry.health_of("probe")["state"])
        registry.record_success("probe")           # 探测成功
        print("恢复后的状态   ->", registry.health_of("probe")["state"])
        agent_registry.CIRCUIT_COOLDOWN_SECONDS = original

    print("\n全程未重启进程、未重新构造 AgentRegistry。")


if __name__ == "__main__":
    main()
