"""统一工具注册中心 —— 所有 Agent 共享"""
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
import json


@dataclass
class ToolDef:
    """工具标准定义"""
    name: str
    description: str
    parameters: Dict[str, Any]             # JSON Schema
    handler: Callable
    agent: str = "any"
    risk: str = "low"
    timeout: int = 30
    source: str = "builtin"               # "builtin" | "mcp" | "skill"
    max_retries: int = 0
    idempotent: bool = False


class ToolRegistry:
    """全局唯一工具注册中心"""

    def __init__(self):
        self._tools: Dict[str, ToolDef] = {}

    def register(self, tool: ToolDef, namespace: str = None):
        name = tool.name
        if namespace:
            name = f"{namespace}:{name}"
            tool.name = name
        if name in self._tools:
            # Auto-disambiguate: add _2, _3, ... suffix
            base = name
            counter = 2
            while name in self._tools:
                name = f"{base}_{counter}"
                counter += 1
            tool.name = name
        self._tools[name] = tool

    def unregister(self, name: str):
        self._tools.pop(name, None)

    def unregister_by_prefix(self, prefix: str):
        keys = [k for k in self._tools if k.startswith(prefix)]
        for k in keys:
            del self._tools[k]

    def get(self, name: str) -> Optional[ToolDef]:
        return self._tools.get(name)

    def list_for_agent(self, agent_type: str) -> List[ToolDef]:
        if agent_type == "mcp":
            return [t for t in self._tools.values() if t.source == "mcp"]
        return [t for t in self._tools.values()
                if t.agent == agent_type or t.agent == "any"]

    def list_by_source(self, source: str) -> List[ToolDef]:
        return [t for t in self._tools.values() if t.source == source]

    def to_openai_schema(self, agent_type: str) -> List[Dict]:
        """生成 OpenAI Function Call schema"""
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in self.list_for_agent(agent_type)
        ]

    def execute(self, name: str, arguments: Dict,
                approval_granted: bool = False) -> Dict:
        """统一执行入口：参数校验、风险检查、超时与有界重试。"""
        tool = self._tools.get(name)
        if not tool:
            return {
                "status": "error",
                "code": "tool_not_found",
                "message": f"未知工具: {name}",
            }

        # 参数校验
        required = tool.parameters.get("required", [])
        props = tool.parameters.get("properties", {})
        for p in required:
            if p not in arguments:
                return {
                    "status": "error",
                    "code": "invalid_arguments",
                    "message": f"缺少必填参数: {p}",
                }
        for k in arguments:
            if k not in props:
                return {
                    "status": "error",
                    "code": "invalid_arguments",
                    "message": f"未知参数: {k}",
                }

        if tool.risk in {"high", "critical", "external"} and not approval_granted:
            return {
                "status": "checkpoint",
                "code": "permission_required",
                "risk": tool.risk,
                "requires_approval": True,
                "message": f"工具 {name} 需要批准后执行",
            }

        attempts = 0
        max_attempts = max(1, int(tool.max_retries) + 1)
        while attempts < max_attempts:
            attempts += 1
            executor = ThreadPoolExecutor(max_workers=1)
            future = executor.submit(tool.handler, **arguments)
            try:
                result = future.result(timeout=tool.timeout)
                executor.shutdown(wait=False, cancel_futures=True)
                return {
                    "status": "success",
                    "data": result,
                    "attempts": attempts,
                    "risk": tool.risk,
                }
            except FutureTimeout:
                future.cancel()
                executor.shutdown(wait=False, cancel_futures=True)
                code = "timeout"
                message = f"工具执行超时({tool.timeout}s)"
            except Exception as exc:
                executor.shutdown(wait=False, cancel_futures=True)
                code = (
                    "transient_error"
                    if isinstance(exc, (ConnectionError, TimeoutError))
                    else "tool_error"
                )
                message = str(exc)
            retryable = code == "transient_error" and tool.idempotent
            if not retryable or attempts >= max_attempts:
                return {
                    "status": "error",
                    "code": code,
                    "message": message,
                    "attempts": attempts,
                    "risk": tool.risk,
                }

        raise RuntimeError("unreachable tool execution state")

    def summary(self) -> str:
        lines = [f"已注册 {len(self._tools)} 个工具:"]
        for t in self._tools.values():
            lines.append(f"  [{t.agent}] {t.name} — {t.description[:60]}...")
        return "\n".join(lines)


# 全局单例
registry = ToolRegistry()
