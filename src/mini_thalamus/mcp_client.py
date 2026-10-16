"""MCP Client —— JSON-RPC over stdio, no external SDK dependency

协议流程:
  1. 启动 MCP 服务器子进程
  2. initialize 握手
  3. tools/list 发现工具
  4. tools/call 调用工具

MCP 配置: data/.mcp.json (Claude Code 兼容格式)
"""
import os, sys, json, time, re, shutil, subprocess, threading, asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# 凭证防泄漏
_PKG_ROOT = os.path.dirname(os.path.abspath(__file__))
from mini_thalamus.sandbox import scrub_env


@dataclass
class MCPServerConfig:
    name: str
    command: str
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)


class MCPError(Exception):
    pass


class MCPClientManager:
    """MCP 客户端管理器：后台 asyncio 线程 + 同步接口"""

    def __init__(self, config_path: str):
        self.config_path = config_path
        self.servers: Dict[str, MCPServerConfig] = {}
        self._connections: Dict[str, "_MCPConnection"] = {}
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._ready = False

    # ── lifecycle ──
    def load_config(self) -> bool:
        """解析 data/.mcp.json，返回是否有有效配置"""
        if not os.path.exists(self.config_path):
            print("  [MCP] 无配置文件，跳过 MCP 初始化")
            return False
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception as e:
            print(f"  [MCP] 配置文件解析失败: {e}")
            return False

        servers = cfg.get("mcpServers", {})
        if not servers:
            return False

        for name, srv in servers.items():
            if srv.get("type", "stdio") != "stdio":
                print(f"  [MCP] {name}: 仅支持 stdio 类型，跳过")
                continue
            env = {}
            for k, v in srv.get("env", {}).items():
                # 环境变量替换 ${VAR}
                resolved = v
                for m in re.finditer(r'\$\{(\w+)\}', v):
                    resolved = resolved.replace(m.group(0), os.environ.get(m.group(1), ""))
                env[k] = resolved
            self.servers[name] = MCPServerConfig(
                name=name,
                command=srv["command"],
                args=srv.get("args", []),
                env=env,
            )

        print(f"  [MCP] 加载 {len(self.servers)} 个服务器配置: {list(self.servers.keys())}")
        return len(self.servers) > 0

    def start(self):
        """启动后台 asyncio 线程"""
        if self._thread is not None:
            return
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self._ready = True
        print("  [MCP] 后台事件循环已启动")

    def stop(self):
        """关闭所有连接和事件循环"""
        for name in list(self._connections.keys()):
            self._disconnect(name)
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=3)
        self._ready = False
        print("  [MCP] 已关闭")

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    # ── 连接管理 ──
    def connect_all(self) -> dict:
        results = {}
        for name in self.servers:
            results[name] = self.connect_server(name)
        return results

    def connect_server(self, name: str) -> bool:
        if name not in self.servers:
            print(f"  [MCP] {name}: 未配置")
            return False
        if name in self._connections:
            return True  # 已连接

        srv = self.servers[name]
        try:
            conn = _MCPConnection(srv)
            # 同步等待连接完成
            future = asyncio.run_coroutine_threadsafe(conn.connect(), self._loop)
            future.result(timeout=30)
            self._connections[name] = conn
            print(f"  [MCP] {name}: 已连接, {len(conn.tools)} 个工具")
            return True
        except Exception as e:
            print(f"  [MCP] {name}: 连接失败 - {e}")
            return False

    def _disconnect(self, name: str):
        conn = self._connections.pop(name, None)
        if conn:
            try:
                future = asyncio.run_coroutine_threadsafe(conn.close(), self._loop)
                future.result(timeout=5)
            except Exception:
                pass

    # ── 工具访问 ──
    def get_all_tools(self) -> List[dict]:
        tools = []
        for name, conn in self._connections.items():
            for t in conn.tools:
                t_copy = dict(t)
                t_copy["server"] = name
                tools.append(t_copy)
        return tools

    def call_tool_sync(self, server_name: str, tool_name: str, arguments: dict) -> dict:
        """同步调用 MCP 工具（阻塞等待异步结果）"""
        conn = self._connections.get(server_name)
        if not conn:
            # 尝试重连
            if not self.connect_server(server_name):
                return {"ok": False, "error": f"MCP 服务器 '{server_name}' 未连接"}
            conn = self._connections[server_name]

        try:
            future = asyncio.run_coroutine_threadsafe(
                conn.call_tool(tool_name, arguments), self._loop
            )
            return future.result(timeout=60)
        except Exception as e:
            return {"ok": False, "error": str(e)}


class _MCPConnection:
    """单个 MCP 服务器的连接（运行在后台事件循环中）"""

    def __init__(self, config: MCPServerConfig):
        self.config = config
        self.process: Optional[subprocess.Popen] = None
        self.tools: List[dict] = []
        self._next_id = 1
        self._reader_lock = None  # asyncio.Lock

    async def connect(self):
        env = scrub_env()
        env.update(self.config.env)
        command = self.config.command
        if os.name == "nt" and not os.path.splitext(command)[1]:
            resolved = shutil.which(command)
            if not resolved:
                resolved = shutil.which(command + ".cmd")
            if resolved:
                command = resolved
        self.process = subprocess.Popen(
            [command] + self.config.args,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=env, text=True, encoding="utf-8", bufsize=1,
        )
        # Initialize
        result = await self._request("initialize", {
            "protocolVersion": "2025-03-26",
            "capabilities": {},
            "clientInfo": {"name": "mini-thalamus", "version": "1.0"},
        })
        if "error" in result:
            raise MCPError(f"initialize 失败: {result['error']}")
        # Send initialized notification
        await self._notify("notifications/initialized", {})
        # List tools
        tools_result = await self._request("tools/list", {})
        self.tools = tools_result.get("result", {}).get("tools", [])

    async def call_tool(self, tool_name: str, arguments: dict) -> dict:
        result = await self._request("tools/call", {
            "name": tool_name,
            "arguments": arguments,
        })
        if "error" in result:
            return {"ok": False, "error": result["error"].get("message", str(result["error"]))}
        content = result.get("result", {}).get("content", [])
        text = "\n".join(c.get("text", str(c)) for c in content if c.get("type") == "text")
        return {"ok": True, "data": text or content}

    async def _request(self, method: str, params: dict) -> dict:
        req_id = self._next_id
        self._next_id += 1
        msg = {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params}
        return await self._send_recv(msg)

    async def _notify(self, method: str, params: dict):
        msg = {"jsonrpc": "2.0", "method": method, "params": params}
        self._send(msg)

    def _send(self, msg: dict):
        line = json.dumps(msg, ensure_ascii=False) + "\n"
        self.process.stdin.write(line)
        self.process.stdin.flush()

    async def _send_recv(self, msg: dict) -> dict:
        self._send(msg)
        # 读取响应（逐行读，跳过非 JSON 行）
        loop = asyncio.get_event_loop()
        try:
            line = await asyncio.wait_for(
                loop.run_in_executor(None, self.process.stdout.readline),
                timeout=30)
        except asyncio.TimeoutError:
            raise MCPError(f"请求超时: {msg.get('method', '?')}")
        if not line:
            raise MCPError("服务器进程已退出")
        line = line.strip()
        while line:
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                # 尝试读下一行（跳过非 JSON 日志行）
                try:
                    line = await asyncio.wait_for(
                        loop.run_in_executor(None, self.process.stdout.readline),
                        timeout=5)
                except asyncio.TimeoutError:
                    raise MCPError(f"请求超时: {msg.get('method', '?')}")
                if not line:
                    raise MCPError("服务器进程已退出")
                line = line.strip()
        raise MCPError(f"请求超时: {msg.get('method', '?')}")

    async def close(self):
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None
