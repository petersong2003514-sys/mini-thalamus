"""Shell Agent —— 执行 Shell 命令（ReAct loop + 安全分级 + 流式输出）

FEATURE_PLAN.md §2。ReAct 循环：LLM 出 tool_call → risk 分级 →
blocked/checkpoint/auto/exec → 流式 SSE stdout/stderr 返回。

安全层次：
  1. HARD_BLOCK_PATTERNS   永远拒绝，不进 LLM
  2. CONFIRM_PATTERNS/sudo 高风险，等用户二次确认
  3. SHELL_CONFIRM_MEDIUM  env 控制中风险是否自动执行（默认自动）
  4. SECRET_PATTERNS       stdout/stderr 脱敏后再入 history.json
  5. cwd_allowed           cwd 白名单，拒绝越权路径
"""
from __future__ import annotations

import os
import re
import sys
import shutil
import tempfile
import subprocess
import time
import importlib.util
from typing import Optional, Iterator

# ─── Config 显式路径加载 ───
_MINICODE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _MINICODE_ROOT not in sys.path:
    sys.path.insert(0, _MINICODE_ROOT)
_cfg_path = os.path.join(_MINICODE_ROOT, "config.py")
_spec = importlib.util.spec_from_file_location("shell_config", _cfg_path)
_cfg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cfg)
LLM_CONFIG = _cfg.LLM_CONFIG

from sandbox import split_argv, make_preexec_fn, kill_process_tree, scrub_env

AGENT_META = {
    "name": "Shell Agent",
    "id": "shell",
    "description": "执行 Shell 命令：文件操作、git、npm、系统信息、脚本运行",
    "keywords": ["shell", "bash", "命令", "终端", "git", "npm", "执行", "运行脚本"],
    "icon": "🐚",
    "capabilities": {
        "needs_history": False,
        "needs_files": False,
        "needs_long_term": False,
        "max_context_chars": 4000,
        "auto_chainable": False,
        "side_effects": "external",
        "needs_credentials": [],
        "actions": ["execute_shell"],
        "can_handle_errors": [],
    },
    "input_schema": {"required": ["instruction"], "properties": {}},
    "output_schema": {"exports": ["answer", "summary", "data.structured"]},
}

# ─── 永远拒绝（不进 UI、不进 LLM）───
HARD_BLOCK_PATTERNS: list[re.Pattern] = [re.compile(p, re.IGNORECASE) for p in [
    r"rm\s+-[a-z]*r[a-z]*\s+/(?:\s|$)",          # rm -rf /
    r":\(\)\s*\{[^}]*:\s*&\s*\}",                  # fork bomb :(){:|:&};:
    r"dd\s+if=",                                    # dd if= (disk write)
    r"mkfs\b",                                      # 格式化
    r">\s*/dev/s[a-z]",                             # >/dev/sda 覆盖磁盘
    r"chmod\s+-[a-z]*R[a-z]*\s+777\s+/(?:\s|$)",   # chmod -R 777 /
    r"shred\b.*--zero",                             # 安全擦除整盘
    r"wipefs\b",                                    # 擦除文件系统签名
]]

# ─── 高风险（要二次确认）───
CONFIRM_PATTERNS: list[re.Pattern] = [re.compile(p, re.IGNORECASE) for p in [
    r"rm\s+-[a-z]*r",                              # rm -r* (非/)
    r"git\s+push\s+.*--force",                     # git push --force
    r"npm\s+publish\b",
    r"pip\s+uninstall\b",
    r"kubectl\s+delete\b",
    r"docker\s+rm\b",
    r"systemctl\s+(stop|disable|mask)\b",
    r"kill\s+-9\b",
    r"pkill\b",
    r"truncate\b",
    r">/dev/",                                      # 写设备（non-sda）
    r"ssh\b.*-o.*StrictHostKeyChecking=no",
]]

# ─── stdout/stderr 脱敏（R10）───
SECRET_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"AKIA[0-9A-Z]{16}", re.I), "[AWS_KEY_REDACTED]"),
    (re.compile(r"sk-[A-Za-z0-9]{20,}", re.I), "[SK_KEY_REDACTED]"),
    (re.compile(r"ghp_[A-Za-z0-9]{36,}", re.I), "[GH_TOKEN_REDACTED]"),
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
                re.S), "[PRIVATE_KEY_REDACTED]"),
    (re.compile(r"Bearer\s+[A-Za-z0-9\-._~+/=]{20,}", re.I), "Bearer [TOKEN_REDACTED]"),
    (re.compile(r"(?i)(?:api[_-]?key|secret|token|password)\s*[:=]\s*\S+"), "[CREDENTIAL_REDACTED]"),
]

# ─── 允许的工作目录白名单 ───
SHELL_CWD_ALLOW: list[str] = [
    os.getcwd(),
    tempfile.gettempdir(),
]

_SHELL_CONFIRM_MEDIUM = os.environ.get("SHELL_CONFIRM_MEDIUM", "0") == "1"

# ─── 只读低风险命令前缀 ───
_LOW_RISK_PREFIXES = (
    "ls", "cat ", "echo ", "pwd", "whoami", "date", "which ", "type ",
    "head ", "tail ", "wc ", "grep ", "find ", "stat ", "file ",
    "git status", "git log", "git diff", "git branch", "git show",
    "git remote", "git fetch",
    "env", "printenv", "df ", "du ", "ps ", "top ", "uname",
    "python3 --version", "python --version", "node --version",
    "npm --version", "pip --version",
)

_MEDIUM_PATTERNS: list[re.Pattern] = [re.compile(p, re.IGNORECASE) for p in [
    r"pip\s+install\b",
    r"apt(-get)?\s+install\b",
    r"brew\s+install\b",
    r"npm\s+install\b",
    r"curl\b",
    r"wget\b",
    r"git\s+clone\b",
    r"git\s+push\b",
    r"ssh\b",
    r"scp\b",
    r"rsync\b",
]]


def _scrub_output(text: str) -> str:
    for pat, repl in SECRET_PATTERNS:
        text = pat.sub(repl, text)
    return text


def _classify_risk(cmd: str) -> str:
    """返回 'blocked' | 'high' | 'medium' | 'low'"""
    stripped = cmd.strip()
    for pat in HARD_BLOCK_PATTERNS:
        if pat.search(stripped):
            return "blocked"
    for pat in CONFIRM_PATTERNS:
        if pat.search(stripped):
            return "high"
    if re.search(r"\bsudo\b", stripped):
        return "high"
    for pat in _MEDIUM_PATTERNS:
        if pat.search(stripped):
            return "medium"
    low = stripped.lower()
    if any(low.startswith(p) for p in _LOW_RISK_PREFIXES):
        return "low"
    return "medium"


def _cwd_allowed(cwd: str) -> bool:
    try:
        real = os.path.realpath(cwd)
    except Exception:
        return False
    # 动态构建允许列表：模块级 SHELL_CWD_ALLOW + 当前工作目录（防 import 时固化）
    dynamic_allow = list(SHELL_CWD_ALLOW)
    try:
        dynamic_allow.append(os.getcwd())
    except Exception:
        pass
    for allowed in dynamic_allow:
        try:
            if real.startswith(os.path.realpath(allowed)):
                return True
        except Exception:
            pass
    return False


def _pre_validate_cmd(cmd: str) -> dict:
    """检查命令引用的脚本文件是否存在，不存在则 early-return 让 MetaReasoner 介入。"""
    m = re.search(r'(?<![\w-])(?:bash|sh)\s+([^\s;|&>]+)', cmd)
    if not m:
        m2 = re.search(r'\./([^\s;|&>]+)', cmd)
        if not m2:
            return None
        filepath = m2.group(0)
    else:
        filepath = m.group(1)
    try:
        if not os.path.exists(filepath) and not filepath.startswith("http"):
            return {
                "status": "error",
                "code": "file_not_found",
                "answer": f"文件不存在: {filepath}，无法执行命令。请先确认文件路径或使用其他方式。",
                "data": {"cmd": cmd, "missing_file": filepath},
            }
    except Exception:
        pass
    return None


def _stream_subprocess(
    cmd: str,
    cwd: str,
    env: dict,
    timeout: int,
    event_sink: list,         # 收集 SSE dict
) -> dict:
    """执行命令，流式收集 stdout/stderr，返回 {status, exit_code, stdout, stderr}"""
    bash = shutil.which("bash")
    if not bash:
        return {"status": "error", "exit_code": -1,
                "stdout": "", "stderr": "bash 不可用（Windows 兜底）",
                "error_code": "cwd_denied"}

    argv = [bash, "-c", cmd]
    preexec = make_preexec_fn()

    stdout_lines: list[str] = []
    stderr_lines: list[str] = []

    try:
        proc = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=cwd,
            env=env,
            preexec_fn=preexec,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except Exception as e:
        return {"status": "error", "exit_code": -1, "stdout": "", "stderr": str(e)}

    import threading

    def _reader(pipe, lines, subtype):
        for raw in pipe:
            line = _scrub_output(raw.rstrip("\n"))
            lines.append(line)
            event_sink.append({
                "type": "shell_output",
                "subtype": subtype,
                "text": line,
                "agent": "shell",
            })
        pipe.close()

    t_out = threading.Thread(target=_reader, args=(proc.stdout, stdout_lines, "stdout"), daemon=True)
    t_err = threading.Thread(target=_reader, args=(proc.stderr, stderr_lines, "stderr"), daemon=True)
    t_out.start()
    t_err.start()

    try:
        proc.wait(timeout=timeout)
        t_out.join(timeout=2)
        t_err.join(timeout=2)
    except subprocess.TimeoutExpired:
        kill_process_tree(proc)
        t_out.join(timeout=1)
        t_err.join(timeout=1)
        return {"status": "timeout", "exit_code": -1,
                "stdout": "\n".join(stdout_lines),
                "stderr": f"命令超时（{timeout}s）"}

    return {
        "status": "success" if proc.returncode == 0 else "error",
        "exit_code": proc.returncode,
        "stdout": "\n".join(stdout_lines),
        "stderr": "\n".join(stderr_lines),
    }


class ShellAgent:
    """Shell 命令执行 Agent（ReAct loop，安全分级）"""

    capabilities = AGENT_META["capabilities"]

    def __init__(self):
        # 懒加载 LLM（单测不需要，且无 openai 时静默失败）
        self._client = None
        self._register_tools_once()

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=LLM_CONFIG["api_key"],
                base_url=LLM_CONFIG["base_url"],
            )
        except Exception:
            pass
        return self._client

    @staticmethod
    def _register_tools_once():
        try:
            from tool_registry import registry, ToolDef
            if registry.get("shell.exec") is not None:
                return

            def _shell_exec_handler(cmd: str, cwd: str = None, timeout: int = 30,
                                    event_sink=None, **_):
                if event_sink is None:
                    risk = _classify_risk(cmd)
                    if risk == "high":
                        return {"status": "blocked",
                                "error": "shell.exec: high-risk cmd requires event_sink (R17)"}
                agent = ShellAgent.__new__(ShellAgent)
                agent._client = None
                return agent._exec_single(cmd, cwd=cwd, timeout=timeout,
                                          event_sink=event_sink or [])

            registry.register(ToolDef(
                name="shell.exec",
                description="在沙箱中执行一行 Shell 命令，返回 stdout/stderr/exit_code。高风险命令必须经用户确认。",
                parameters={
                    "type": "object",
                    "properties": {
                        "cmd":     {"type": "string"},
                        "cwd":     {"type": "string"},
                        "timeout": {"type": "integer", "default": 30},
                    },
                    "required": ["cmd"],
                },
                handler=_shell_exec_handler,
                agent="shell",
                risk="high",
                source="builtin",
            ))
        except Exception:
            pass  # 单测环境无 tool_registry 时静默

    def _scrub_env_for_agent(self, cwd: str) -> dict:
        """清理 env，只保留安全变量，注入 agent credentials"""
        env = scrub_env(
            extra_drop=[k for k in os.environ if k.endswith("_API_KEY") or k.endswith("_TOKEN")]
        )
        # 白名单保留
        keep = {"PATH", "HOME", "LANG", "LC_ALL", "LC_CTYPE", "TERM", "USER",
                "LOGNAME", "SHELL", "TMPDIR", "TMP", "TEMP", "PYTHONIOENCODING",
                "XDG_RUNTIME_DIR", "HOSTNAME"}
        env = {k: v for k, v in env.items() if k in keep or k.startswith("SHELL_")}
        # 只注入 Agent 显式声明的 credentials；空白名单意味着不注入。
        try:
            from credentials import get_store
            cred_store = get_store()
            names = self.capabilities.get("needs_credentials", [])
            if names:
                env.update(cred_store.get_env_dict("shell", names=names))
        except Exception:
            pass
        return env

    def _exec_single(self, cmd: str, cwd: str = None, timeout: int = 30,
                     event_sink: list = None) -> dict:
        """执行单条命令并返回结果 dict"""
        if event_sink is None:
            event_sink = []
        cwd = cwd or os.getcwd()
        if not _cwd_allowed(cwd):
            return {
                "status": "error",
                "error_code": "cwd_denied",
                "stdout": "",
                "stderr": f"工作目录不在允许范围内: {cwd}",
            }
        env = self._scrub_env_for_agent(cwd)
        return _stream_subprocess(cmd, cwd=cwd, env=env, timeout=timeout,
                                  event_sink=event_sink)

    def run(self, query: str, context: dict = None) -> dict:
        """主入口：从 query 中提取命令并执行（ReAct loop，最多 6 步）"""
        context = context or {}
        event_sink: list = context.get("_event_sink", [])
        cwd = context.get("cwd") or os.getcwd()
        max_steps = 6

        # 先尝试直接从 query 提取命令
        cmd = self._extract_command(query)
        if not cmd:
            # LLM 生成命令
            cmd = self._llm_plan(query, context)
        if not cmd:
            return {
                "status": "error",
                "answer": "无法从描述中生成有效命令，请直接输入 Shell 命令。",
            }

        # 方案E: 预执行验证 —— 脚本文件不存在直接 early-return 让 MetaReasoner 介入
        pre_check = _pre_validate_cmd(cmd)
        if pre_check:
            return pre_check

        all_stdout: list[str] = []
        all_stderr: list[str] = []

        for step in range(max_steps):
            risk = _classify_risk(cmd)

            if risk == "blocked":
                answer = f"命令被永久拒绝（安全规则）: `{cmd}`"
                return {"status": "blocked", "code": "hard_blocked", "answer": answer}

            if risk == "high":
                return {
                    "status": "checkpoint",
                    "code": "shell_confirm",
                    "answer": f"以下命令为高风险操作，需要您确认：\n```\n{cmd}\n```",
                    "data": {"cmd": cmd, "risk": risk, "subtype": "shell_confirm"},
                }

            if risk == "medium" and _SHELL_CONFIRM_MEDIUM:
                return {
                    "status": "checkpoint",
                    "code": "shell_confirm",
                    "answer": f"以下命令需要确认（SHELL_CONFIRM_MEDIUM=1）：\n```\n{cmd}\n```",
                    "data": {"cmd": cmd, "risk": risk, "subtype": "shell_confirm"},
                }

            # low / medium（自动） → 执行
            result = self._exec_single(cmd, cwd=cwd, event_sink=event_sink)
            if result.get("stdout"):
                all_stdout.append(result["stdout"])
            if result.get("stderr"):
                all_stderr.append(result["stderr"])

            # 最后一步或执行成功则返回
            if result["status"] in ("success", "timeout") or step == max_steps - 1:
                break

            # 失败时尝试 LLM 自修复
            fix_cmd = self._llm_fix(cmd, result.get("stderr", ""), query)
            if not fix_cmd or fix_cmd == cmd:
                break
            cmd = fix_cmd

        stdout_text = "\n".join(all_stdout)
        stderr_text = "\n".join(all_stderr)
        status = result.get("status", "error")

        if status == "success":
            answer = stdout_text.strip() if stdout_text.strip() else "命令执行成功（无输出）"
        elif status == "timeout":
            answer = f"命令执行超时: {stderr_text[:200]}"
        else:
            answer = (f"命令执行出错（exit {result.get('exit_code', '?')}）:\n"
                      f"{stderr_text[:500] or stdout_text[:300]}")

        return {
            "status": status,
            "answer": answer,
            "data": {
                "structured": {
                    "format": "shell_result",
                    "data": {
                        "cmd": cmd,
                        "stdout": stdout_text[:2000],
                        "stderr": stderr_text[:500],
                        "exit_code": result.get("exit_code", -1),
                        "risk": risk,
                    },
                    "sources": [],
                    "confidence": "high" if status == "success" else "low",
                }
            },
        }

    def _extract_command(self, query: str) -> str:
        """从用户输入中提取 Shell 命令（含 ``` 代码块或 $ 前缀）"""
        m = re.search(r"```(?:bash|sh|shell)?\s*\n?(.*?)```", query, re.S)
        if m:
            return m.group(1).strip()
        m = re.search(r"^\$\s+(.+)$", query, re.M)
        if m:
            return m.group(1).strip()
        # 整行看起来像命令（不含中文、非疑问句）
        stripped = query.strip()
        if (stripped and " " in stripped
                and not re.search(r"[一-鿿？！]", stripped)
                and not stripped.endswith("？")):
            return stripped
        return ""

    def _llm_plan(self, query: str, context: dict) -> str:
        client = self._get_client()
        if not client:
            return ""
        try:
            from token_budget import max_tokens_for
            resp = client.chat.completions.create(
                model=LLM_CONFIG["model"],
                messages=[
                    {"role": "system",
                     "content": ("你是 Shell 专家。将用户需求转换为 Shell 命令。"
                                 "只输出命令本身，不要解释，不要 markdown。"
                                 "⚠️ 严格根据上一步提供的文档内容执行，禁止编造不存在的包名(pip install xxx)、"
                                 "命令名或工具。如果文档里写的是 bash bin/xxx.sh，就输出 bash bin/xxx.sh，"
                                 "不要自己发明 npm/pip/brew 安装命令。"
                                 "如果无法安全完成，输出 CANNOT_DO。")},
                    {"role": "user", "content": query[:8000]},
                ],
                temperature=0.0,
                max_tokens=200,
            )
            cmd = resp.choices[0].message.content.strip()
            return "" if cmd.startswith("CANNOT_DO") else cmd
        except Exception:
            return ""

    def _llm_fix(self, failed_cmd: str, stderr: str, original_query: str) -> str:
        client = self._get_client()
        if not client:
            return ""
        try:
            resp = client.chat.completions.create(
                model=LLM_CONFIG["model"],
                messages=[
                    {"role": "system",
                     "content": "命令执行失败了，给出一行修复命令。只输出命令本身。"},
                    {"role": "user",
                     "content": f"原始需求: {original_query}\n失败命令: {failed_cmd}\n错误: {stderr[:300]}"},
                ],
                temperature=0.0,
                max_tokens=200,
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            return ""
