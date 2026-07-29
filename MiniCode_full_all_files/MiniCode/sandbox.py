"""sandbox.py —— 跨平台 subprocess 工具

FEATURE_PLAN.md §1 P0.5 (audit R4): 把 Shell Agent / Code Agent 共用的
跨平台 subprocess 工具集中到这里，让上层 Agent 不再各自写 `if sys.platform == "win32"` 分支。

仅提供 helper，不做执行策略（risk 分级、cwd 白名单、deny-list）—— 那些属于
shell_agent / code_agent 各自的安全层。
"""
from __future__ import annotations
import os
import sys
import shlex
import signal
import subprocess
import shutil
from typing import Iterable

IS_WIN: bool = sys.platform == "win32"


def split_argv(cmd: str, posix: bool | None = None) -> list[str]:
    """解析一行 shell 命令为 argv 列表。

    posix 默认随平台自动：POSIX 系统 True，Windows False。
    Windows 用 posix=False 是为了避免 `cd C:\\Users` 被吃掉反斜杠转义成 `C:Users`。
    """
    if posix is None:
        posix = not IS_WIN
    return shlex.split(cmd, posix=posix)


def make_preexec_fn():
    """返回适合 Popen(preexec_fn=...) 的回调；Windows 上无 setsid。

    在 POSIX 上调用 os.setsid 让子进程独立成进程组，便于后续 killpg 杀掉
    整棵子进程树（例如 make / docker build 派生的孙子进程）。
    """
    if IS_WIN:
        return None
    return os.setsid


def kill_process_tree(proc: subprocess.Popen) -> None:
    """跨平台杀掉 Popen 进程及其子进程树。

    POSIX: os.killpg(getpgid(pid), SIGKILL) —— 依赖 preexec_fn=setsid 已建组。
    Windows: proc.kill() （底层 TerminateProcess，子进程不会被杀，但 Popen
             返回的 handle 终止；保持简单，子进程清理留 OS）。

    幂等：进程已死时不抛 ProcessLookupError。
    """
    try:
        if IS_WIN:
            proc.kill()
        else:
            try:
                pgid = os.getpgid(proc.pid)
            except (ProcessLookupError, OSError):
                # 进程已退出，无需杀
                return
            os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, OSError):
        # 已被回收 / 权限不足 —— 都属于"已经杀不到了"，吞掉
        pass


def scrub_env(env: dict | None = None,
              extra_drop: Iterable[str] = ()) -> dict:
    """复制并清理 env，去掉容易泄漏 / 注入的变量。

    默认清掉：
      - LD_PRELOAD / LD_LIBRARY_PATH / DYLD_* —— 共享库注入
      - PYTHONPATH —— 防 nl2sql_agent 把自己的 path 污染传给子进程
      - 各家 LLM API Key —— 不让 curl/git 等子命令拿到主进程凭证

    Shell Agent / Code Agent 可通过 extra_drop 追加自家黑名单（如 Code 不需要
    用户的 PATH，可以再 drop 之类）。
    """
    base = dict(env if env is not None else os.environ)
    drop = {
        "LD_PRELOAD", "LD_LIBRARY_PATH",
        "DYLD_INSERT_LIBRARIES", "DYLD_LIBRARY_PATH",
        "PYTHONPATH",
        # LLM 凭证防泄漏（凭证应由 Credential Store 显式注入，不靠继承）
        "DEEPSEEK_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
        "BING_API_KEY", "GITHUB_TOKEN",
        # 各类可导致子进程代码注入 / 路径重定向的环境变量
        "PYTHONSTARTUP", "PYTHONHOME",
        "BASH_ENV", "ENV", "PROMPT_COMMAND", "IFS",
        "PERL5LIB", "PERLLIB",
        "RUBYLIB", "RUBYOPT",
        "GEM_PATH", "GEM_HOME",
        "LD_AUDIT", "LD_ORIGIN_PATH",
        "TMPDIR",  # 某些程序用 TMPDIR 定位 lib
    }
    drop.update(extra_drop)
    for k in drop:
        base.pop(k, None)
    base.setdefault("PYTHONIOENCODING", "utf-8")
    return base


class ExecutionPolicy:
    """Separate execution isolation from approval decisions."""

    def __init__(self, workspace_root: str,
                 local_execution_enabled: bool = False,
                 docker_backend_enabled: bool = False,
                 docker_cli_available: bool | None = None):
        self.workspace_root = os.path.realpath(workspace_root)
        self.local_execution_enabled = bool(local_execution_enabled)
        self.docker_backend_enabled = bool(docker_backend_enabled)
        self.docker_cli_available = (
            shutil.which("docker") is not None
            if docker_cli_available is None
            else bool(docker_cli_available)
        )

    def path_allowed(self, path: str) -> bool:
        try:
            target = os.path.realpath(path)
            return os.path.commonpath([self.workspace_root, target]) == self.workspace_root
        except (OSError, ValueError, TypeError):
            return False

    def evaluate_call(self, route_call, agent_spec: dict) -> dict:
        capabilities = agent_spec.get("capabilities", {}) or {}
        risk = capabilities.get("side_effects", "external")
        agent_id = route_call.function.name
        uses_execution_backend = agent_id in {"code", "shell"}
        docker_ready = (
            agent_id == "code"
            and self.docker_backend_enabled
            and self.docker_cli_available
        )
        sandboxed = docker_ready

        if (
            uses_execution_backend
            and not docker_ready
            and not self.local_execution_enabled
        ):
            return {
                "action": "deny",
                "risk_level": risk,
                "requires_approval": False,
                "sandboxed": False,
                "reason": "本地执行未启用，且适用的 Docker 隔离后端不可用",
            }

        if risk in {"none", "database_read", "read_only"}:
            action = "allow"
            requires_approval = False
            reason = "只读或无外部副作用"
        else:
            action = "checkpoint"
            requires_approval = True
            reason = f"Agent 声明了 {risk} 副作用"
        return {
            "action": action,
            "risk_level": risk,
            "requires_approval": requires_approval,
            "sandboxed": sandboxed,
            "reason": reason,
        }


__all__ = [
    "IS_WIN",
    "split_argv",
    "make_preexec_fn",
    "kill_process_tree",
    "scrub_env",
    "ExecutionPolicy",
]
