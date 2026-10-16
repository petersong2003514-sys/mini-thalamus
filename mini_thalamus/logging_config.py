"""统一日志配置 —— 所有 print() 的替代方案

用法:
    from logging_config import get_logger
    logger = get_logger(__name__)
    logger.info("正常消息")
    logger.warning("警告")
    logger.exception("异常（自动附带 traceback）")

配置:
    INFO 及以上 → logs/server.log（每天 rotate，保留 7 天）
    WARNING/ERROR → 同时输出到 stderr
    格式: [时间] [级别] [模块] 消息
"""

import logging
import logging.handlers
import os
import re
import sys

_log_initialized = False
_log_dir = None


# FEATURE_PLAN.md §5 Credential Store —— 日志脱敏过滤器
# 不依赖凭证模块；用静态正则覆盖最常见的明文 secret 形态。
# 任何 record 的 args / msg 都会被过滤，避免开发者无意 _log.info("token=%s", token) 漏出去。
_SECRET_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # DeepSeek / OpenAI / Anthropic 等 sk-* 系列
    (re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}"), "sk-***"),
    # GitHub PAT / Fine-grained token
    (re.compile(r"\bghp_[A-Za-z0-9]{20,}"), "ghp_***"),
    (re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}"), "github_pat_***"),
    # AWS access key
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "AKIA***"),
    # Bearer / Authorization header values
    (re.compile(r"(?i)\b(Bearer|Token)\s+[A-Za-z0-9._\-]{12,}"), r"\1 ***"),
    # 显式 key=val 形态（api_key / password / secret / token = ...）
    (re.compile(r"(?i)\b(api[_-]?key|password|secret|access[_-]?token)\s*[:=]\s*[\"']?([^\s\"',;]{6,})"),
     lambda m: f"{m.group(1)}={'*' * 8}"),
    # PEM 头（不可能在单行；但作为兜底）
    (re.compile(r"-----BEGIN [A-Z ]+PRIVATE KEY-----"), "-----BEGIN <REDACTED>-----"),
]


def _scrub(text: str) -> str:
    """对一段文本做 secret 脱敏。"""
    if not isinstance(text, str) or not text:
        return text
    for pat, repl in _SECRET_PATTERNS:
        try:
            text = pat.sub(repl, text)
        except Exception:
            continue
    return text


class SecretRedactFilter(logging.Filter):
    """脱敏 filter：拦在 root logger 的所有 handler 之前。

    既扫格式化后的最终消息（getMessage()），也扫 args 里的字符串元素，
    避免 lazy format `_log.info("%s", token)` 漏过。
    """
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            if isinstance(record.msg, str):
                record.msg = _scrub(record.msg)
            if record.args:
                if isinstance(record.args, tuple):
                    record.args = tuple(
                        _scrub(a) if isinstance(a, str) else a for a in record.args
                    )
                elif isinstance(record.args, dict):
                    record.args = {k: (_scrub(v) if isinstance(v, str) else v)
                                   for k, v in record.args.items()}
        except Exception:
            pass
        return True


def setup(log_dir: str = None):
    """初始化日志系统（web_server.py 启动时调用一次）"""
    global _log_initialized, _log_dir

    if _log_initialized:
        return

    if log_dir is None:
        # 默认放在 Mini Thalamus 根目录下的 logs/
        log_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "logs"
        )
    _log_dir = log_dir
    os.makedirs(log_dir, exist_ok=True)

    root_logger = logging.getLogger("mini_thalamus")
    root_logger.setLevel(logging.DEBUG)  # 捕获所有级别，由 handler 过滤

    # FEATURE_PLAN.md §5：脱敏 filter 加在 root 上，所有 handler 都受益
    root_logger.addFilter(SecretRedactFilter())

    # 格式
    fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)-7s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ── 文件 handler：INFO 及以上，每天轮转 ──
    # NOTE: TimedRotatingFileHandler 在 Windows 上存在文件锁定问题——
    # 日志文件被 handler 持有期间，外部程序（如记事本、备份脚本）无法读取/移动
    # 该文件。若需在 Windows 上查看当前日志，建议先复制文件再打开副本。
    log_path = os.path.join(log_dir, "server.log")
    file_handler = logging.handlers.TimedRotatingFileHandler(
        log_path, when="midnight", interval=1, backupCount=7,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(fmt)
    root_logger.addHandler(file_handler)

    # ── 控制台 handler：WARNING 及以上 → stderr ──
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(fmt)
    root_logger.addHandler(console_handler)

    _log_initialized = True

    # 记录启动事件
    root_logger.info("══════ 日志系统初始化 ══════")
    root_logger.info(f"日志目录: {log_dir}")


def get_logger(name: str = "mini_thalamus") -> logging.Logger:
    """获取 logger 实例。name 建议传 __name__ 以显示模块路径。"""
    # 剥离前缀，只保留简短模块名
    if name.startswith("mini_thalamus."):
        pass
    elif "." in name:
        # 例如 "agents.vision_agent" → "mini_thalamus.agents.vision_agent"
        name = "mini_thalamus." + name
    else:
        name = "mini_thalamus." + name

    return logging.getLogger(name)


def get_log_dir() -> str:
    """返回日志目录路径"""
    global _log_dir
    if _log_dir is None:
        _log_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "logs"
        )
    return _log_dir
