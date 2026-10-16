"""Mini Thalamus 全局配置"""
import os


PROJECT_ROOT = os.path.dirname(__file__)


def _load_env_file(path: str) -> None:
    """Load simple KEY=VALUE pairs without overriding existing env vars."""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


_load_env_file(os.path.join(PROJECT_ROOT, ".env"))


def _resolve_deepseek_key() -> str:
    """优先级链：env 变量 → CredentialStore → 空字符串。每次调用都重新解析，支持运行时写入凭证。"""
    key = os.getenv("DEEPSEEK_API_KEY", "")
    if key:
        return key
    try:
        from credentials import get_store
        cred = get_store().get("deepseek")  # admin（agent_id=None）
        if cred and cred.get("api_key"):
            return cred["api_key"]
    except Exception:
        import logging
        logging.getLogger(__name__).warning(
            "CredentialStore 读取失败，将使用环境变量 key", exc_info=True)
    return ""


# ─── LLM 配置 ───
# api_key 在模块加载时只做一次快速探测，用于 warn 判断。
# 运行时调用方应使用 get_llm_config() 获取带实时 key 的副本。
LLM_CONFIG = {
    "api_key": _resolve_deepseek_key(),
    "base_url": os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
    "model": os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
    "temperature": 0.1,
    "max_tokens": 2048,
}

if not LLM_CONFIG["api_key"]:
    import warnings
    warnings.warn(
        "DEEPSEEK_API_KEY 未设置（已尝试 env 变量）。"
        "请 export DEEPSEEK_API_KEY=… 或在 UI 凭证面板添加 'deepseek'。"
        "首次 LLM 调用前若仍为空，OpenAI SDK 会返回 401。",
        RuntimeWarning,
    )


def get_llm_config() -> dict:
    """返回带实时解析 api_key 的 LLM 配置副本。

    每次调用都重新从 env/CredentialStore 解析 key，
    避免启动时 key 为空、UI 写入后仍用旧值的问题。
    """
    return {
        **LLM_CONFIG,
        "api_key": _resolve_deepseek_key(),
        "base_url": os.getenv("DEEPSEEK_BASE_URL", LLM_CONFIG["base_url"]),
        "model": os.getenv("DEEPSEEK_MODEL", LLM_CONFIG["model"]),
    }


# ─── Cascaded Router thresholds ───
ROUTER_SEMANTIC_THRESHOLD = float(
    os.environ.get("MT_ROUTER_SEMANTIC_THRESHOLD", "0.78")
)
ROUTER_SEMANTIC_MARGIN = float(
    os.environ.get("MT_ROUTER_SEMANTIC_MARGIN", "0.12")
)
ROUTER_SEMANTIC_FALLBACK_THRESHOLD = float(
    os.environ.get("MT_ROUTER_SEMANTIC_FALLBACK_THRESHOLD", "0.55")
)
ROUTER_TOP_K = int(os.environ.get("MT_ROUTER_TOP_K", "3"))

# ─── 运行时重规划(MAO)与自我评估 ───
# 步骤失败/假成功时,基于已完成中间结果重写下游 DAG(WorkflowExecutor replan 钩子)。
REPLAN_ENABLED = os.getenv("MT_REPLAN_ENABLED", "true").lower() == "true"
REPLAN_MAX = int(os.getenv("MT_REPLAN_MAX", "2"))
# 每步成功后用 SelfEvaluator 判断"技术成功但目标未完成"(关键词信号优先,LLM 只判模糊地带)。
SELF_EVAL_ENABLED = os.getenv("MT_SELF_EVAL", "true").lower() == "true"

# ─── 子 Agent 路径 ───
NL2SQL_PROJECT = os.path.join(os.path.dirname(__file__), "..", "nl2sql_agent")
RAG_PROJECT = os.path.join(os.path.dirname(__file__), "..", "RAG学习", "DeepRAG-Internal")

# ─── Router 配置 ───
# 此处的 ROUTER_AGENTS 作为**默认/兜底**字典存在。
# 真实运行时（web_server / orchestrator / router）会优先调用
# AgentRegistry.get_router_config()，并把发现到的 Skill/MCP/动态 Agent merge 进来。
# 见 REPAIR_PLAN.md #23：『ROUTER_AGENTS 改为动态生成』
_BUILTIN_ROUTER_AGENTS = {
    "nl2sql": {
        "name": "NL2SQL Agent",
        "icon": "📊",
        "description": "查询数据库。适合：查数据、统计、对比、报表、销售额/库存/用户数据类问题。",
        "keywords": ["查", "数据", "统计", "销售", "库存", "用户", "销量", "对比", "排名", "分析", "SQL"],
    },
    "rag": {
        "name": "RAG Agent",
        "icon": "📚",
        "description": "检索文档和知识库。适合：操作手册、规定、文档内容、知识问答。",
        "keywords": ["文档", "手册", "规定", "政策", "说明", "如何", "怎么做", "是什么", "介绍"],
    },
    "code": {
        "name": "Code Agent",
        "icon": "💻",
        "description": "执行 Python 代码。适合：数据分析、画图、计算、文件处理。",
        "keywords": ["画图", "计算", "分析文件", "csv", "excel", "python", "图表", "可视化"],
    },
    "docparse": {
        "name": "DocParse Agent",
        "icon": "📄",
        "description": "解析文档并基于文档内容问答。适合：分析PDF/Word/Excel/TXT、合同审阅、报告解读。",
        "keywords": ["文档", "文件", "解析", "解读", "分析文档", "合同", "报告", "PDF", "Word", "Excel"],
    },
    "websearch": {
        "name": "WebSearch Agent",
        "icon": "🌐",
        "description": "联网搜索互联网实时信息。适合：最新资讯、新闻、在线内容、需要实时数据的问题。",
        "keywords": ["搜索", "网上", "最新", "新闻", "互联网", "联网", "网上怎么说", "搜一下"],
    },
    "vision": {
        "name": "Vision Agent",
        "icon": "👁️",
        "description": "图片理解。分析截图、文档拍照、图表、照片中的文字和内容。适合：识别图片文字、分析图表、理解截图内容。",
        "keywords": ["图片", "截图", "照片", "识别", "看图", "图表", "图像"],
    },
    "data_analyst": {
        "name": "DataAnalyst Agent",
        "icon": "📈",
        "description": "上传CSV/Excel，自动数据分析：统计摘要、趋势图、分布图、异常检测。支持追问。适合：数据探索、报表分析、趋势洞察。",
        "keywords": ["分析数据", "统计", "数据", "csv", "excel", "表格", "趋势", "可视化", "报表", "数据分析"],
    },
    "deep_research": {
        "name": "DeepResearch Agent",
        "icon": "🔬",
        "description": "【优先使用】多源深度研究：自动搜索、交叉验证、生成含引用来源的研究报告。适用于：对比分析（Python vs Rust、React vs Vue）、技术选型、优缺点分析、竞品调研、市场研究。当用户问'对比''优缺点''哪个更好''vs'时，优先使用此Agent而非WebSearch。",
        "keywords": ["研究", "调研", "对比", "优缺点", "vs", "比较", "哪个更好", "区别", "差异", "选型", "竞品", "行业分析", "市场研究", "深度分析", "综述"],
    },
    "report_writer": {
        "name": "ReportWriter Agent",
        "icon": "📝",
        "description": "根据主题或分析结果生成结构化报告（Markdown）。适合：周报、研究报告、数据分析报告、总结。",
        "keywords": ["生成报告", "写报告", "周报", "报告", "总结报告", "汇总", "综述"],
    },
    "general": {
        "name": "General Agent",
        "icon": "💬",
        "description": "通用问答。无法归类到此 Agent 直接回答。",
        "keywords": [],
        "is_fallback": True,   # 标记为兜底 Agent，Router 匹配不到时 fallback
    },
}


def get_router_agents() -> dict:
    """动态生成 ROUTER_AGENTS：合并 内置 + AgentRegistry 发现的 Skill/MCP/动态 Agent

    返回值与原 ROUTER_AGENTS 字典完全同结构，便于调用方无侵入替换。
    AgentRegistry 不可用或尚未初始化时退化为内置静态字典。

    REPAIR_PLAN.md #23：『AgentRegistry 自动发现』+ 动态生成 ROUTER_AGENTS
    """
    merged = dict(_BUILTIN_ROUTER_AGENTS)
    try:
        # 优先复用 web_server 已初始化的 registry，避免重复 discover
        import sys as _sys
        _ws = _sys.modules.get("web_server") or _sys.modules.get("__main__")
        registry = None
        if _ws and hasattr(_ws, "_dynamic_cache"):
            registry = _ws._dynamic_cache.get("registry")
        if registry is None:
            from agent_registry import AgentRegistry  # 延迟 import 避免循环
            registry = AgentRegistry()
            registry.discover()
        dyn = registry.get_router_config()
        for aid, info in dyn.items():
            if aid not in merged:
                merged[aid] = info
    except Exception:
        pass
    return merged


# `ROUTER_AGENTS` 仍作为模块级名字暴露 —— 但现在它默认仅包含 _BUILTIN_ROUTER_AGENTS。
# 运行时若需要合并 Skill/MCP/动态 Agent 的最新版本，应改调
# `from config import get_router_agents` 然后 `get_router_agents()`。
# 注意：这里不在 import 时主动调用 get_router_agents()，避免触发 agent_registry.discover()
# 在 import 链中导致的循环加载和重型 Agent（EasyOCR/Vision）初始化。
ROUTER_AGENTS = dict(_BUILTIN_ROUTER_AGENTS)

# ─── Agent 循环配置 ───
MAX_STEPS = 10              # 单次请求最大 ReAct 步数
MAX_TOKENS = 8000           # 对话 token 预算
DUP_DETECT_LIMIT = 2        # 同工具同参数连续调用上限

# ─── Code Agent 沙箱配置 ───
SANDBOX_TIMEOUT = 30        # 代码执行超时(秒)
SANDBOX_MEMORY = "256m"     # 内存限制
SANDBOX_USE_DOCKER = os.getenv("MT_SANDBOX_USE_DOCKER", "false").lower() == "true"
SANDBOX_DOCKER_IMAGE = os.getenv("MT_SANDBOX_IMAGE", "python:3.11-slim")
# 本地执行不是沙箱；即使启用，也必须先通过 ExecutionPolicy 审批检查点。
ALLOW_LOCAL_EXECUTION = os.getenv("MT_ALLOW_LOCAL_EXECUTION", "false").lower() == "true"
# 限制开关：true=不限制（自动放行高风险步骤），false=限制（高风险需审批）
AUTO_APPROVE = os.getenv("MT_AUTO_APPROVE", "true").lower() == "true"

# ─── Skill & MCP 配置 ───
SKILLS_DIR = os.path.join(os.path.dirname(__file__), "skills")
MCP_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "data", ".mcp.json")
MCP_AUTO_START = True          # False to skip MCP at startup
SKILL_AUTO_DISCOVER = True
