"""
Agent 工厂 —— 被 main.py 和 app.py 共享导入，不含命令行模块级副作用。
"""

import sys
import os

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


def _dotenv_candidates():
    """Where to look for a .env when the caller did not name one.

    Checked in order, and the first file that exists is the one that is read.
    The repository root comes first so this sub-package shares the single .env
    the install instructions tell you to create; the package directory is the
    fallback for anyone using the package on its own.

    The repository root is only considered when a pyproject.toml sits beside
    it. Under a real install PROJECT_ROOT is inside site-packages, and two
    levels up is some unrelated directory whose .env must not be picked up.
    """
    root = os.path.abspath(os.path.join(PROJECT_ROOT, "..", ".."))
    candidates = [os.path.join(PROJECT_ROOT, ".env")]
    if os.path.isfile(os.path.join(root, "pyproject.toml")):
        candidates.insert(0, os.path.join(root, ".env"))
    candidates.append(os.path.join(os.getcwd(), ".env"))
    return candidates


def load_dotenv(env_file=None):
    """Load a dotenv into the isolated ``NL2SQL_*`` namespace."""
    if env_file is None:
        env_file = next(
            (path for path in _dotenv_candidates() if os.path.isfile(path)),
            os.path.join(PROJECT_ROOT, ".env"),
        )
    if not os.path.isfile(env_file):
        return
    aliases = {
        "OPENAI_API_KEY": "NL2SQL_API_KEY",
        "LLM_MODEL": "NL2SQL_MODEL",
        "LLM_BASE_URL": "NL2SQL_BASE_URL",
        "LIGHT_MODEL": "NL2SQL_LIGHT_MODEL",
        "LIGHT_BASE_URL": "NL2SQL_LIGHT_BASE_URL",
        "LIGHT_API_KEY": "NL2SQL_LIGHT_API_KEY",
    }
    with open(env_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            scoped_key = aliases.get(key, key if key.startswith("NL2SQL_") else "")
            if scoped_key and scoped_key not in os.environ:
                os.environ[scoped_key] = val


def create_llm(api_key=None, model=None, base_url=None):
    """创建 LLM 实例（复用 core/llm.py 的工厂函数）"""
    from nl2sql_agent.core.llm import create_llm as _create

    api_key = api_key or os.environ.get("NL2SQL_API_KEY", "")
    model = model or os.environ.get("NL2SQL_MODEL", "deepseek-chat")
    base_url = base_url or os.environ.get("NL2SQL_BASE_URL", "")

    if not api_key:
        raise RuntimeError(
            "未找到 NL2SQL API Key；请设置 NL2SQL_API_KEY 或配置包内 .env"
        )

    return _create(
        "openai",
        api_key=api_key,
        model=model,
        base_url=base_url or None,
    )


def build_agent(schema_strategy="full", max_rounds=MAX_ROUNDS, db_path=None):
    """
    构建完整的 NL2SQL Agent，按依赖顺序组装所有组件。

    依赖链:
      db_connector → schema_info → schema_provider
      → llm → light_llm
      → difficulty_classifier → fewshot_selector
      → sql_validator → error_enhancer → stop_judge
      → NL2SQLAgent
    """
    from nl2sql_agent.interfaces.database_connector import SQLiteConnector
    from nl2sql_agent.interfaces.schema_provider import FullSchemaProvider, KeywordSchemaProvider, HybridSchemaProvider
    from nl2sql_agent.interfaces.difficulty_classifier import LLMDifficultyClassifier
    from nl2sql_agent.interfaces.fewshot_selector import TagBasedFewShotSelector
    from nl2sql_agent.interfaces.sql_validator import LayeredSQLValidator
    from nl2sql_agent.interfaces.error_enhancer import HybridErrorEnhancer
    from nl2sql_agent.interfaces.stop_judge import HybridStopJudge
    from nl2sql_agent.interfaces.task_decomposer import LLMTaskDecomposer
    from nl2sql_agent.agent.nl2sql_agent import NL2SQLAgent
    from nl2sql_agent.agent.fewshot_examples import FEWSHOT_POOL
    from nl2sql_agent.tools.schema_tools import register_schema_tools
    from nl2sql_agent.tools.sql_tools import register_sql_tools

    _resolved_db_path = db_path if db_path else DB_PATH

    print("[1/8] 连接数据库...")
    db_connector = SQLiteConnector(_resolved_db_path, readonly=DB_READONLY)
    if not db_connector.test_connection():
        raise RuntimeError(f"无法连接 NL2SQL 数据库: {_resolved_db_path}")
    print(f"      数据库: {_resolved_db_path} ({db_connector.db_type})")

    print("[2/8] 读取 Schema...")
    schema_info = db_connector.get_schema_info()
    print(f"      共 {len(schema_info.tables)} 张表, {sum(len(t.columns) for t in schema_info.tables)} 个列")

    print("[3/8] 连接 LLM...")
    primary_api_key = os.environ.get("NL2SQL_API_KEY", PRIMARY_API_KEY)
    primary_model = os.environ.get("NL2SQL_MODEL", PRIMARY_MODEL)
    primary_base_url = os.environ.get("NL2SQL_BASE_URL", PRIMARY_BASE_URL)
    llm = create_llm(primary_api_key, primary_model, primary_base_url)
    print(f"      模型: {primary_model}  (所有模块共用同一API)")

    print(f"[4/8] 初始化 Schema Provider (策略: {schema_strategy})...")
    if schema_strategy == "full":
        schema_provider = FullSchemaProvider(schema_info)
    elif schema_strategy == "keyword":
        schema_provider = KeywordSchemaProvider(schema_info)
    elif schema_strategy == "hybrid":
        schema_provider = HybridSchemaProvider(schema_info, llm=llm)
    else:
        print(f"      [WARN] 策略 '{schema_strategy}' 尚未实现，回退到 'full'")
        schema_provider = FullSchemaProvider(schema_info)

    print("[5/8] 注册工具...")
    register_schema_tools(schema_provider)
    register_sql_tools(db_connector)
    from nl2sql_agent.core.tools import TOOL_REGISTRY
    print(f"      已注册 {len(TOOL_REGISTRY)} 个工具: {', '.join(TOOL_REGISTRY.keys())}")

    print("[6/8] 初始化分级器 & 选择器...")
    difficulty_classifier = LLMDifficultyClassifier(llm)
    fewshot_selector = TagBasedFewShotSelector(FEWSHOT_POOL)
    print(f"      Few-shot 示例池: {len(FEWSHOT_POOL)} 条")

    print("[7/8] 初始化校验器 & 增强器 & 停止判断器...")
    sql_validator = LayeredSQLValidator(llm)
    error_enhancer = HybridErrorEnhancer()
    stop_judge = HybridStopJudge(llm)
    task_decomposer = LLMTaskDecomposer(llm)

    print("[8/8] 组装 Agent...")
    agent = NL2SQLAgent(
        schema_provider=schema_provider,
        db_connector=db_connector,
        llm=llm,
        difficulty_classifier=difficulty_classifier,
        fewshot_selector=fewshot_selector,
        sql_validator=sql_validator,
        error_enhancer=error_enhancer,
        stop_judge=stop_judge,
        task_decomposer=task_decomposer,
        light_llm=llm,
        max_rounds=max_rounds,
    )

    return agent, db_connector, schema_provider
