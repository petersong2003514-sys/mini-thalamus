"""Skill 固化 —— 从对话历史中提取工作流，生成可复用 Skill 文件"""
import os, re, json, time
import importlib.util

_MINICODE_ROOT = os.path.dirname(os.path.abspath(__file__))
_cfg_path = os.path.join(_MINICODE_ROOT, "config.py")
_spec = importlib.util.spec_from_file_location("crystallize_config", _cfg_path)
_cfg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cfg)
LLM_CONFIG = _cfg.LLM_CONFIG

from openai import OpenAI

try:
    from logging_config import get_logger
    _log = get_logger("crystallize")
except Exception:
    import logging as _logging
    _log = _logging.getLogger("crystallize")

SKILLS_DIR = os.path.join(_MINICODE_ROOT, "skills")
os.makedirs(SKILLS_DIR, exist_ok=True)

# 代码安全扫描模式（防止 LLM 生成恶意代码）
_CODE_DANGER = [
    r'os\.system\(', r'subprocess\.', r'eval\(', r'exec\(', r'__import__\(',
    r'open\(.*[\'\"]w', r'shutil\.rmtree', r'socket\.', r'ctypes',
    r'import\s+requests', r'urllib', r'urlopen', r'os\.remove\(', r'os\.unlink\(',
    r'os\.rmdir\(', r'os\.popen', r'pickle\.loads', r'yaml\.load',
]


def _scan_code(code: str) -> list:
    """扫描生成的 Skill 代码，返回危险模式列表"""
    warnings = []
    for pattern in _CODE_DANGER:
        if re.search(pattern, code, re.IGNORECASE):
            warnings.append(pattern)
    return warnings


def crystallize(session_id: str, skill_name: str = None) -> dict:
    """从 session 对话历史生成 Skill 文件

    Returns: {"ok": bool, "skill_id": str, "skill_file": str, "meta": dict, "summary": str}
    """
    from memory_store import get_store
    store = get_store()

    # 读取对话历史（优先 JSON，缺失时回退到 SQLite turns）
    history = store._load_history(session_id)
    if not history:
        records = store.list_turns(session_id)
        history = [
            {
                "id": turn.turn_id,
                "q": turn.query,
                "a": turn.answer or "",
                "agent": turn.agent_id,
                "tags": list(turn.tags),
                "ts": turn.created_at,
            }
            for turn in records
        ]
    if not history:
        return {"ok": False, "error": "该会话没有对话记录"}

    # Shell agent 的 turn 拒绝 crystallize（外部命令不可复用为 Skill）
    shell_turns = [t for t in history if t.get("agent") in {"shell"}]
    if shell_turns:
        return {"ok": False, "error": "包含 Shell 命令的会话不支持固化为 Skill"}

    # 构建对话摘要给 LLM 分析
    dialogue = []
    for t in history[-20:]:  # 最近 20 轮
        dialogue.append({
            "q": t["q"],
            "a": t["a"][:500],
            "agent": t.get("agent", "?"),
            "tags": t.get("tags", []),
        })

    # LLM 分析对话 → 提取工作流
    client = OpenAI(api_key=LLM_CONFIG["api_key"], base_url=LLM_CONFIG["base_url"])
    analysis = _analyze_conversation(client, dialogue, skill_name)

    if not analysis.get("ok"):
        return analysis

    # 生成 Skill 文件
    skill_id = _generate_skill_file(analysis)
    if not skill_id:
        return {"ok": False, "error": "生成 Skill 文件失败"}

    return {
        "ok": True,
        "skill_id": skill_id,
        "skill_file": os.path.join(SKILLS_DIR, f"{skill_id.replace('skill:', '')}.py"),
        "meta": analysis["meta"],
        "summary": analysis["summary"],
    }


def _analyze_conversation(client: OpenAI, dialogue: list, skill_name: str = None) -> dict:
    """LLM 分析对话记录，直接生成完整的可执行 Python Skill 代码"""
    dl_text = []
    for i, d in enumerate(dialogue, 1):
        dl_text.append(f"[轮{i}] [{d['agent']}] Q: {d['q']}\nA: {d['a'][:300]}")
    dl_str = "\n\n".join(dl_text)

    name_hint = f'建议名称: "{skill_name}"' if skill_name else "根据对话内容自动命名"

    prompt = f"""你是一个 Python 代码生成专家。根据以下多轮对话记录，生成一个完整可执行的 Skill 文件。

对话记录:
{dl_str}

{name_hint}

可用的 Agent 及其接口:
- nl2sql: 查询数据库，输入自然语言 → 返回 {{"answer": "...", "data": {{"rows": [...], "sql": "..."}}}}
- code: 执行 Python 代码/画图，输入代码或自然语言 → 返回 {{"answer": "...", "data": {{"images": [...], "stdout": "..."}}}}
- websearch: 联网搜索，输入关键词 → 返回 {{"answer": "...", "data": {{"structured": {{...}}, "sources": [...]}}}}
- docparse: 文档分析问答，输入问题+context.files → 返回 {{"answer": "...", "data": {{"structured": {{...}}}}}}
- rag: 知识库检索，输入问题 → 返回 {{"answer": "...", "data": {{"knowledge_base_files": [...]}}}}

_caller(agent_name, prompt, context=None) 辅助函数已可用，它会:
1. 调用指定 Agent
2. 返回 Agent 的完整 result dict

请输出完整的 JSON，包含 meta 信息和可直接运行的 Python 代码:

{{
  "is_crystallizable": true/false,
  "reason": "不可固化时说明原因",
  "meta": {{
    "name": "简短中文名",
    "description": "一句话描述",
    "keywords": ["关键词"],
    "icon": "emoji"
  }},
  "code": "完整的 Python run() 函数代码。要求:\\n1. 从 query 中提取参数（用正则或简单字符串匹配）\\n2. 按对话顺序调用 Agent，中间结果传给下一步\\n3. 每一步检查返回状态，失败时给出提示\\n4. 最终返回标准 Agent result dict（含 structured 输出）\\n5. 代码要能直接运行，不依赖外部变量\\n6. 用 _caller(agent_name, prompt, context) 调用其他 Agent",
  "summary": "一句话说明这个 Skill 做什么"
}}

生成规则:
- 提取对话中的参数模式（如时间范围、指标名、数量限制等），在代码中用字符串匹配从 query 提取
- 按对话轮次顺序组织 Agent 调用链
- 如果某个 Agent 的输入依赖前一个 Agent 的输出，用前一步的 result 构造 prompt
- 代码必须包含完整的 run(query, context=None) 函数定义
- 使用 _caller(agent, prompt, context) 来调用 Agent，不要自己实现 HTTP 请求
- 所有字符串使用双引号，避免转义问题

只输出 JSON，code 字段中的代码用 \\n 表示换行。"""

    try:
        resp = client.chat.completions.create(
            model=LLM_CONFIG["model"],
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1, max_tokens=2000,
        )
        raw = resp.choices[0].message.content.strip()
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        analysis = json.loads(m.group(0)) if m else json.loads(raw)
        analysis["ok"] = analysis.get("is_crystallizable", True)
        return analysis
    except Exception as e:
        return {"ok": False, "error": f"LLM 分析失败: {e}"}


def _indent_block(code: str) -> str:
    lines = code.strip().splitlines()
    if not lines:
        return "    pass\n"
    return "\n".join("    " + line for line in lines) + "\n"


def _fallback_return(name: str) -> str:
    return (
        "    return {\"status\": \"success\", "
        "\"answer\": f\"Skill '{name}' 执行完成\", "
        "\"data\": {}, \"summary\": \"\"}\n"
    )


def _generate_skill_file(analysis: dict) -> str:
    """根据 LLM 生成的完整代码创建 Skill 文件"""
    meta = analysis.get("meta", {})
    code = analysis.get("code", "")
    name = meta.get("name", "未命名")
    summary = analysis.get("summary", "")
    safe_name = re.sub(r'[^\w一-鿿]', '_', name)[:20]
    skill_id = f"skill:{safe_name}"
    file_path = os.path.join(SKILLS_DIR, f"{safe_name}.py")

    # 如果 LLM 返回了完整代码，用它；否则降级为骨架
    if code and len(code) > 50:
        run_code = code
        if re.search(r"(?m)^def run\(", run_code):
            run_block = run_code.rstrip() + "\n"
        else:
            run_block = (
                "def run(query: str, context: dict = None) -> dict:\n"
                + _indent_block(run_code)
                + _fallback_return(name)
            )
    else:
        agent_chain = analysis.get("pattern", {}).get("agent_chain", [])
        run_code = (
            "    # 工作流: " + " → ".join(agent_chain) + "\n"
            "    # TODO: 根据对话模式完善具体调用逻辑\n"
            "    pass\n"
        )
        run_block = (
            "def run(query: str, context: dict = None) -> dict:\n"
            + run_code
            + _fallback_return(name)
        )


    content = f'''"""固化 Skill: {name}

自动生成于: {time.strftime("%Y-%m-%d %H:%M")}
{summary}
"""
import json, re

META = {{
    "name": "{name}",
    "description": "{meta.get("description", "自动生成的固化 Skill")}",
    "keywords": {json.dumps(meta.get("keywords", []), ensure_ascii=False)},
    "icon": "{meta.get("icon", "⚡")}",
}}


{run_block}


def _caller(agent_name: str, prompt: str, ctx: dict = None) -> dict:
    """调用其他 Agent"""
    import sys, os as _os
    _root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    sys.path.insert(0, _root)
    from web_server import get_agents
    agents = get_agents()
    agent = agents.get(agent_name)
    if not agent:
        return {{"status": "error", "answer": f"Agent {{agent_name}} 不可用"}}
    return agent.run(prompt, ctx)
'''

    # 安全扫描：拒绝含危险模式的代码
    danger = _scan_code(run_code)
    if danger:
        _log.warning("Skill 代码被拦截: 含危险模式 %s", danger)
        return ""  # 返回空字符串使调用方 crystallize() 走 ok=False 路径

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

    # 触发重新发现
    try:
        from skill_loader import SkillLoader
        sl = SkillLoader(SKILLS_DIR)
        sl.discover()
    except Exception:
        pass

    _log.info("已生成 Skill: %s", file_path)
    return skill_id
