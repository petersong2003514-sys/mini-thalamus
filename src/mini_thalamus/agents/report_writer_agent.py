"""ReportWriter Agent —— 生成结构化 Markdown 报告

使用:
  输入主题或分析结果 → 生成目录/章节/数据表格/引用的完整报告
  可接收上一 Agent 的输出（DataAnalyst/DeepResearch）串联写作
"""
import sys, os, json, re

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from mini_thalamus import config as _cfg
LLM_CONFIG = _cfg.LLM_CONFIG

from openai import OpenAI
from mini_thalamus.token_budget import max_tokens_for
from mini_thalamus.structured_output import build_prompt_instruction, parse_llm_output

try:
    from mini_thalamus.logging_config import get_logger
    _log = get_logger("report_writer")
except Exception:
    import logging as _logging
    _log = _logging.getLogger("report_writer")

AGENT_META = {
    "name": "ReportWriter Agent",
    "id": "report_writer",
    "description": "生成结构化 Markdown 报告：周报、研究报告、数据分析报告",
        "keywords": ["报告", "周报", "总结", "生成报告", "写报告", "正式报告", "研究报告", "数据分析报告", "整理成报告", "汇报", "汇总", "综述"],
    "intents": ["report_generation", "summary"],
    "examples": ["把已有材料整理成正式报告", "写一份周报", "生成数据分析报告", "把分析结果写成报告"],
    "icon": "📝",
    "capabilities": {
        "needs_history": True, "needs_files": False,
        "needs_long_term": True, "max_context_chars": 8000,
        "auto_chainable": True, "side_effects": "none", "needs_credentials": [],
        "actions": ["text_generation"],
        "can_handle_errors": [],
    },
    "input_schema": {"required": ["instruction"], "properties": {}},
    "output_schema": {"exports": ["answer", "summary", "data.structured"]},
}


class ReportWriterAgent:
    """Markdown 报告生成 Agent"""

    capabilities = {
        "needs_history": True,
        "needs_files": False,
        "needs_long_term": True,
        "max_context_chars": 8000,
        "auto_chainable": True,
        "side_effects": "none",
        "needs_credentials": [],
    }

    def __init__(self):
        self.client = OpenAI(api_key=LLM_CONFIG["api_key"], base_url=LLM_CONFIG["base_url"])

    def run(self, query: str, context: dict = None) -> dict:
        ctx = context or {}
        # 只用当前请求的 query，不使用对话历史（避免掺杂前轮对话的无关内容）
        # 如果需要参考上一步 Agent 的结果，由多步编排通过 query 中的"上一步查询结果"传入

        # ── 生成大纲 ──
        outline = self._generate_outline(query)
        if not outline:
            return {"status": "error", "answer": "报告大纲生成失败", "data": {}}

        # ── 逐段写作 ──
        sections = self._write_sections(query, outline)

        # ── 组装报告 ──
        report = self._assemble(outline, sections)
        summary = sections[0][:200] if sections else report[:200]

        return {
            "status": "success",
            "answer": report,
            "data": {
                "structured": {
                    "format": "text",
                    "data": {"sections": outline},
                    "confidence": "high",
                },
                "report_markdown": report,
                "section_count": len(sections),
            },
            "summary": summary,
        }

    def _generate_outline(self, topic: str) -> list:
        """LLM 生成报告大纲"""
        prompt = f"""为以下主题生成一份报告大纲（章节标题列表）。

主题: {topic}

要求:
- 包含引言和总结
- 3-6 个主体章节
- 每个标题简洁明了（10 字以内）
- 只围绕主题展开，不要引入无关内容

返回 JSON: {{"title": "报告标题", "sections": ["章节1", "章节2", ...]}}"""

        try:
            resp = self.client.chat.completions.create(
                model=LLM_CONFIG["model"],
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2, max_tokens=400,
            )
            raw = resp.choices[0].message.content.strip()
            m = re.search(r'\{.*\}', raw, re.DOTALL)
            data = json.loads(m.group(0)) if m else {"title": topic, "sections": []}
            return data
        except Exception as e:
            _log.warning("大纲生成失败: %s", e)
            return {"title": topic, "sections": ["概述", "详细分析", "结论"]}

    def _write_sections(self, topic: str, outline: dict) -> list:
        """逐章节写作"""
        sections = []
        for i, sec in enumerate(outline.get("sections", [])[:8]):
            prompt = f"""写一份报告的「{sec}」章节。

报告主题: {topic}

要求:
- 写 1-3 段充实、专业、简洁的内容
- 只围绕「{topic}」展开，严禁引入与主题无关的内容
- 如果有数据，用 Markdown 表格展示
- 只输出该章节内容，不要写其他章节

{build_prompt_instruction()}"""

            try:
                resp = self.client.chat.completions.create(
                    model=LLM_CONFIG["model"],
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3, max_tokens=max_tokens_for(topic, "answer"),
                )
                structured = parse_llm_output(resp.choices[0].message.content.strip())
                sections.append(structured.get("answer", f"## {sec}\n\n内容生成中..."))
            except Exception as e:
                sections.append(f"## {sec}\n\n*生成失败: {e}*")

        return sections

    def _assemble(self, outline: dict, sections: list) -> str:
        """组装最终报告"""
        title = outline.get("title", "研究报告")
        lines = [
            f"# {title}",
            "",
            f"> 生成时间: {__import__('time').strftime('%Y-%m-%d %H:%M')}",
            "",
            "---",
            "",
        ]

        for i, (sec_title, content) in enumerate(
            zip(outline.get("sections", []), sections), 1
        ):
            lines.append(f"## {i}. {sec_title}")
            lines.append("")
            lines.append(content)
            lines.append("")

        lines.extend([
            "---",
            "",
            "*本报告由 AI 自动生成，仅供参考。*",
        ])
        return "\n".join(lines)
