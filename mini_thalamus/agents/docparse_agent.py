"""DocParse Agent —— 文档解析 + 结构理解 + 基于文档问答"""
import sys, os, re, json, tempfile, shutil

# Config 加载（和 CodeAgent 一样的方式）
_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import importlib.util
_cfg_path = os.path.join(_PKG_ROOT, "config.py")
_spec = importlib.util.spec_from_file_location("mt_config", _cfg_path)
_cfg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cfg)
LLM_CONFIG = _cfg.LLM_CONFIG

from openai import OpenAI
from token_budget import max_tokens_for

MAX_DOC_CHARS = 15000  # 单个文档最多送入 LLM 的字符


# ─── 接入统一记忆存储 ───
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from memory_store import get_store

# 兼容旧接口（web_server 等外部调用方）
def get_session_files(sid: str) -> list:
    return get_store().get_files(sid)

AGENT_META = {
    "name": "DocParse Agent",
    "id": "docparse",
    "description": "解析文档并基于文档内容问答",
    "keywords": ["文档", "文件", "PDF", "Word"],
    "icon": "📄",
    "capabilities": {
        "needs_history": True, "needs_files": True,
        "needs_long_term": False, "max_context_chars": 8000,
        "auto_chainable": True, "side_effects": "local", "needs_credentials": [],
    },
    "input_schema": {"required": ["instruction"], "properties": {}},
    "output_schema": {"exports": ["answer", "summary", "data.structured"]},
}


class DocParseAgent:
    """文档分析 Agent —— 通过 MemoryStore 管理会话文件"""

    capabilities = {
        "needs_history": True,
        "needs_files": True,
        "needs_long_term": False,
        "max_context_chars": 8000,
        "auto_chainable": True,
        "side_effects": "local",
        "needs_credentials": [],
    }

    def __init__(self):
        self.client = OpenAI(api_key=LLM_CONFIG["api_key"], base_url=LLM_CONFIG["base_url"])
        self.store = get_store()

    def run(self, query: str, context: dict = None) -> dict:
        ctx = context or {}
        files = ctx.get("files", [])
        sid = ctx.get("session_id", "default")

        # ── 新上传文件 → 解析 + 注册到 MemoryStore ──
        uploaded = [f for f in files if isinstance(f, dict)]
        if uploaded:
            new_docs, errors = [], []
            for f in uploaded:
                parsed = _parse_file(f)
                if parsed.get("error"):
                    errors.append(parsed["error"])
                    continue
                new_docs.append(parsed)
                # 注册到 MemoryStore（自动保存 JSON + 更新索引）
                self.store.register_file(sid, {
                    "name": f["name"],
                    "pages": parsed.get("pages", "?"),
                    "paragraph_count": parsed.get("paragraph_count", "?"),
                    "format": parsed.get("format", "?"),
                    "summary": _quick_summary(parsed),
                    "parsed": parsed,
                })
            print(f"  [DocParse] 会话{sid[:8]}: 注册 {len(new_docs)} 个新文件")

            # 加载所有文件（新旧合并）
            all_parsed = self.store.load_all_parsed_files(sid)
            seen_names = {d.get("name") for d in new_docs}
            docs = new_docs + [d for d in all_parsed
                              if d.get("name") not in seen_names]
        else:
            docs = self.store.load_all_parsed_files(sid)
            errors = []
            if docs:
                print(f"  [DocParse] 会话{sid[:8]}: 从 MemoryStore 加载 {len(docs)} 个文档")

        if not docs:
            return {"status": "error",
                    "answer": f"文件解析失败: {'; '.join(errors[:3])}" if errors else "未获取到文档内容",
                    "data": {"errors": errors}}

        # Step 2: 多文档对比（2+ 文档 + 对比关键词）
        cmp_keywords = ["对比", "比较", "区别", "差异", "不同", "异同", "哪个更", "优缺点"]
        if len(docs) >= 2 and query.strip() and any(kw in query for kw in cmp_keywords):
            result = _compare_docs(query, docs, self.client)
            if isinstance(result, str):
                structured = {"answer": result, "format": "comparison", "data": {}, "sources": [], "confidence": "medium"}
            else:
                structured = result
        elif query.strip():
            result = _answer_question(query, docs, self.client)
            if isinstance(result, str):
                # LLM 没有输出 JSON → 包装
                structured = {"answer": result, "format": "text", "data": {}, "sources": [], "confidence": "medium"}
            else:
                structured = result
        else:
            summary_text = _generate_summary(docs, self.client)
            structured = {"answer": summary_text, "format": "text", "data": {}, "sources": [], "confidence": "medium"}

        return {
            "status": "success",
            "answer": structured.get("answer", ""),
            "data": {
                "structured": structured,
                "files": [{"name": d["name"], "pages": d.get("pages", "?"),
                           "paragraphs": d.get("paragraph_count", "?")} for d in docs],
                "errors": errors,
            },
            "summary": structured.get("answer", "")[:300],
        }


# ─── 格式检测 ───
def _detect_format(filename: str) -> str:
    ext = os.path.splitext(filename)[1].lower()
    mapping = {".pdf": "pdf", ".docx": "docx", ".xlsx": "xlsx",
               ".xls": "xlsx", ".txt": "txt", ".md": "txt", ".csv": "txt"}
    return mapping.get(ext, "unknown")


# ─── 文件解析入口 ───
def _parse_file(f: dict) -> dict:
    name = f.get("name", "unknown")
    content = f.get("content")  # bytes
    fmt = _detect_format(name)

    if fmt == "unknown":
        ext = os.path.splitext(name)[1].lower()
        return {"error": f"{name}: 暂不支持 {ext} 格式（当前支持 PDF/Word/Excel/TXT）", "name": name}

    # 保存到临时文件
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(name)[1])
    try:
        if isinstance(content, bytes):
            tmp.write(content)
        elif isinstance(content, str):
            tmp.write(content.encode("utf-8"))
        tmp.close()

        if fmt == "pdf":
            result = _parse_pdf(tmp.name, name)
        elif fmt == "docx":
            result = _parse_docx(tmp.name, name)
        elif fmt == "xlsx":
            result = _parse_xlsx(tmp.name, name)
        elif fmt == "txt":
            result = _parse_txt(tmp.name, name)
        else:
            result = {"error": f"不支持的格式: {fmt}"}
    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass

    result["name"] = name
    result["format"] = fmt
    return result


# ─── PDF 双引擎解析 + 置信度合并 ───
def _ocr_pdf(path: str):
    """Render scanned PDF pages and run EasyOCR when text extraction is empty."""
    import io as _io
    from agents.vision_agent import _ocr_image
    paragraphs = []
    pages = 0
    try:
        import pypdfium2 as pdfium
        doc = pdfium.PdfDocument(path)
        pages = len(doc)
        try:
            for i in range(pages):
                page = doc[i]
                pil = page.render(scale=2).to_pil()
                buf = _io.BytesIO()
                pil.save(buf, format="PNG")
                text = _ocr_image(buf.getvalue())
                for line in text.splitlines():
                    line = line.strip()
                    if line:
                        paragraphs.append({"page": i + 1, "text": line})
        finally:
            doc.close()
    except ImportError:
        try:
            import fitz
            doc = fitz.open(path)
            pages = doc.page_count
            try:
                for i in range(pages):
                    pix = doc[i].get_pixmap(dpi=150)
                    text = _ocr_image(pix.tobytes("png"))
                    for line in text.splitlines():
                        line = line.strip()
                        if line:
                            paragraphs.append({"page": i + 1, "text": line})
            finally:
                doc.close()
        except ImportError:
            return None
    except Exception as exc:
        print(f"  [DocParse] OCR failed: {exc}")
        return None
    if not paragraphs:
        return None
    return {"paragraphs": paragraphs, "tables": [], "pages": pages}


def _parse_pdf(path: str, name: str) -> dict:
    """双引擎解析：pdfplumber 优先，PyPDF2 兜底。逐段去重，避免文字重复"""

    result_a = _try_engine("pdfplumber", path)
    result_b = _try_engine("pypdf2", path)

    if result_a is None and result_b is None:
        ocr = _ocr_pdf(path)
        if ocr:
            return _build_result(ocr, "OCR")
        return {"error": f"{name}: 无法读取内容，可能是扫描件（图片型PDF），请文字版或直接粘贴内容",
                "paragraphs": [], "tables": [], "pages": 0}

    selected = result_a if result_a is not None else result_b
    if selected and not selected.get("paragraphs"):
        ocr = _ocr_pdf(path)
        if ocr:
            return _build_result(ocr, "OCR")
    if result_a is not None:
        return _build_result(result_a, "pdfplumber")
    return _build_result(result_b, "PyPDF2(兜底)")


def _try_engine(engine: str, path: str):
    """尝试单个 PDF 引擎，返回 {paragraphs, tables, pages} 或 None"""
    try:
        if engine == "pdfplumber":
            import pdfplumber
            paragraphs, tables = [], []
            with pdfplumber.open(path) as pdf:
                pages = len(pdf.pages)
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if text:
                        for line in text.split("\n"):
                            line = line.strip()
                            if line:
                                paragraphs.append({"page": i + 1, "text": line})
                    else:
                        # 无文本 → 尝试逐词提取（处理分散字符的 PDF）
                        words = page.extract_words()
                        if words:
                            line_text = " ".join(w.get("text", "") for w in words)
                            paragraphs.append({"page": i + 1, "text": line_text})
                    for t in page.extract_tables():
                        if t:
                            tables.append({"page": i + 1, "rows": t[:20]})
            print(f"  [DocParse] pdfplumber: {pages}页, {len(paragraphs)}段, {len(tables)}表")
            return {"paragraphs": paragraphs, "tables": tables, "pages": pages}

        elif engine == "pypdf2":
            import PyPDF2
            paragraphs, tables = [], []
            reader = PyPDF2.PdfReader(path)
            pages = len(reader.pages)
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    for line in text.split("\n"):
                        line = line.strip()
                        if line:
                            paragraphs.append({"page": i + 1, "text": line})
            print(f"  [DocParse] PyPDF2: {pages}页, {len(paragraphs)}段")
            return {"paragraphs": paragraphs, "tables": tables, "pages": pages}
    except Exception as e:
        print(f"  [DocParse] {engine} 异常: {e}")
        return None


def _build_result(engine_result: dict, engine_name: str) -> dict:
    return {
        "paragraphs": engine_result["paragraphs"],
        "tables": engine_result["tables"],
        "pages": engine_result["pages"],
        "paragraph_count": len(engine_result["paragraphs"]),
        "table_count": len(engine_result["tables"]),
        "engine": engine_name,
        "confidence": {"high": engine_result["pages"], "medium": 0, "low": 0},
    }


# ─── DOCX 解析 ───
def _parse_docx(path: str, name: str) -> dict:
    try:
        from docx import Document
        doc = Document(path)
        paragraphs = []
        tables = []
        for para in doc.paragraphs:
            text = para.text.strip()
            if text:
                paragraphs.append({"style": para.style.name, "text": text})
        for t in doc.tables:
            rows = [[cell.text for cell in row.cells] for row in t.rows]
            tables.append({"rows": rows[:20]})
        return {"paragraphs": paragraphs, "tables": tables,
                "paragraph_count": len(paragraphs), "table_count": len(tables),
                "pages": "?"}
    except Exception as e:
        return {"error": f"{name}: Word 文档解析异常，请确认文件未损坏或另存后重试", "paragraphs": [], "tables": [], "pages": 0}


# ─── XLSX 解析 ───
def _parse_xlsx(path: str, name: str) -> dict:
    try:
        import openpyxl
        wb = openpyxl.load_workbook(path, data_only=True)
        sheets = []
        all_rows = []
        for sname in wb.sheetnames:
            ws = wb[sname]
            rows = []
            for row in ws.iter_rows(max_row=min(ws.max_row, 200), values_only=True):
                if any(c is not None for c in row):
                    r = [str(c) if c is not None else "" for c in row]
                    rows.append(r)
                    all_rows.append({"sheet": sname, "row": r})
            sheets.append({"name": sname, "rows": rows[:50], "total_rows": ws.max_row})
        return {"paragraphs": [{"text": json.dumps(r["row"], ensure_ascii=False)} for r in all_rows[:200]],
                "tables": [], "sheets": sheets, "pages": f"{len(sheets)} sheets",
                "paragraph_count": len(all_rows)}
    except Exception as e:
        return {"error": f"{name}: Excel 文件解析异常，请确认文件未加密且未损坏", "paragraphs": [], "tables": [], "pages": 0}


# ─── TXT 解析 ───
def _parse_txt(path: str, name: str) -> dict:
    encodings = ["utf-8", "gbk", "gb2312", "latin-1"]
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc) as f:
                text = f.read()
            lines = [l.strip() for l in text.split("\n") if l.strip()]
            return {"paragraphs": [{"text": l} for l in lines[:500]],
                    "tables": [], "paragraph_count": len(lines), "pages": "?"}
        except UnicodeDecodeError:
            continue
    return {"error": f"{name}: 无法识别文本编码，请转为 UTF-8 格式后重试", "paragraphs": [], "tables": [], "pages": 0}


# ─── 摘要生成（逐文档独立生成，使用 token 预算）───
def _compare_docs(query: str, docs: list, client: OpenAI) -> str:
    """多文档对比分析"""
    from structured_output import build_prompt_instruction, parse_llm_output

    docs_text = []
    for i, d in enumerate(docs[:5], 1):
        text = _doc_to_text(d, 2000)
        docs_text.append(f"[文档{i}] {d.get('name','?')}\n{text[:2000]}")
    combined = "\n\n---\n\n".join(docs_text)

    prompt = f"""对比以下 {len(docs)} 份文档，回答用户问题。用 comparison 格式输出对比结果。

{combined}

用户问题: {query}

请对比各文档在关键维度上的差异，包含具体引用。
{build_prompt_instruction()}"""

    try:
        resp = client.chat.completions.create(
            model=LLM_CONFIG["model"],
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1, max_tokens=max_tokens_for(query, "answer"),
        )
        return parse_llm_output(resp.choices[0].message.content.strip())
    except Exception:
        return {"answer": f"文档对比暂时失败，请重试。", "format": "comparison",
                "data": {"subjects": [d.get("name", "?") for d in docs[:2]]}}


def _generate_summary(docs: list, client: OpenAI) -> str:
    results = []
    for i, d in enumerate(docs, 1):
        text = _doc_to_text(d, max_chars=MAX_DOC_CHARS)
        doc_name = d.get("name", f"文档{i}")
        try:
            # V2: 使用 token_budget 自适应分配，而非硬编码 800
            summary_tokens = max_tokens_for(f"摘要: {doc_name}", "answer")
            resp = client.chat.completions.create(
                model=LLM_CONFIG["model"],
                messages=[{"role": "user", "content": f"""为以下文档生成摘要（200-400字），包含: 文档概述 + 章节结构 + 关键数据/结论。

文档名: {doc_name}

{text}"""}],
                temperature=0.1, max_tokens=summary_tokens,
            )
            summary = resp.choices[0].message.content.strip()
            results.append(f"### 文档{i}：{doc_name}\n\n{summary}")
        except Exception:
            results.append(f"**{doc_name}**: {d.get('pages','?')}页, {d.get('paragraph_count','?')}段")

    return "\n\n---\n\n".join(results)


# ─── 文档问答（V2: 元数据检索 + 更强兜底） ───
def _answer_question(query: str, docs: list, client: OpenAI) -> str:
    from structured_output import build_prompt_instruction, parse_llm_output

    # LLM 语义关键词提取（替代手写分词+停用词）
    all_keywords = _rewrite_query(query, client)
    if not all_keywords:
        all_keywords = _extract_keywords(query)  # LLM 失败时退化为空

    # ═══ V2: 同时检索段落文本 + 文件名/元数据 ═══
    CONTEXT_WINDOW = 5
    relevant = []
    for d in docs:
        paras = d.get("paragraphs", [])
        doc_name = d.get("name", "")

        # 段落级匹配
        scored = []  # (score, index, source)
        for i, p in enumerate(paras):
            score = sum(1 for kw in all_keywords if kw.lower() in p["text"].lower())
            if score > 0:
                scored.append((score, i, "paragraph"))

        # V2: 文件名/元数据匹配 —— 如果关键词命中文件名，标记整篇文档相关
        meta_score = sum(1 for kw in all_keywords if kw.lower() in doc_name.lower())
        if meta_score > 0:
            # 文件名匹配 → 把文档头部段落也加入（标题/摘要通常在开头）
            for i in range(min(5, len(paras))):
                already_matched = any(s[1] == i for s in scored)
                if not already_matched:
                    scored.append((meta_score + 2, i, "meta"))
            # 也检查表格
            for t in d.get("tables", []):
                scored.append((meta_score + 1, -1, "meta"))

        if not scored:
            continue

        scored.sort(key=lambda x: x[0], reverse=True)
        # 取 top 命中 + 上下文窗口扩展
        seen = set()
        passages = []
        for score, idx, source in scored[:8]:  # top 8 命中（V2 从 5 提升到 8）
            if idx < 0:
                continue
            # 上下文窗口: idx-CONTEXT_WINDOW 到 idx+CONTEXT_WINDOW
            for ni in range(max(0, idx - CONTEXT_WINDOW),
                           min(len(paras), idx + CONTEXT_WINDOW + 1)):
                if ni not in seen:
                    seen.add(ni)
                    passages.append((score if ni == idx else max(0, score - 1),
                                    paras[ni]))
        passages.sort(key=lambda x: x[0], reverse=True)
        relevant.append({"doc": d, "passages": passages[:20],  # V2: 从 15 提升到 20
                        "meta_match": meta_score > 0})

    # ═══ V2: 无关键词匹配 → LLM 相关性检查 + 多级兜底 ═══
    if not relevant:
        answerable, suggested_sections = _llm_relevance_check(query, docs, client)

        # 第 2 级：用 LLM 建议的关键词重新搜索
        if suggested_sections:
            for d in docs:
                paras = d.get("paragraphs", [])
                matched = []
                for s in suggested_sections:
                    for p in paras:
                        if s.lower() in p["text"].lower() and p not in matched:
                            matched.append(p)
                if matched:
                    relevant.append({"doc": d, "passages": [(1, p) for p in matched[:15]],
                                    "meta_match": False})

        # 第 3 级：兜底 —— 无论 answerable 是什么，都给 LLM 文档概览
        if not relevant:
            for d in docs:
                text = _doc_to_text(d, 3000)  # V2: 从 2000 提升到 3000
                if text.strip():
                    relevant.append({"doc": d, "passages": [
                        (0, {"text": f"[文档概览] 文件名: {d['name']}\n"
                                       f"共 {d.get('pages','?')} 页, "
                                       f"{d.get('paragraph_count','?')} 段\n"
                                       f"解析引擎: {d.get('engine','未知')}\n\n"
                                       f"{text[:3000]}",
                             "page": "全文概览"})
                    ], "meta_match": True})

    # 最终安全检查
    if not relevant:
        doc_names = ", ".join(d["name"] for d in docs)
        return (f"抱歉，在文档（{doc_names}）中未找到与「{query}」直接相关的内容。"
                f"建议您尝试：\n"
                f"1. 换个角度提问（例如用文档中可能出现的术语）\n"
                f"2. 先让我生成文档摘要了解全貌\n"
                f"3. 确认文档是否为扫描件（图片型PDF无法提取文字）")

    # ═══ 构建最终上下文：文档全局概览 + 匹配段落 ═══
    ctx_parts = []

    # 每份文档先放全局概览（确保 LLM 看到文档全貌，不依赖关键词命中）
    for d in docs:
        outline = f"## {d['name']}\n{d.get('pages','?')}页 {d.get('paragraph_count','?')}段"
        if d.get("tables"):
            outline += f" {len(d['tables'])}个表格"
        # 文档前 500 字作为结构预览（标题/摘要通常在开头）
        preview = _doc_to_text(d, 500)
        if preview:
            outline += f"\n前500字预览: {preview[:500]}"
        ctx_parts.append(outline)

    ctx_parts.append("---")

    # 关键词命中的具体段落
    for r in relevant:
        d = r["doc"]
        for score, p in r["passages"]:
            page = f"第{p.get('page','?')}页" if p.get("page") else ""
            ctx_parts.append(f"[{page}] {p['text']}")

    context = "\n".join(ctx_parts)
    prompt = f"""根据以下文档内容回答用户问题。

文档内容:
{context[:MAX_DOC_CHARS]}

用户问题: {query}

{build_prompt_instruction()}"""

    try:
        resp = client.chat.completions.create(
            model=LLM_CONFIG["model"],
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1, max_tokens=max_tokens_for(query, "answer"),
        )
        return parse_llm_output(resp.choices[0].message.content.strip())
    except Exception as e:
        # 最终降级: 返回找到的段落原文
        fallback = [f"关于「{query}」，文档中找到以下相关段落（LLM调用暂时失败）:"]
        for r in relevant:
            for _, p in r["passages"][:5]:
                fallback.append(f"- [{r['doc']['name']}] {p['text'][:200]}")
        return "\n".join(fallback)


def _llm_relevance_check(query: str, docs: list, client: OpenAI) -> tuple:
    """LLM 判断问题是否可基于文档回答，并返回建议的关键词定位

    V2: 改进采样策略 —— 文件名 + 表格也加入，更均匀覆盖全文
    """
    doc_info = []
    for d in docs:
        paras = d.get("paragraphs", [])
        n = len(paras)

        # V2: 多段均匀采样（头5 + 25%处3 + 50%处3 + 75%处3 + 尾5）
        q1, q3 = max(0, n // 4), max(0, 3 * n // 4)
        indices = set(range(min(5, n)))           # 头部 0-4
        indices.update(range(max(0, n//2-1), min(n, n//2+2)))  # 中部
        indices.update(range(max(0, q1-1), min(n, q1+2)))      # 1/4 处
        indices.update(range(max(0, q3-1), min(n, q3+2)))      # 3/4 处
        indices.update(range(max(0, n-5), n))                  # 尾部

        samples = [paras[i] for i in sorted(indices) if i < n]
        texts = [p["text"][:80] for p in samples]

        # 加入文件名信息
        info = f"{d['name']}: {d.get('pages','?')}页/{n}段"
        # 加入表格行数
        tables = d.get("tables", [])
        if tables:
            info += f"/{len(tables)}表"
        info += f"\n  抽样: {' | '.join(texts)}"
        doc_info.append(info)

    doc_text = "\n".join(doc_info)

    try:
        resp = client.chat.completions.create(
            model=LLM_CONFIG["model"],
            messages=[{
                "role": "system",
                "content": """判断用户问题能否从文档中回答。输出 JSON:
{"answerable": true/false, "keywords": ["定位词1", "定位词2"]}

注意:
- 文件名中包含的信息也算文档可回答（如文件名含"Q3报告"=回答"哪季度"）
- 表格数据也算
- 即使抽样没直接命中，也应考虑问题主题是否可能在文档中出现
- when in doubt, 倾向于 answerable=true（后续会用全文概览兜底）

如果能回答，keywords 列出文档中可能出现答案的关键词（同义词、正式表达、英文对应词）。"""
            }, {
                "role": "user",
                "content": f"文档信息:\n{doc_text[:2000]}\n\n用户问题: {query}"
            }],
            temperature=0.0, max_tokens=200,
        )
        result = json.loads(resp.choices[0].message.content.strip())
        return result.get("answerable", True), result.get("keywords", [])
    except Exception:
        # 解析失败 → 保守返回 True，让后续兜底处理
        return True, []


def _quick_summary(doc: dict) -> str:
    """一句话摘要，不占上下文"""
    name = doc.get("name", "?")
    pages = doc.get("pages", "?")
    paras = doc.get("paragraph_count", 0)
    tables = doc.get("table_count", 0)
    parts = [f"{name}: {pages}页, {paras}段"]
    if tables:
        parts.append(f"{tables}表")
    return ", ".join(parts)


def cleanup_session(sid: str):
    """删除会话数据（委托给 MemoryStore）"""
    get_store().cleanup_session(sid)
    print(f"  [DocParse] 会话{sid[:8]}: 已清理")


def _extract_keywords(query: str) -> list:
    """委托给 LLM 做语义关键词提取，不再手写分词规则"""
    return []  # 实际工作由 _rewrite_query 完成，这里只返回空列表避免重复


def _rewrite_query(query: str, client) -> list:
    """LLM 理解 query 语义 → 输出检索关键词列表（直接返回 list）"""
    try:
        resp = client.chat.completions.create(
            model=LLM_CONFIG["model"],
            messages=[{
                "role": "system",
                "content": (
                    "从用户问题中提取用于文档检索的关键词。"
                    "输出规则:\n"
                    "- 每行一个关键词，最多 10 个\n"
                    "- 包含同义词和近义表达（如'学历'→也输出'教育经历''教育背景''学位'）\n"
                    "- 包含文档中可能出现的正式表达\n"
                    "- 只输出关键词，不要序号、解释或标点"
                )
            }, {
                "role": "user",
                "content": query
            }],
            temperature=0.0, max_tokens=100,
        )
        lines = resp.choices[0].message.content.strip().split("\n")
        keywords = [l.strip() for l in lines if l.strip() and len(l.strip()) >= 2]
        print(f"  [DocParse] LLM关键词: '{query[:40]}' → {keywords[:8]}")
        return keywords
    except Exception:
        return []


def _doc_to_text(doc: dict, max_chars: int) -> str:
    """文档解析结果 → 纯文本"""
    parts = []
    for p in doc.get("paragraphs", [])[:200]:
        loc = f"P{p.get('page','')}" if p.get("page") else ""
        parts.append(f"[{loc}] {p['text']}" if loc else p["text"])
    text = "\n".join(parts)
    return text[:max_chars]
