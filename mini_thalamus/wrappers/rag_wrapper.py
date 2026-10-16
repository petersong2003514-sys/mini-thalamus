"""RAG Agent —— 知识库检索（语义向量 + 关键词双引擎）

知识库文件存放在 data/knowledge_base/，跨 session 持久。
用户在聊天中上传文件自动加入知识库，RAG 从知识库中检索。

检索策略:
  1. 向量语义检索（sentence-transformers embedding → 余弦相似度）
  2. 关键词匹配兜底（向量模型不可用时）
"""
import sys, os, json, re
import numpy as np

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PKG_ROOT)

import importlib.util
_cfg_path = os.path.join(_PKG_ROOT, "config.py")
_spec = importlib.util.spec_from_file_location("mt_rag_config", _cfg_path)
_cfg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cfg)
LLM_CONFIG = _cfg.LLM_CONFIG

from openai import OpenAI
from token_budget import max_tokens_for

# 日志
try:
    from logging_config import get_logger
    _log = get_logger("rag_agent")
except Exception:
    import logging as _logging
    _log = _logging.getLogger("rag_agent")

KB_DIR = os.path.join(_PKG_ROOT, "data", "knowledge_base")
os.makedirs(KB_DIR, exist_ok=True)

# ─── 向量模型（懒加载，sentence-transformers 已安装）───
_embed_model = None


def _get_embed_model():
    """懒加载语义向量模型（~500MB，首次加载后缓存）"""
    global _embed_model
    if _embed_model is not None:
        return _embed_model
    try:
        # Windows DLL conflict: torch must not load before pyarrow.
        import pyarrow  # noqa: F401
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer(
            "paraphrase-multilingual-MiniLM-L12-v2",
            device="cpu",
        )
        _log.info("向量模型已加载: MiniLM-L12-v2")
        return _embed_model
    except Exception as e:
        _log.warning("向量模型不可用: %s，降级为关键词匹配", str(e)[:150])
        _embed_model = False
        return None


def _embed_texts(texts: list) -> np.ndarray:
    """将文本列表转为向量（已归一化）"""
    model = _get_embed_model()
    if not model:
        return None
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """余弦相似度（输入已归一化时等于点积）"""
    return float(np.dot(a, b))


class RAGWrapper:
    """知识库检索 Agent —— 文件解析 → 存储 → LLM 检索问答"""

    capabilities = {
        "auto_chainable": True,
        "side_effects": "none",
        "needs_credentials": [],
    }

    def __init__(self):
        self.client = OpenAI(api_key=LLM_CONFIG["api_key"], base_url=LLM_CONFIG["base_url"])

    def run(self, query: str, context: dict = None) -> dict:
        ctx = context or {}
        files = ctx.get("files", [])
        sid = ctx.get("session_id", "kb")

        # ── 有新文件 → 解析并加入知识库 ──
        if files:
            from agents.docparse_agent import _parse_file
            for f in files:
                parsed = _parse_file(f)
                if parsed.get("error"):
                    continue
                # 保存到知识库目录
                safe_name = re.sub(r'[\\/:*?"<>|]', '_', f["name"]) + ".json"
                path = os.path.join(KB_DIR, safe_name)
                with open(path, "w", encoding="utf-8") as fp:
                    json.dump(parsed, fp, ensure_ascii=False, default=str)
                _log.info("知识库新增: %s", f['name'])

        # ── 加载知识库 ──
        docs = []
        for fname in os.listdir(KB_DIR):
            if fname.endswith(".json"):
                path = os.path.join(KB_DIR, fname)
                try:
                    with open(path, "r", encoding="utf-8") as fp:
                        docs.append(json.loads(fp.read()))
                except Exception:
                    pass

        _log.info("知识库: %d 个文档, 查询: %s", len(docs), query[:80])
        if not docs:
            return {"status": "error",
                    "answer": (f"知识库为空。请上传知识库文件（文档/手册/规定等），"
                               "系统会自动解析并加入知识库，之后即可检索问答。"),
                    "data": {"structured": {"format": "text", "data": {}, "sources": [], "confidence": "low"}}}

        # ── 检索 + 回答（语义优先 → 关键词兜底）──
        relevant = self._semantic_retrieve(query, docs)
        if not relevant:
            keywords = self._llm_keywords(query)
            relevant = self._keyword_retrieve(docs, keywords)
            _log.info("语义无结果，关键词匹配: %d 相关段落", len(relevant))
        else:
            _log.info("语义检索: %d 相关段落", len(relevant))
        structured = self._generate_answer(query, relevant, docs)

        kb_files = [d.get("name", "?") for d in docs]
        return {
            "status": "success",
            "answer": structured.get("answer", ""),
            "data": {
                "structured": structured,
                "knowledge_base_files": kb_files,
                "matched_chunks": len(relevant),
            },
            "summary": structured.get("answer", "")[:300],
        }

    def _llm_keywords(self, query: str) -> list:
        try:
            resp = self.client.chat.completions.create(
                model=LLM_CONFIG["model"],
                messages=[{
                    "role": "system",
                    "content": "从用户问题提取检索关键词（每行一个，含同义词）。最多 10 个。只输出关键词。"
                }, {"role": "user", "content": query}],
                temperature=0.0, max_tokens=100,
            )
            lines = resp.choices[0].message.content.strip().split("\n")
            return [l.strip() for l in lines if l.strip() and len(l.strip()) >= 2]
        except Exception:
            return []

    def _semantic_retrieve(self, query: str, docs: list) -> list:
        """向量语义检索：query embedding vs 所有段落 embedding → top-K"""
        model = _get_embed_model()
        if not model:
            return []

        # 收集所有段落
        all_paras = []
        for d in docs:
            for p in d.get("paragraphs", []):
                if p.get("text", "").strip():
                    all_paras.append((d, p))

        if not all_paras:
            return []

        # 计算 query embedding
        q_vec = _embed_texts([query])
        if q_vec is None:
            return []
        q_vec = q_vec[0]

        # 批量计算段落 embeddings
        texts = [p["text"] for _, p in all_paras]
        p_vecs = _embed_texts(texts)
        if p_vecs is None:
            return []

        # 余弦相似度 → top-10
        scores = [(_cosine_similarity(q_vec, p_vecs[i]), i) for i in range(len(p_vecs))]
        scores.sort(key=lambda x: x[0], reverse=True)

        # 取 top-10，带上下文窗口（±3）
        seen = set()
        doc_paras = {}
        for score, idx in scores[:10]:
            if score < 0.2:  # 相似度太低，跳过
                continue
            d, p = all_paras[idx]
            doc_name = d.get("name", "?")
            if doc_name not in doc_paras:
                doc_paras[doc_name] = {"doc": d, "indices": set()}
            paras = d.get("paragraphs", [])
            p_idx = paras.index(p) if p in paras else idx
            # 上下文窗口
            for ni in range(max(0, p_idx - 3), min(len(paras), p_idx + 4)):
                if ni not in doc_paras[doc_name]["indices"]:
                    doc_paras[doc_name]["indices"].add(ni)

        relevant = []
        for entry in doc_paras.values():
            passages = [entry["doc"]["paragraphs"][i] for i in sorted(entry["indices"])]
            relevant.append({"doc": entry["doc"], "passages": passages[:10]})
        return relevant

    def _keyword_retrieve(self, docs: list, keywords: list) -> list:
        """关键词 + 上下文窗口检索（语义不可用时的兜底）"""
        if not keywords:
            return []
        relevant = []
        for d in docs:
            paras = d.get("paragraphs", [])
            scored = []
            for i, p in enumerate(paras):
                score = sum(1 for kw in keywords if kw.lower() in p["text"].lower())
                if score > 0:
                    scored.append((score, i))
            if not scored:
                continue
            scored.sort(key=lambda x: x[0], reverse=True)
            seen = set()
            passages = []
            for score, idx in scored[:5]:
                for ni in range(max(0, idx - 3), min(len(paras), idx + 4)):
                    if ni not in seen:
                        seen.add(ni)
                        passages.append(paras[ni])
            relevant.append({"doc": d, "passages": passages[:10]})
        return relevant

    def _generate_answer(self, query: str, relevant: list, all_docs: list) -> dict:
        from structured_output import build_prompt_instruction, parse_llm_output

        ctx_parts = []
        for d in all_docs:
            ctx_parts.append(f"知识库文件: {d.get('name','?')} ({d.get('pages','?')}页 {d.get('paragraph_count','?')}段)")
        ctx_parts.append("---")

        if relevant:
            for r in relevant:
                d = r["doc"]
                ctx_parts.append(f"\n[{d.get('name','?')}]:")
                for p in r["passages"]:
                    page = f"P{p.get('page','?')}" if p.get("page") else ""
                    ctx_parts.append(f"  [{page}] {p['text']}")
        else:
            for d in all_docs:
                from agents.docparse_agent import _doc_to_text
                text = _doc_to_text(d, 2000)
                if text.strip():
                    ctx_parts.append(f"\n[{d.get('name','?')} 全文概览]:\n{text[:2000]}")

        context = "\n".join(ctx_parts)
        prompt = f"""根据以下知识库内容回答用户问题。如果答案是列表型信息（步骤/规定/要点），用 list 格式。

知识库内容:
{context[:12000]}

用户问题: {query}

{build_prompt_instruction()}"""

        try:
            resp = self.client.chat.completions.create(
                model=LLM_CONFIG["model"],
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1, max_tokens=max_tokens_for(query, "answer"),
            )
            return parse_llm_output(resp.choices[0].message.content.strip())
        except Exception:
            return {"answer": f"关于「{query}」，知识库检索暂时失败，请重试。",
                    "format": "text", "data": {}, "sources": [],
                    "confidence": "low"}
