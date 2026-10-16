"""Vision Agent —— EasyOCR 文字提取 + LLM 分析

无文字图片需装 Ollama 视觉模型:
  1. 下载安装: https://ollama.com/download/windows
  2. set OLLAMA_MODELS=E:/ollama_models
  3. ollama pull moondream:1.8b
  4. ollama serve
  → 自动检测并使用
"""
import os as _os
_os.environ["FLAGS_use_onednn"] = "0"
_os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys, os, json, base64, io, re, tempfile, urllib.request

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PKG_ROOT)
import importlib.util
_cfg_path = os.path.join(_PKG_ROOT, "config.py")
_spec = importlib.util.spec_from_file_location("vision_config", _cfg_path)
_cfg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cfg)
LLM_CONFIG = _cfg.LLM_CONFIG

from openai import OpenAI
from token_budget import max_tokens_for
from structured_output import build_prompt_instruction, parse_llm_output

try:
    from logging_config import get_logger
    _log = get_logger("vision_agent")
except Exception:
    import logging as _logging
    _log = _logging.getLogger("vision_agent")

_ocr_reader = None


def _init_ocr():
    global _ocr_reader
    if _ocr_reader is not None:
        return
    try:
        import easyocr
        try:
            _r = easyocr.Reader(['ch_sim', 'en'], gpu=False, verbose=False)
        except Exception:
            _r = easyocr.Reader(['ch_sim'], gpu=False, verbose=False)
        class _W:
            def __init__(self, r): self._r = r
            def ocr(self, path):
                return self._r.readtext(path)
        _ocr_reader = _W(_r)
        _log.info("EasyOCR 已加载")
    except Exception as e:
        _log.warning("EasyOCR 不可用: %s", str(e)[:150])
        _ocr_reader = False


def _ocr_image(image_bytes: bytes) -> str:
    _init_ocr()
    if not _ocr_reader:
        return ""
    from PIL import Image as PImg
    img = PImg.open(io.BytesIO(image_bytes)).convert("RGB")
    import tempfile as _tempfile
    with _tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = os.path.join(tmp_dir, "ocr.png")
        img.save(tmp_path, format="PNG")
        results = _ocr_reader.ocr(tmp_path)
    if not results:
        return ""
    lines = []
    for line in results:
        item = line[1] if len(line) >= 2 else ""
        text = item[0] if isinstance(item, (list, tuple)) else item
        if isinstance(text, str) and text.strip():
            lines.append(text.strip())
    return "\n".join(lines)

AGENT_META = {
    "name": "Vision Agent",
    "id": "vision",
    "description": "图片理解：识别文字、分析图表、理解截图内容",
    "keywords": ["图片", "截图", "识别"],
    "icon": "👁️",
    "capabilities": {
        "needs_history": False, "needs_files": True,
        "needs_long_term": False, "max_context_chars": 2000,
        "auto_chainable": True, "side_effects": "local", "needs_credentials": [],
    },
    "input_schema": {"required": ["instruction"], "properties": {}},
    "output_schema": {"exports": ["answer", "summary", "data.structured"]},
}


class VisionAgent:
    capabilities = {
        "needs_history": False,
        "needs_files": True,
        "needs_long_term": False,
        "max_context_chars": 2000,
        "auto_chainable": True,
        "side_effects": "local",
        "needs_credentials": [],
    }

    def __init__(self, mode: str = "ocr"):
        self.client = OpenAI(api_key=LLM_CONFIG["api_key"], base_url=LLM_CONFIG["base_url"])
        self.mode = mode

    def run(self, query: str, context: dict = None) -> dict:
        ctx = context or {}
        files = ctx.get("files", [])
        if not files:
            return {"status": "error", "answer": "请上传图片文件（PNG/JPG/WEBP）。", "data": {}}

        # 预计算 OCR 结果，缓存避免 any() 决策与 _ocr_analyze 重复调用 _ocr_image
        ocr_cache: dict = {}
        for i, f in enumerate(files):
            cb = f.get("content")
            if not cb:
                continue
            if isinstance(cb, str):
                cb = cb.encode()
            elif not isinstance(cb, bytes):
                continue
            ocr_cache[i] = _ocr_image(cb)

        # 尝试视觉 API（Ollama 或 VISION_MODEL）
        if self.mode == "visual" or not any(ocr_cache.values()):
            vis_result = self._try_visual(query, files)
            if vis_result:
                return vis_result

        # OCR 文字分析（传入缓存避免二次 OCR）
        return self._ocr_analyze(query, files, ocr_cache)

    def _ocr_analyze(self, query, files, ocr_cache: dict = None):
        descs = []
        for i, f in enumerate(files):
            cb = f.get("content")
            if not cb: continue
            if isinstance(cb, str): cb = cb.encode()
            ocr = (ocr_cache or {}).get(i) or _ocr_image(cb)
            name = f.get("name", "?")
            if ocr:
                descs.append(f"[{name}]\n图片文字:\n{ocr[:3000]}")
            else:
                try:
                    from PIL import Image as PImg
                    img = PImg.open(io.BytesIO(cb))
                    descs.append(f"[{name}]\n尺寸:{img.size[0]}x{img.size[1]} 格式:{img.format}\n(图中未检测到文字)\n提示:安装Ollama+moondream获取视觉理解: ollama pull moondream:1.8b")
                except:
                    descs.append(f"[{name}]\n(无法解析)")

        prompt = "\n\n---\n\n".join(descs)
        prompt += f"\n\n用户问题: {query}\n\n{build_prompt_instruction()}"
        try:
            resp = self.client.chat.completions.create(
                model=LLM_CONFIG["model"], messages=[{"role":"user","content":prompt}],
                temperature=0.1, max_tokens=max_tokens_for(query,"answer"))
            s = parse_llm_output(resp.choices[0].message.content.strip())
        except Exception as e:
            return {"status":"error","answer":f"分析失败:{e}","data":{}}
        return {"status":"success","answer":s.get("answer",""),"data":{"structured":s},"summary":s.get("answer","")[:300]}

    def _try_visual(self, query, files):
        # 1. Ollama
        try:
            req = urllib.request.Request("http://localhost:11434/api/tags", method="GET")
            models = [m["name"] for m in json.loads(urllib.request.urlopen(req, timeout=3).read()).get("models", [])]
            for kw in ("moondream","llava","minicpm","bakllava"):
                for m in models:
                    if kw in m.lower():
                        return self._call_vision(query, files, "http://localhost:11434/v1", m)
        except: pass
        # 2. VISION_MODEL
        vm = os.environ.get("VISION_MODEL","")
        if vm:
            return self._call_vision(query, files, os.environ.get("VISION_BASE_URL", LLM_CONFIG["base_url"]), vm,
                                     os.environ.get("VISION_API_KEY", LLM_CONFIG["api_key"]))
        return None

    def _call_vision(self, query, files, base_url, model, api_key=None):
        if api_key is None: api_key = LLM_CONFIG["api_key"]
        try:
            vc = OpenAI(api_key=api_key, base_url=base_url)
            content = [{"type":"text","text":query+"\n\n"+build_prompt_instruction()}]
            for f in files:
                cb = f.get("content")
                if isinstance(cb, str): cb = cb.encode()
                content.append({"type":"image_url","image_url":{"url":f"data:image/png;base64,{base64.b64encode(cb).decode()}"}})
            resp = vc.chat.completions.create(model=model, messages=[{"role":"user","content":content}],
                                              temperature=0.1, max_tokens=max_tokens_for(query,"answer"))
            s = parse_llm_output(resp.choices[0].message.content.strip())
        except Exception as e:
            return None
        return {"status":"success","answer":s.get("answer",""),"data":{"structured":s,"mode":"visual"},"summary":s.get("answer","")[:300]}
