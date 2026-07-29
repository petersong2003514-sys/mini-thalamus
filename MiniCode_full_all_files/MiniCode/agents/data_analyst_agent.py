"""DataAnalyst Agent —— 上传 CSV/Excel → 自动数据分析 + 图表 + 洞察

使用:
  上传 CSV/Excel 文件 → 自动路由到 DataAnalyst
  支持追问: "按地区分组", "找出异常值", "对比Q1和Q2"
"""
import sys, os, json, re, tempfile, subprocess, shutil, hashlib, time as _time

_MINICODE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _MINICODE_ROOT)

from config import get_llm_config, SANDBOX_TIMEOUT
from sandbox import scrub_env

from openai import OpenAI
from token_budget import max_tokens_for
from structured_output import build_prompt_instruction, parse_llm_output

try:
    from logging_config import get_logger
    _log = get_logger("data_analyst")
except Exception:
    import logging as _logging
    _log = _logging.getLogger("data_analyst")

# 图表输出目录
_IMG_DIR = os.path.join(_MINICODE_ROOT, "static", "generated")
os.makedirs(_IMG_DIR, exist_ok=True)

# ─── 数据集缓存（按 session 缓存解析后的 CSV 路径）───
_dataset_cache = {}


def _parse_to_csv(file_data: dict, sid: str) -> tuple:
    """解析上传的文件为 CSV，返回 (csv_path, profile_dict)"""
    content = file_data.get("content")
    name = file_data.get("name", "data")
    if isinstance(content, str):
        content = content.encode("utf-8")

    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""

    tmp = tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False)
    try:
        tmp.write(content)
        tmp.close()

        import pandas as pd
        if ext in ("xlsx", "xls"):
            df = pd.read_excel(tmp.name)
        else:
            # CSV: 尝试多种编码
            for enc in ["utf-8", "gbk", "gb2312", "latin-1"]:
                try:
                    df = pd.read_csv(tmp.name, encoding=enc)
                    break
                except (UnicodeDecodeError, UnicodeError):
                    continue
            else:
                df = pd.read_csv(tmp.name, encoding="utf-8", errors="replace")

        # 存为干净的 CSV 供沙箱使用
        cache_path = os.path.join(_MINICODE_ROOT, "data", f"_ds_{sid}.csv")
        df.to_csv(cache_path, index=False)
        _dataset_cache[sid] = cache_path

        # 生成 profile
        profile = {
            "name": name,
            "rows": len(df),
            "columns": len(df.columns),
            "col_names": list(df.columns),
            "dtypes": {str(c): str(df[c].dtype) for c in df.columns},
            "nulls": {str(c): int(df[c].isna().sum()) for c in df.columns},
            "numeric_cols": [str(c) for c in df.select_dtypes(include="number").columns],
            "numeric_summary": {},
        }
        for c in profile["numeric_cols"]:
            try:
                s = df[c].dropna()
                profile["numeric_summary"][c] = {
                    "min": round(float(s.min()), 2),
                    "max": round(float(s.max()), 2),
                    "mean": round(float(s.mean()), 2),
                    "median": round(float(s.median()), 2),
                    "std": round(float(s.std()), 2),
                }
            except Exception:
                pass

        # 分类列 top 值
        cat_cols = [c for c in df.columns if c not in profile["numeric_cols"]]
        profile["categorical_cols"] = cat_cols[:10]  # 取前 10 个
        profile["cat_sample"] = {}
        for c in cat_cols[:5]:
            try:
                top = df[c].value_counts().head(5).to_dict()
                profile["cat_sample"][str(c)] = {str(k): int(v) for k, v in top.items()}
            except Exception:
                pass

        return cache_path, profile

    except Exception as e:
        return None, {"error": str(e)}
    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass

AGENT_META = {
    "name": "DataAnalyst Agent",
    "id": "data_analyst",
    "description": "上传CSV/Excel，自动数据分析：统计摘要、趋势图、异常检测",
    "keywords": ["分析数据", "csv", "excel", "趋势"],
    "icon": "📈",
    "capabilities": {
        "needs_history": True, "needs_files": True,
        "needs_long_term": False, "max_context_chars": 6000,
        "auto_chainable": True, "side_effects": "local", "needs_credentials": [],
    },
    "input_schema": {"required": ["instruction"], "properties": {}},
    "output_schema": {"exports": ["answer", "summary", "data.structured"]},
}


class DataAnalystAgent:
    """自动数据分析 Agent"""

    capabilities = {
        "needs_history": True,
        "needs_files": True,
        "needs_long_term": False,
        "max_context_chars": 6000,
        "auto_chainable": True,
        "side_effects": "local",
        "needs_credentials": [],
    }

    def __init__(self):
        cfg = get_llm_config()
        self.client = OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"])
        self._model = cfg["model"]

    def run(self, query: str, context: dict = None) -> dict:
        ctx = context or {}
        files = ctx.get("files", [])
        sid = ctx.get("session_id", "default")

        # ── 跨轮持久化 session 状态（不写入长期知识）──
        from memory_service import get_memory_service
        memory = get_memory_service()

        # ── 有新文件 → 解析并缓存路径 ──
        if files:
            f = files[0]
            csv_path, profile = _parse_to_csv(f, sid)
            if profile.get("error"):
                return {"status": "error",
                        "answer": f"文件解析失败: {profile['error']}\n支持 CSV、Excel(.xlsx/.xls) 格式。",
                        "data": {}}
            # 持久化数据集路径到 session state
            memory.set_session_state(sid, "dataset", {"path": csv_path, "name": profile.get("name", "")})
        else:
            # 从 session state 恢复
            cached = memory.get_session_state(sid, "dataset")
            if cached and isinstance(cached, dict):
                csv_path = cached.get("path", "")
            else:
                csv_path = _dataset_cache.get(sid, "")
            if not csv_path or not os.path.exists(csv_path):
                return {"status": "error",
                        "answer": "请先上传 CSV 或 Excel 数据文件。",
                        "data": {}}
            # 从缓存 CSV 重建 profile
            import pandas as pd
            df = pd.read_csv(csv_path)
            profile = {"name": os.path.basename(csv_path), "rows": len(df),
                       "columns": len(df.columns), "col_names": list(df.columns)}

        _log.info("数据集: %s (%d 行 x %d 列), 查询: %s",
                  profile.get("name", "?"), profile.get("rows", 0),
                  profile.get("columns", 0), query[:80])

        # ── LLM 生成分析代码 ──
        code = self._generate_analysis_code(query, csv_path, profile)
        if not code:
            return {"status": "error", "answer": "代码生成失败", "data": {}}

        # ── 沙箱执行 ──
        exec_result = self._sandbox_execute(code)
        stdout = exec_result.get("stdout", "")
        stderr = exec_result.get("stderr", "")
        work_dir = exec_result.get("work_dir", "")

        # ── 提取图表 ──
        images = []
        if work_dir and os.path.isdir(work_dir):
            for fname in sorted(os.listdir(work_dir)):
                if re.search(r'\.(png|jpg|jpeg|gif|svg)$', fname, re.IGNORECASE):
                    url = _persist_image(work_dir, fname)
                    if url:
                        images.append(url)
            try:
                shutil.rmtree(work_dir, ignore_errors=True)
            except Exception:
                pass

        # ── LLM 解读结果 ──
        insights = self._interpret(query, profile, stdout, stderr, images)

        # ── 构建结构化输出（表格 + 图表）──
        structured_data = {"code": code[:2000], "output": stdout[:1000], "images": images}
        fmt = "code"

        # 如果有数值列，生成统计表格供前端结构化渲染
        num_cols = profile.get("numeric_cols", [])
        if num_cols and profile.get("numeric_summary"):
            headers = [""] + num_cols
            stats_rows = []
            for stat in ["min", "max", "mean", "median", "std"]:
                row = [stat]
                for c in num_cols:
                    row.append(str(profile["numeric_summary"].get(c, {}).get(stat, "-")))
                stats_rows.append(row)
            structured_data["headers"] = headers
            structured_data["rows"] = stats_rows
            fmt = "table" if not images else "code"

        return {
            "status": "success",
            "answer": insights,
            "data": {
                "structured": {
                    "format": fmt,
                    "data": structured_data,
                    "confidence": "high",
                },
                "profile": profile,
            },
            "summary": insights[:300],
        }

    def _generate_analysis_code(self, query: str, csv_path: str, profile: dict) -> str:
        """LLM 根据数据 profile 生成 pandas 分析代码"""
        profile_text = json.dumps(profile, ensure_ascii=False, indent=2)
        prompt = f"""你是数据分析专家。数据已加载为 CSV: {csv_path}

数据概况:
{profile_text}

用户需求: {query}

写一段 Python 代码分析数据。要求:
1. 用 pandas 读取 CSV: df = pd.read_csv(r'{csv_path}')
2. 根据用户需求做相应分析（统计/分组/排序/筛选/异常检测等）
3. 如果需要可视化，用 matplotlib 画图，保存为 chart.png
   - 中文字体: plt.rcParams['font.sans-serif'] = ['SimHei','Microsoft YaHei','DejaVu Sans']
   - plt.rcParams['axes.unicode_minus'] = False
4. 用 print() 输出关键结论和数据（简洁明了）
5. 只输出 Python 代码，不要 markdown 标记和解释
6. 避免 requests/urlopen 等网络请求——数据已在 CSV 中"""

        try:
            resp = self.client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1, max_tokens=max_tokens_for(query, "thought"),
            )
            code = resp.choices[0].message.content.strip()
            if code.startswith("```"):
                code = re.sub(r"^```\w*\n?", "", code)
                code = re.sub(r"\n?```$", "", code)
            return code
        except Exception as e:
            _log.warning("代码生成失败: %s", e)
            return ""

    def _sandbox_execute(self, code: str) -> dict:
        """在子进程沙箱中执行代码"""
        work_dir = tempfile.mkdtemp(prefix="shushu_da_")
        script_path = os.path.join(work_dir, "script.py")
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(code)
        try:
            env = scrub_env()
            env["PYTHONIOENCODING"] = "utf-8"
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True, text=True, encoding="utf-8",
                timeout=SANDBOX_TIMEOUT, cwd=work_dir,
                errors="replace", env=env,
            )
            return {
                "status": "success" if result.returncode == 0 else "error",
                "stdout": (result.stdout or "")[:5000],
                "stderr": (result.stderr or "")[:5000],
                "work_dir": work_dir,
            }
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "stdout": "", "stderr": f"超时({SANDBOX_TIMEOUT}s)", "work_dir": work_dir}
        except Exception as e:
            return {"status": "error", "stdout": "", "stderr": str(e), "work_dir": work_dir}

    def _interpret(self, query: str, profile: dict, stdout: str, stderr: str, images: list) -> str:
        """LLM 解读执行结果"""
        prompt = f"""根据数据分析结果回答用户问题。用通俗语言解释数据发现的洞察。

数据概况: {json.dumps(profile, ensure_ascii=False)[:1000]}
用户问题: {query}
分析输出:
{stdout[:3000]}
错误输出（如有）:
{stderr[:500]}
图表: {'已生成' if images else '无'}

{build_prompt_instruction()}"""

        try:
            resp = self.client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3, max_tokens=max_tokens_for(query, "answer"),
            )
            structured = parse_llm_output(resp.choices[0].message.content.strip())
            return structured.get("answer", stdout[:500])
        except Exception as e:
            return f"分析完成。\n{stdout[:500]}"


def _persist_image(work_dir: str, fname: str) -> str:
    """复制图片到 static/generated/，返回 URL"""
    src = os.path.join(work_dir, fname)
    if not os.path.isfile(src):
        return ""
    h = hashlib.md5(fname.encode() + str(_time.time()).encode()).hexdigest()[:8]
    dest_name = f"da_{h}.png"
    dest = os.path.join(_IMG_DIR, dest_name)
    shutil.copy2(src, dest)
    return f"/static/generated/{dest_name}"
