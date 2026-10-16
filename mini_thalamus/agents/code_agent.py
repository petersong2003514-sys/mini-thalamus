"""Code Agent —— LLM 生成代码 + 沙箱执行 + Self-Correction"""
import subprocess, tempfile, os, re, json, sys, shutil, hashlib, time as _time

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PKG_ROOT)

from config import (
    SANDBOX_DOCKER_IMAGE,
    SANDBOX_MEMORY,
    SANDBOX_TIMEOUT,
    SANDBOX_USE_DOCKER,
    get_llm_config,
)
from sandbox import scrub_env

from openai import OpenAI
from token_budget import max_tokens_for

DANGEROUS = [
    r'os\.system\(', r'subprocess\.', r'eval\(', r'exec\(', r'__import__\(',
    r'open\(.*[\'\"]w', r'shutil\.rmtree', r'socket\.', r'ctypes',
    r'while\s+True',
    # 多步场景下禁止重抓数据
    r'import\s+requests', r'from\s+requests', r'urllib', r'urlopen',
    # 绕过检测的常见手段
    r'getattr\(', r'chr\(', r'compile\(', r'__builtins__',
    r'base64', r'exec\(', r'globals\(\)', r'locals\(\)',
    r'http\.client', r'ftplib', r'smtplib',
]

# ─── 图片输出目录 ───
_IMG_DIR = os.path.join(_PKG_ROOT, "static", "generated")
os.makedirs(_IMG_DIR, exist_ok=True)

CODE_GEN_PROMPT = """你是 Python 专家。根据用户需求生成代码。

规则:
1. 如果用户要求"写代码"/"实现"/"编写"但不要求执行 → 只输出代码文本
2. 如果用户要求"画图"/"分析数据"/"计算"/"执行" → 生成可执行代码:
   - ⚠️ 优先使用 query 中"上一步查询结果"提供的数据（不要重新用 requests 抓取）
   - 如果 query 中有具体数据（温度、日期、数值等），直接硬编码或用 dict/list 写入代码中
   - 完全没有数据时才用 numpy 生成模拟数据
   - 图表保存为 chart.png（当前目录），用 matplotlib
   - 最后用 print() 输出关键结论
3. 只输出 Python 代码，不要 markdown 标记，不要解释
4. 避免死循环、大循环(>10000次)
5. 禁止 import requests / urllib 等网络库——数据已在上下文中

⚠ 中文字体: matplotlib 画图前加:
import matplotlib; matplotlib.rcParams['font.sans-serif']=['SimHei','Microsoft YaHei','WenQuanYi Micro Hei','Noto Sans CJK SC','DejaVu Sans']; matplotlib.rcParams['axes.unicode_minus']=False

用户需求: {query}"""

AGENT_META = {
    "name": "Code Agent",
    "id": "code",
    "description": "执行 Python 代码：数据分析、画图、计算、文件处理",
        "keywords": ["画图", "计算", "python", "代码", "编写代码", "检查代码", "程序代码", "写程序", "实现", "运行代码", "脚本", "调试", "debug"],
    "intents": ["code_generation", "code_execution", "data_visualization", "file_processing"],
    "examples": ["请编写并检查程序代码", "写一段 Python 代码并运行", "用代码画趋势图", "计算并输出结果"],
    "icon": "💻",
    "capabilities": {
        "needs_history": True, "needs_files": True,
        "needs_long_term": False, "max_context_chars": 6000,
        "auto_chainable": False, "side_effects": "workspace_write", "needs_credentials": [],
        "actions": ["python_exec", "code_generation"],
        "can_handle_errors": ["syntax_error", "logic_error"],
    },
    "input_schema": {
        "required": ["instruction"],
        "properties": {"instruction": {"type": "string"}},
    },
    "output_schema": {"exports": ["answer", "data.structured"]},
}


class CodeAgent:
    """LLM 生成代码 + 沙箱执行 + 失败重试"""

    capabilities = {
        "needs_history": True,
        "needs_files": True,
        "needs_long_term": False,
        "max_context_chars": 6000,
        "auto_chainable": False,
        "side_effects": "workspace_write",
        "needs_credentials": [],
    }

    def __init__(self):
        cfg = get_llm_config()
        self.client = OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"])
        self._model = cfg["model"]

    def run(self, query: str, context: dict = None) -> dict:
        if self._looks_like_code(query):
            return self._execute_and_report(query)

        # LLM 判断: 只要代码展示 vs 需要执行
        show_only = self._is_show_code_request(query)

        code = self._generate_code(query, show_only=show_only)
        if not code:
            return {"status": "error", "answer": "代码生成失败"}

        if show_only:
            explanation = self._explain_code(code, query)
            # 智能多语言提示：如果已经是非 Python 语言，就不再提示
            detected = self._detect_language(query)
            other_langs = [l for l in ["Java", "JavaScript", "Go", "Rust", "C++", "C#"]
                          if l.lower() != detected.lower()]
            if detected == "Python":
                lang_hint = f"\n\n💡 需要其他语言版本（{'/'.join(other_langs)}）吗？请直接告诉我。"
            else:
                lang_hint = f"\n\n💡 这是 {detected} 版本。需要 Python 或 {'/'.join(other_langs[:3])} 版本吗？"
            return {
                "status": "success",
                "answer": f"{explanation}{lang_hint}",
                "data": {
                    "structured": {
                        "format": "code",
                        "data": {"code": code[:3000], "output": explanation},
                        "sources": [],
                        "confidence": "high",
                    }
                },
                "stdout": "", "stderr": "", "code": code,
            }

        # 执行模式（交互式调试：最多 3 次尝试，每次把错误喂给 LLM 修复）
        for attempt in range(3):
            result = self._execute_and_report(code)
            if result["status"] in ("success", "timeout"):
                return result
            if attempt < 2:  # 还有重试机会
                error_info = result.get("stderr", "")[:500]
                output_info = result.get("stdout", "")[:300]
                debug_hint = f"{query}\n\n上次执行出错:\n{error_info}"
                if output_info:
                    debug_hint += f"\n\n已输出:\n{output_info}"
                code = self._generate_code(debug_hint, show_only=False)
        return result

    def _detect_language(self, query: str) -> str:
        """从 query 中检测目标编程语言（取最后一次出现的语言，避免被历史污染）"""
        lowered = query.lower()
        lang_map = [
            ("python", "Python"), ("py", "Python"),       # Python 优先
            ("java", "Java"), ("javascript", "JavaScript"), ("js", "JavaScript"),
            ("c++", "C++"), ("cpp", "C++"), ("c#", "C#"), ("csharp", "C#"),
            ("go", "Go"), ("golang", "Go"), ("rust", "Rust"), ("ruby", "Ruby"),
            ("swift", "Swift"), ("kotlin", "Kotlin"), ("typescript", "TypeScript"),
            ("ts", "TypeScript"), ("php", "PHP"), ("scala", "Scala"),
            ("c语言", "C"), ("c #", "C#"),
        ]
        # 找最后一次出现的位置（最靠近当前用户消息）
        best_pos, best_lang = -1, "Python"
        for key, lang in lang_map:
            pos = lowered.rfind(key)  # rfind = 从右往左找
            if pos > best_pos:
                best_pos, best_lang = pos, lang
        return best_lang

    def _generate_code(self, query: str, show_only: bool = False) -> str:
        try:
            lang = self._detect_language(query)
            if show_only:
                system = (
                    f"你是 {lang} 编程专家。用户要你写一段 {lang} 代码。"
                    f"只输出 {lang} 代码本身，不要markdown标记，不要解释，不要print执行结果，不要示例调用。"
                    f"只输出完整的代码定义。"
                )
            else:
                system = CODE_GEN_PROMPT
            resp = self.client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": query},
                ],
                temperature=0.1, max_tokens=max_tokens_for(query, "thought"),
            )
            raw = resp.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = re.sub(r"^```\w*\n?", "", raw)
                raw = re.sub(r"\n?```$", "", raw)
            return raw
        except Exception as e:
            print(f"  [CodeAgent] LLM 生成失败: {e}")
            return ""

    def _explain_code(self, code: str, query: str) -> str:
        """LLM 生成代码的自然语言解释（非 Python 语言也会标注）"""
        try:
            lang = self._detect_language(query)
            resp = self.client.chat.completions.create(
                model=self._model,
                messages=[{
                    "role": "system",
                    "content": f"用 1-2 句话简要解释这段{lang}代码做什么。口语化，50字以内。如果语言不是Python要说明。"
                }, {
                    "role": "user",
                    "content": f"需求: {query}\n{lang}代码:\n{code[:2000]}\n\n简要解释:"
                }],
                temperature=0.0, max_tokens=100,
            )
            return f"📝 {resp.choices[0].message.content.strip()}"
        except Exception:
            return ""

    def _execute_and_report(self, code: str) -> dict:
        warnings = self._scan(code)
        if warnings:
            return {"status": "blocked", "answer": f"代码被拦截: {warnings}",
                    "stdout": "", "stderr": f"危险模式: {warnings}"}

        exec_result = self._execute(code)
        stdout = exec_result.get("stdout", "")
        stderr = exec_result.get("stderr", "")
        work_dir = exec_result.get("work_dir", "")

        # 检测生成的图片 → 复制到静态目录
        images = []
        if work_dir and os.path.isdir(work_dir):
            for fname in sorted(os.listdir(work_dir)):
                if re.search(r'\.(png|jpg|jpeg|gif|svg)$', fname, re.IGNORECASE):
                    url = _persist_image(work_dir, fname)
                    if url:
                        images.append(url)
            # 清理临时目录（图片已复制到 static/generated/）
            try:
                shutil.rmtree(work_dir, ignore_errors=True)
            except Exception:
                pass

        if exec_result["status"] == "success":
            text = stdout.strip() if stdout.strip() else "代码执行成功"
            if images:
                text += "\n" + "\n".join(f"![chart]({url})" for url in images)
            answer = text
        elif exec_result["status"] == "timeout":
            answer = f"代码执行超时({SANDBOX_TIMEOUT}s)"
        else:
            answer = f"代码执行出错:\n{stderr[:500]}" if stderr else "执行失败"

        structured = {
            "format": "code",
            "answer": answer,
            "data": {
                "code": code[:1000],
                "output": stdout[:500] if stdout else (stderr[:200] if stderr else ""),
                "images": images,
            },
            "sources": [],
            "confidence": "high" if exec_result["status"] == "success" else "medium",
        }

        return {
            "status": exec_result["status"],
            "answer": answer,
            "data": {"structured": structured},
            "stdout": stdout, "stderr": stderr,
            "code": code[:500], "images": images,
        }

    def _is_show_code_request(self, query: str) -> bool:
        """判断是否只需要展示代码（不执行）"""
        lowered = query.lower()
        # 非 Python 语言 → 绝对不能执行，只能展示
        non_python = ["c++", "java ", "javascript", "golang", "go语言", "rust", "c#", "c语言",
                      "ruby", "swift", "kotlin", "typescript", "php", "scala", "haskell"]
        if any(lang in lowered for lang in non_python):
            return True
        # 明确要"写"、"实现"、"再给我" → 展示
        if any(w in lowered for w in ["写一段", "写个", "实现一个", "编写", "代码怎么写", "示例代码",
                                       "再给我", "给我", "来一个", "来一段", "版本的"]):
            return True
        # 明确要"执行"、"运行"、"画图"、"计算" → 执行
        if any(w in lowered for w in ["执行", "运行", "画图", "计算", "统计", "分析数据"]):
            return False
        # 兜底：LLM 判断
        try:
            resp = self.client.chat.completions.create(
                model=self._model,
                messages=[{
                    "role": "system",
                    "content": "判断用户是想要「代码文本」（只看不执行）还是「执行任务」（运行代码产出结果）。只输出 show 或 run。"
                }, {
                    "role": "user", "content": query[:500]
                }],
                temperature=0.0, max_tokens=10,
            )
            return "show" in resp.choices[0].message.content.strip().lower()
        except:
            return False

    def _looks_like_code(self, text: str) -> bool:
        code_kw = ["import ", "def ", "class ", "print(", "plt.", "pd.", "df.",
                   "=", "for ", "if ", "return "]
        return sum(1 for kw in code_kw if kw in text) >= 3

    def _scan(self, code: str) -> list:
        return [pat for pat in DANGEROUS if re.search(pat, code)]

    def _execute(self, code: str) -> dict:
        if SANDBOX_USE_DOCKER:
            return self._docker_execute(code)
        return self._subprocess_execute(code)

    def _subprocess_execute(self, code: str) -> dict:
        work_dir = tempfile.mkdtemp(prefix="mt_")
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
                "exit_code": result.returncode,
                "work_dir": work_dir,
            }
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "stdout": "", "stderr": f"超时({SANDBOX_TIMEOUT}s)"}
        except Exception as e:
            return {"status": "error", "stdout": "", "stderr": str(e)}

    def _docker_execute(self, code: str) -> dict:
        work_dir = tempfile.mkdtemp(prefix="mt_docker_")
        script_path = os.path.join(work_dir, "script.py")
        with open(script_path, "w", encoding="utf-8") as handle:
            handle.write(code)
        command = [
            "docker", "run", "--rm",
            "--network", "none",
            "--memory", SANDBOX_MEMORY,
            "--cpus", "1",
            "--pids-limit", "64",
            "--read-only",
            "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m",
            "--mount", f"type=bind,source={work_dir},target=/workspace",
            "--workdir", "/workspace",
            SANDBOX_DOCKER_IMAGE,
            "python", "script.py",
        ]
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=SANDBOX_TIMEOUT,
                env=scrub_env(),
            )
            return {
                "status": "success" if result.returncode == 0 else "error",
                "stdout": (result.stdout or "")[:5000],
                "stderr": (result.stderr or "")[:5000],
                "exit_code": result.returncode,
                "work_dir": work_dir,
            }
        except subprocess.TimeoutExpired:
            shutil.rmtree(work_dir, ignore_errors=True)
            return {
                "status": "timeout",
                "stdout": "",
                "stderr": f"Docker 执行超时({SANDBOX_TIMEOUT}s)",
            }
        except Exception as exc:
            shutil.rmtree(work_dir, ignore_errors=True)
            return {"status": "error", "stdout": "", "stderr": str(exc)}


def _persist_image(work_dir: str, fname: str) -> str:
    """复制沙箱图片到静态目录，返回 URL"""
    src = os.path.join(work_dir, fname)
    h = hashlib.md5(f"{_time.time()}{fname}".encode()).hexdigest()[:8]
    ext = os.path.splitext(fname)[1]
    dst_name = f"chart_{h}{ext}"
    dst = os.path.join(_IMG_DIR, dst_name)
    try:
        shutil.copy2(src, dst)
        return f"/static/generated/{dst_name}"
    except Exception as e:
        print(f"  [CodeAgent] 图片复制失败: {e}")
        return ""
