"""
LLM 抽象层 —— 封装大模型调用。

两种实现:
- MockLLM: 零依赖演示模式，展示 Agent 循环逻辑（学习用）
- OpenAILLM: 真实 API 调用，支持 OpenAI 及所有兼容接口
  （DeepSeek / Qwen / Zhipu / Moonshot / Ollama 等）
"""

import re
import json
import time
import os
from abc import ABC, abstractmethod


class LLMBase(ABC):
    """LLM 抽象基类"""

    @abstractmethod
    def generate(self, messages: list[dict[str, str]]) -> str:
        ...

    def generate_with_tools(
        self, messages: list[dict[str, str]], tools: list[dict],
        tool_choice: str | dict = "auto",
    ) -> dict:
        """
        带工具调用的 LLM 请求（阶段 2 新增）。

        与 generate() 的区别:
        - generate():  返回纯文本 str，Agent 自己用正则解析工具调用
        - generate_with_tools():  返回结构化 dict，API 保证 tool_calls 字段正确

        返回格式:
          {"type": "text",      "content": "..."}           # LLM 直接回复
          {"type": "tool_call", "calls": [{"name": ..., "arguments": {...}}]}  # 工具调用

        默认实现: 回退到 generate()（子类可覆盖）
        """
        text = self.generate(messages)
        return {"type": "text", "content": text}


# ═══════════════════════════════════════════════════════════════
#  MockLLM — 零依赖演示，展示 Agent 框架逻辑
# ═══════════════════════════════════════════════════════════════

class MockLLM(LLMBase):
    """
    模拟 LLM，不调任何 API，用规则生成响应。

    设计目的: 在没有 API key 的情况下也能观察 Agent 循环机制。
    限制: 不做真正的语言理解，只做模式匹配。
    """

    def __init__(self):
        self._call_count = 0

    def generate(self, messages: list[dict[str, str]]) -> str:
        self._call_count += 1
        system = ""
        user = ""
        for m in messages:
            if m["role"] == "system":
                system += m.get("content", "") + "\n"
            elif m["role"] == "user":
                user += m.get("content", "") + "\n"
            elif m["role"] == "tool":
                # Function Calling 的 tool 结果 → 转成 Observation 格式
                user += f"Observation: {m.get('content', '')}\n"
            elif m["role"] == "assistant":
                # 跳过 assistant 的 tool_calls 消息（Mock 不需要）
                content = m.get("content", "")
                if content:
                    user += content + "\n"

        full = system + user

        # 路由到不同的响应生成器
        if "ReAct" in system:
            return self._react(full, user)
        elif "Planner" in system:
            return self._planner(user)
        elif "Executor" in system:
            return self._executor(full, user)
        elif "Replanner" in system:
            return self._replanner(user)
        elif "aggregator" in system.lower():
            return self._aggregate(user)
        else:
            return self._default(user)

    def generate_with_tools(
        self, messages: list[dict[str, str]], tools: list[dict],
        tool_choice: str | dict = "auto",
    ) -> dict:
        """
        Mock 版 Function Calling —— 把 generate() 的文本输出转成结构化 dict。

        真实 API (OpenAILLM) 的做法: 传 tools 参数，API 返回 tool_calls JSON
        MockLLM 的做法:   复用现有文本生成 → 检测 Action: 模式 → 转成 tool_call dict

        这演示了 Function Calling 的返回格式，但不依赖真实 API。
        """
        text = self.generate(messages)

        # 检测 ReAct 格式的工具调用: "Action: calculator\nAction Input: ..."
        action_match = re.search(r"Action:\s*(\S+?)\s*$", text, re.MULTILINE | re.IGNORECASE)
        input_match = re.search(
            r"Action Input:\s*(.+?)(?=\n(?:Thought|Action|Observation)|\Z)",
            text, re.DOTALL | re.IGNORECASE,
        )

        if action_match and action_match.group(1).lower() != "finish":
            tool_name = action_match.group(1).strip()
            raw_input = input_match.group(1).strip() if input_match else ""

            # 构造 Function Calling 风格的 arguments dict
            # 从 TOOL_REGISTRY 查这个工具的参数名
            from .tools import TOOL_REGISTRY
            tool_info = TOOL_REGISTRY.get(tool_name, {})
            param_names = list(tool_info.get("parameters", {}).get("properties", {}).keys())

            if param_names:
                arguments = {param_names[0]: raw_input}
            else:
                arguments = {"input": raw_input}

            return {
                "type": "tool_call",
                "calls": [{"id": f"mock_{self._call_count}", "name": tool_name, "arguments": arguments}],
            }

        # 检测 Plan-Execute 格式: "TOOL: name | INPUT: args"
        tool_match = re.search(r"TOOL:\s*(\S+)\s*\|\s*INPUT:\s*(.+?)$", text, re.DOTALL)
        if tool_match:
            tool_name = tool_match.group(1).strip()
            raw_input = tool_match.group(2).strip()
            return {
                "type": "tool_call",
                "calls": [{"id": f"mock_{self._call_count}", "name": tool_name, "arguments": {"input": raw_input}}],
            }

        return {"type": "text", "content": text}

    # ── ReAct ────────────────────────────────────────────────

    def _react(self, full: str, user: str) -> str:
        # Only count actual Observation: lines (start of line), not quoted references in prompt
        obs_count = len(re.findall(r"^Observation:", full, re.MULTILINE))
        task = user

        if obs_count == 0:
            return self._react_step1(task)
        elif obs_count == 1:
            return self._react_step2(full, task)
        else:
            return self._react_final(full)

    def _react_step1(self, task: str) -> str:
        """ReAct 第一步: 分析问题并调用第一个工具"""
        if any(w in task for w in ["买", "剩下", "元", "本书", "支笔"]):
            return (
                "Thought: 这是一个购物找零问题。已知总金额120元、买3本书后剩45元，"
                "可以先算出3本书的总价。\n"
                "Action: calculator\n"
                "Action Input: (120 - 45) / 3"
            )
        elif "GDP" in task or "对比" in task:
            return (
                "Thought: 需要搜索中美GDP数据。先搜索中国的。\n"
                "Action: search\n"
                "Action Input: 中国 GDP 2024"
            )
        elif "三角形" in task or "斜边" in task:
            return (
                "Thought: 这是一个直角三角形问题，需要用到勾股定理。先查一下公式。\n"
                "Action: knowledge_base\n"
                "Action Input: 勾股定理"
            )
        elif "半径" in task or "面积" in task or "圆形" in task:
            return (
                "Thought: 这个几何问题需要圆的面积公式。先查询知识库。\n"
                "Action: knowledge_base\n"
                "Action Input: 圆的面积公式"
            )
        elif "光" in task and ("时间" in task or "距离" in task):
            return (
                "Thought: 需要确认光速和日地距离的准确数值。\n"
                "Action: search\n"
                "Action Input: 光速 地球 太阳 距离"
            )
        elif "马拉松" in task:
            return (
                "Thought: 先确认马拉松距离和地球周长，然后计算。\n"
                "Action: search\n"
                "Action Input: 地球赤道周长"
            )
        else:
            expr = self._extract_math(task)
            return (
                f"Thought: 让我先计算第一步。\n"
                f"Action: calculator\n"
                f"Action Input: {expr}"
            )

    def _react_step2(self, full: str, task: str) -> str:
        """ReAct 第二步: 基于第一个 Observation 继续"""
        # 提取最后一个 Observation
        obs_matches = re.findall(r"^Observation:\s*(.+?)(?=\n(?:Thought|Action|Observation)|\Z)", full, re.DOTALL | re.MULTILINE)
        last_obs = obs_matches[-1].strip() if obs_matches else ""

        if "买" in task and "笔" in task and "支" in task:
            # shopping 问题: 第一步算出了书价=25，现在算笔价
            return (
                f"Thought: 上一步算出了每本书的价格。现在算每支笔的价格："
                f"每支笔比每本书便宜8元。\n"
                f"Action: calculator\n"
                f"Action Input: 25 - 8"
            )
        elif "剩下" in task and "元" in task:
            # 最后一步: 算能买几支
            return (
                f"Thought: 现在知道剩余45元，每支笔17元，可以算能买几支了。\n"
                f"Action: calculator\n"
                f"Action Input: 45 / 17"
            )
        elif "GDP" in task:
            return (
                f"Thought: 搜到了中国GDP数据。现在搜索美国的。\n"
                f"Action: search\n"
                f"Action Input: 美国 GDP 2024"
            )
        elif "三角形" in task:
            return (
                f"Thought: 知道了勾股定理 a²+b²=c²。现在计算斜边。\n"
                f"Action: calculator\n"
                f"Action Input: sqrt(3^2 + 4^2)"
            )
        elif "半径" in task:
            return (
                f"Thought: 有了圆面积公式 S=πr²。计算外圆(半径6米)和内圆(半径5米)的面积。\n"
                f"Action: calculator\n"
                f"Action Input: 3.14 * 6^2 - 3.14 * 5^2"
            )
        elif "光" in task:
            return (
                f"Thought: 有了日地距离和光速数据。现在算时间: 距离/速度。\n"
                f"Action: calculator\n"
                f"Action Input: (1.5 * 10^8) / (3 * 10^5) / 60"
            )
        elif "马拉松" in task:
            return (
                f"Thought: 知道了地球周长约40075公里。计算需要多少个马拉松。\n"
                f"Action: calculator\n"
                f"Action Input: 40075 / 42.195"
            )
        else:
            expr = self._extract_math(last_obs)
            return (
                f"Thought: 基于上一步结果，继续计算。\n"
                f"Action: calculator\n"
                f"Action Input: {expr}"
            )

    def _react_final(self, full: str) -> str:
        """ReAct 第三步: 综合所有 Observation，给出最终答案"""
        obs_matches = re.findall(r"^Observation:\s*(.+?)(?=\n(?:Thought|Action|Observation)|\Z)", full, re.DOTALL | re.MULTILINE)
        obs_text = " | ".join(o.strip()[:60] for o in obs_matches[-2:])
        return (
            f"Thought: 我已经收集了所有需要的信息和计算结果，"
            f"可以给出最终答案了。\n"
            f"Action: Finish\n"
            f"Action Input: 根据逐步推理和计算:\n"
            f"  {obs_text}\n"
            f"  综合以上结果，最终答案为上述计算所得。"
        )

    # ── Planner ──────────────────────────────────────────────

    def _planner(self, user: str) -> str:
        task = user

        if "买" in task and ("笔" in task or "书" in task):
            return (
                "1. 计算3本书的总价: 用总金额减去剩余金额\n"
                "2. 计算每本书的单价: 书总价除以3\n"
                "3. 计算每支笔的单价: 书单价减去8元\n"
                "4. 计算可买笔的数量: 剩余金额除以笔单价\n"
                "5. 汇总结果，验证总花费不超过120元"
            )
        elif "三角形" in task and ("直角" in task or "斜边" in task):
            return (
                "1. 查询勾股定理公式\n"
                "2. 将两直角边代入公式，计算斜边长度\n"
                "3. 用三角形面积公式计算面积\n"
                "4. 将三边相加计算周长\n"
                "5. 汇总三个答案"
            )
        elif "半径" in task or ("圆" in task and "面积" in task):
            return (
                "1. 查询圆的面积公式\n"
                "2. 计算外圆(花坛+小路)面积: 外半径=5+1=6米\n"
                "3. 计算内圆(花坛)面积: 内半径=5米\n"
                "4. 计算小路面积: 外圆面积减内圆面积\n"
                "5. 汇总结果"
            )
        elif "GDP" in task or "对比" in task:
            return (
                "1. 搜索中国2024年GDP数据\n"
                "2. 搜索美国2024年GDP数据\n"
                "3. 统一单位，按汇率换算为同一货币\n"
                "4. 计算中国GDP占美国GDP的百分比\n"
                "5. 整理对比结果并给出结论"
            )
        elif "光" in task and "太阳" in task:
            return (
                "1. 确认日地距离(公里)和光速(m/s)的数值\n"
                "2. 统一单位: 将日地距离从公里转换为米\n"
                "3. 计算时间: t = 距离 / 光速，得到秒数\n"
                "4. 将秒转换为分钟\n"
                "5. 汇总并验证结果"
            )
        elif "马拉松" in task:
            return (
                "1. 确认地球赤道周长(公里)\n"
                "2. 计算需要跑多少个马拉松: 地球周长 / 42.195\n"
                "3. 计算需要多少年: 马拉松个数 / 365\n"
                "4. 汇总两个答案"
            )
        else:
            return (
                "1. 理解问题，提取已知量和未知量\n"
                "2. 确定解题方法和需要的公式/工具\n"
                "3. 按依赖关系逐步执行计算\n"
                "4. 汇总中间结果得出最终答案\n"
                "5. 验证答案的合理性"
            )

    # ── Executor ─────────────────────────────────────────────

    def _executor(self, full: str, user: str) -> str:
        """执行单个步骤，可能需要调用工具"""
        # 提取当前 step 的描述
        step_match = re.search(r'Your task: Execute step \d+\s*\n(.+)', user, re.DOTALL)
        step_desc = step_match.group(1).strip() if step_match else user

        # 匹配具体操作
        if re.search(r"(\d+)\s*/\s*(\d+)", step_desc):
            # 包含具体数字计算
            expr = self._extract_math(step_desc)
            return f"TOOL: calculator | INPUT: {expr}"
        elif re.search(r"(\d+)\s*\-\s*(\d+)", step_desc):
            expr = self._extract_math(step_desc)
            return f"TOOL: calculator | INPUT: {expr}"
        elif re.search(r"(\d+)\s*\*\s*(\d+)", step_desc) or "平方" in step_desc or "^" in step_desc:
            expr = self._extract_math(step_desc)
            return f"TOOL: calculator | INPUT: {expr}"
        elif any(w in step_desc for w in ["搜索", "查询", "确认"]):
            keyword = step_desc.split("。")[0][:60]
            return f"TOOL: search | INPUT: {keyword}"
        elif any(w in step_desc for w in ["公式", "定理", "定律", "知识"]):
            keyword = step_desc[:50]
            return f"TOOL: knowledge_base | INPUT: {keyword}"
        elif any(w in step_desc for w in ["验证", "汇总", "整理", "给出", "总结"]):
            return step_desc.split("。")[0] + " — 该步骤已完成。"
        else:
            expr = self._extract_math(step_desc)
            if expr != "0":
                return f"TOOL: calculator | INPUT: {expr}"
            return f"执行完成: {step_desc[:100]}"

    # ── Replanner ────────────────────────────────────────────

    def _replanner(self, user: str) -> str:
        """判断是否继续/replan/结束"""
        # 统计已完成步数
        completed = user.count("Step")
        remaining_text = ""
        rem_match = re.search(r'Remaining steps:\s*\n(.+)', user, re.DOTALL)
        if rem_match:
            remaining_text = rem_match.group(1).strip()
        remaining_count = len(re.findall(r'\d+\.', remaining_text)) if remaining_text else 0

        if remaining_count == 0:
            return "FINISH"
        elif any(w in user for w in ["错误", "不对", "矛盾"]):
            return "REPLAN: 需要根据当前结果调整剩余步骤。"
        else:
            return "CONTINUE"

    # ── Aggregator ───────────────────────────────────────────

    def _aggregate(self, user: str) -> str:
        """汇总各步骤结果"""
        # 提取步骤结果
        results = re.findall(r'Result:\s*(.+?)(?=\nStep|\Z)', user, re.DOTALL)
        results_text = "; ".join(r.strip()[:80] for r in results)
        return (
            f"综合所有步骤的计算结果:\n"
            f"{results_text}\n\n"
            f"以上为完整的最终答案。"
        )

    # ── Default ──────────────────────────────────────────────

    def _default(self, user: str) -> str:
        # 如果 user 中包含"汇总"等关键词，视为aggregation
        if any(w in user for w in ["汇总", "synthesize", "最终答案", "Synthesize"]):
            return self._aggregate(user)
        return f"请指定 Agent 角色模式（ReAct / Planner / Executor / Replanner）。"

    # ── 工具方法 ────────────────────────────────────────────

    def _extract_math(self, text: str) -> str:
        """从文本中提取最可能的数学表达式"""
        patterns = [
            r'(\d+\.?\d*\s*[\+\-\*/]\s*\d+\.?\d*(?:\s*[\+\-\*/]\s*\d+\.?\d*)*)',
            r'(\d+\.?\d*\s*[\+\-\*/]\s*\d+\.?\d*)',
            r'(\w+\s*\([^)]+\))',  # sqrt(...) 等函数调用
        ]
        for pat in patterns:
            matches = re.findall(pat, text)
            if matches:
                return matches[0].strip()
        # 找单独的数字
        nums = re.findall(r'\d+\.?\d*', text)
        return nums[0] if nums else "0"


# ═══════════════════════════════════════════════════════════════
#  OpenAILLM — 真实 API 调用
# ═══════════════════════════════════════════════════════════════

class OpenAILLM(LLMBase):
    """
    OpenAI 兼容接口封装。

    支持所有 OpenAI 兼容的 API:
    - OpenAI 官方:        base_url="https://api.openai.com/v1"
    - DeepSeek:          base_url="https://api.deepseek.com/v1"
    - 阿里 Qwen (百炼):   base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    - 智谱 GLM:           base_url="https://open.bigmodel.cn/api/paas/v4"
    - Moonshot/Kimi:     base_url="https://api.moonshot.cn/v1"
    - Ollama 本地:        base_url="http://localhost:11434/v1"
    - vLLM 本地:         base_url="http://localhost:8000/v1"
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        base_url: str | None = None,
        temperature: float = 0.0,
        max_retries: int = 3,
        timeout: int = 120,
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.max_retries = max_retries
        self.timeout = timeout
        self._client = None

    @property
    def client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError(
                    "需要安装 openai 包: pip install openai\n"
                    "然后设置 API key: export OPENAI_API_KEY=sk-xxx"
                )
            kwargs = {"api_key": self.api_key, "timeout": self.timeout, "max_retries": 0}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = OpenAI(**kwargs)
        return self._client

    def generate(self, messages: list[dict[str, str]]) -> str:
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                )
                content = response.choices[0].message.content
                return content or ""

            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    wait = 2 ** attempt  # 1s, 2s, 4s...
                    time.sleep(wait)
                else:
                    raise RuntimeError(
                        f"LLM 调用失败（重试 {self.max_retries} 次后）: {last_error}\n"
                        f"请检查: 1) API key 是否正确  2) base_url 是否正确  3) 网络连接"
                    )

    def generate_with_tools(
        self, messages: list[dict[str, str]], tools: list[dict],
        tool_choice: str | dict = "auto",
    ) -> dict:
        """
        带 Function Calling 的 LLM 请求（阶段 2 核心方法）。

        与 generate() 的本质区别:
        ┌──────────────────┬────────────────────────────────────┐
        │  generate()       │  返回纯文本                        │
        │                   │  Agent 用正则掏 Thought/Action    │
        │                   │  格式错一个字符就解析失败          │
        ├──────────────────┼────────────────────────────────────┤
        │  generate_with_   │  API 参数传 tools=[{...}]         │
        │  tools()          │  LLM 原生返回 tool_calls JSON     │
        │                   │  字段名由 API 保证，不需要正则     │
        └──────────────────┴────────────────────────────────────┘

        返回格式:
          {"type": "text", "content": "直接回答的文本"}
          {"type": "tool_call", "calls": [
              {"name": "calculator", "arguments": {"expression": "(120-45)/3"}},
              ...
          ]}

        参数:
          messages: 对话历史
          tools:    get_openai_tools() 返回的工具 schema 列表
        """
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=tools,
                    tool_choice=tool_choice,
                    temperature=self.temperature,
                )
                msg = response.choices[0].message

                # 情况 1: LLM 想调工具
                if msg.tool_calls:
                    calls = []
                    for tc in msg.tool_calls:
                        try:
                            args = json.loads(tc.function.arguments)
                        except json.JSONDecodeError:
                            args = {}
                        calls.append({
                            "id": tc.id,
                            "name": tc.function.name,
                            "arguments": args,
                        })
                    return {"type": "tool_call", "calls": calls}

                # 情况 2: LLM 直接文本回复
                return {"type": "text", "content": msg.content or ""}

            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    wait = 2 ** attempt
                    time.sleep(wait)
                else:
                    raise RuntimeError(
                        f"LLM (tools) 调用失败（重试 {self.max_retries} 次后）: {last_error}"
                    )


# ═══════════════════════════════════════════════════════════════
#  工厂函数
# ═══════════════════════════════════════════════════════════════

def create_llm(backend: str = "mock", **kwargs) -> LLMBase:
    """
    创建 LLM 实例。

    Args:
        backend: "mock" 或 "openai"
        **kwargs: 传递给具体实现 (api_key, model, base_url 等)

    Examples:
        # 模拟模式（学习用）
        llm = create_llm("mock")

        # OpenAI 官方
        llm = create_llm("openai", api_key="sk-xxx", model="gpt-4o")

        # DeepSeek
        llm = create_llm("openai", api_key="sk-xxx",
                         model="deepseek-chat",
                         base_url="https://api.deepseek.com/v1")

        # 从环境变量读取
        llm = create_llm("openai",
                         api_key=os.environ["OPENAI_API_KEY"],
                         model=os.environ.get("LLM_MODEL", "gpt-4o"),
                         base_url=os.environ.get("LLM_BASE_URL"))
    """
    if backend == "mock":
        return MockLLM()
    elif backend == "openai":
        return OpenAILLM(**kwargs)
    else:
        raise ValueError(f"不支持的 LLM backend: {backend}")
