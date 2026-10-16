# NL2SQL Agent

> 自然语言查询数据库 —— 将数据获取周期从**小时级压缩到秒级**。

## 快速开始

```bash
pip install openai gradio numpy

# 在仓库根目录创建 .env（模板见仓库根的 .env.example），填入 NL2SQL_API_KEY /
# NL2SQL_BASE_URL / NL2SQL_MODEL。也可以直接写 OPENAI_API_KEY / LLM_MODEL /
# LLM_BASE_URL —— 加载时会自动映射到 NL2SQL_* 命名空间。
cp ../../.env.example ../../.env

python data/generate_data.py    # 生成测试数据
python main.py --question "北京用户买得最多的品类是什么？"
python app.py                   # Web UI → http://localhost:7860
```

## 分享给他人

```bash
# 1. 编辑 ngrok.yml 填入 authtoken (https://ngrok.com 免费注册)
# 2. 双击 start.bat  →  弹出 ngrok 窗口显示公网链接
# 3. 把 https://xxxx.ngrok-free.app 发给朋友
```

## 架构

```
Plan-informed Self-Correction Agent (ReAct 专化变体)

Plan 阶段:           Self-Correction:        兜底:
难度分级              LLM生成SQL ──┐           Plan-Execute
Schema检索               ↓        │           ├── CTE链 (顺序型)
Few-shot选择          预校验 ← 纠错增强        └── 子任务 (独立型)
Prompt构造             执行 → 停止判断
```

## CLI 用法

```bash
python main.py --question "..."              # 单次查询
python main.py --interactive                 # 交互模式
python main.py --mode auto|simple|plan       # 执行模式
python main.py --schema full|keyword|hybrid  # Schema 策略
python main.py --list-sources                # 数据源列表
```

## 评测

```bash
python eval.py                    # 25条全量
python eval.py --verbose          # 详细模式 (SQL+结果+总结)
python eval.py --difficulty hard  # 按难度筛选
```

**成绩: 25 条 100% 执行成功, 平均 3s/条, ~2,500 tokens/条。**

## 项目结构

```
nl2sql_agent/
├── interfaces/              # 9个核心接口 (全部可插拔)
├── agent/                   # Agent 循环 + Prompt + Few-shot
├── tools/                   # Function Calling 工具
├── core/                    # LLM 抽象层
├── data/                    # 数据库 + 评测用例 + 数据源注册
├── config.py / schemas.py
├── main.py                  # CLI 入口
├── eval.py                  # 批量评测
├── app.py                   # Web UI (Gradio)
├── start.bat                # 一键启动 (Agent + ngrok)
├── REPORT.md                # 技术报告
└── README.md                # 本文件
```

## 技术栈

- **LLM**: DeepSeek API (兼容 OpenAI 格式)
- **数据库**: SQLite (架构支持 MySQL/PG 扩展)
- **Web UI**: Gradio
- **分享**: ngrok
- **评测**: 25 条自动化用例
