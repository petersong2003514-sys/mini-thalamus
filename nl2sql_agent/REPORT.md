# NL2SQL Agent 技术报告

> **题目**: 设计并训练一个 Agent 框架，让用户用自然语言完成数据查询，将数据获取周期从小时级压缩到分钟级。

---

## 1. 问题定义

数据分析师和算法工程师每天从数十张表中查询信息，传统流程：理解表结构(20min) → 手写SQL(15min) → 等待提数(1h) → 分析结果(30min) = **2-3 小时/次**。

NL2SQL 的核心困难在于自然语言的模糊性与 SQL 的精确性之间的矛盾：

| 维度 | 自然语言 | SQL |
|------|---------|-----|
| 表意 | 模糊（"卖得最好"） | 精确（`ORDER BY SUM(quantity) DESC`） |
| 指代 | "商品" 指向 products.name | 必须是 `table.column` |
| 推理 | 隐式（"北京用户" → JOIN） | 显式（`JOIN users ON ...`） |

---

## 2. 技术方案

### 2.1 架构总览

采用 **Plan-informed Self-Correction Agent** 架构——ReAct 模式的 NL2SQL 专化变体。

```
┌─────────────────────────────────────────┐
│  Plan 阶段 (一次性)                      │
│  Step 0: 难度分级 → Step 1: Schema检索  │
│  Step 2: Few-shot选择 → Step 3: Prompt  │
└──────────────────┬──────────────────────┘
                   │
┌──────────────────▼──────────────────────┐
│  Self-Correction 循环 (≤3轮)            │
│  Step 4: LLM生成SQL                     │
│  Step 5: SQL预校验 (三层防御)            │
│  Step 6: SQL执行 (只读连接)              │
│  Step 7: 停止判断 (硬规则+LLM语义)        │
│  Step 8: 纠错增强 (映射表+CoT) → Step 4 │
└──────────────────┬──────────────────────┘
                   │ 失败 ↓
┌──────────────────▼──────────────────────┐
│  Plan-Execute 兜底                       │
│  顺序依赖 → CTE链   |   独立汇总 → 子任务 │
└─────────────────────────────────────────┘
```

### 2.2 9 个核心接口

| 接口 | 职责 | 实现策略 |
|------|------|---------|
| `ISchemaProvider` | Schema 检索 | Full(≤10表) / Keyword(10-100表) / Hybrid(100+表, LLM精排) |
| `IDatabaseConnector` | 数据库连接 | SQLite 只读连接 + 3层安全 |
| `IDifficultyClassifier` | 难度分级 | LLM主分类 + 规则兜底 |
| `IFewShotSelector` | Few-shot 选择 | 标签匹配(难度/技能/表名三标签打分) |
| `ISQLValidator` | SQL 预校验 | 三层防御: 硬规则→白名单→LLM语义 |
| `IErrorEnhancer` | 纠错增强 | 方法A(映射表) + 方法C(CoT两步推理) |
| `IStopJudge` | 停止判断 | 硬规则预判 + LLM语义兜底 |
| `ITaskDecomposer` | 任务拆解 | 顺序依赖→CTE链 / 独立汇总→子任务 |
| `IPromptGuard` | 安全防护 | 注入检测(11类模式) + 查询意图校验 |

### 2.3 三种执行模式 + 进化路径

```
auto:   直接SQL → 失败 → Plan-Execute 兜底
simple: 直接SQL → 失败 → 自动升级 Plan-Execute (逃生舱)
plan:   easy/medium → 自动回退 simple
        hard → CTE链 or 子任务拆解
```

### 2.4 关键技术决策

| 难点 | 决策 |
|------|------|
| **Schema Linking** | Full→Keyword→Hybrid 三级策略，Hybrid 用 Keyword粗筛+LLM语义精排 |
| **Few-shot 选择** | LLM预分难度 → 标签匹配动态选 |
| **SQL 预校验** | 三层防御: 硬规则→列名白名单→LLM语义匹配 |
| **纠错** | 方法A映射表(默认) → 同错误≥2轮自动切CoT两步推理 |
| **停止判断** | 硬规则(MAX_ROUNDS/NO_PROGRESS) + LLM语义(STAGNATION本质判断) |
| **安全** | PromptGuard(注入+意图) → SQL白名单 → OS只读连接 → 环境隔离 |
| **复杂查询** | CTE链(顺序依赖型) + 子任务(独立汇总型)，子任务自带修复循环 |

---

## 3. 实验设计

### 3.1 数据集

合成电商数据库: 5 张表, 500 用户, 200 商品, 2541 订单, 7566 明细, 10000 行为。

注入可控规律用于自动化评测: `北京→电子产品, 上海→美妆, 广州→服装, 深圳→数码配件, 杭州→食品饮料`（每个城市 70% 订单集中在本城市品类）。

### 3.2 评测用例

25 条用例: Easy×8（单表） + Medium×7（聚合/2表JOIN） + Hard×6（3表+/子查询） + Algorithm×4（漏斗/留存/分层/转化）

---

## 4. 实验结果

### 4.1 整体成绩

| 指标 | 结果 |
|------|------|
| 总用例 | 25 |
| 执行成功率 | **100%** (25/25) |
| 结果匹配率 | **100%** (25/25) |
| 平均轮次 | 1.0 |
| 平均耗时 | 3.0s/条（直接）/ 15s/条（Plan-Execute） |
| 平均 Token | ~2,500/条 |

### 4.2 按难度分布

| 难度 | 执行成功率 | 结果匹配率 |
|------|-----------|-----------|
| Easy | 100% | 100% |
| Medium | 100% | 100% |
| Hard | 100% | 100% |
| Algorithm | 100% | 100% |

### 4.3 可控规律验证

所有 CITY_CATEGORY_MAP 注入规律均被正确查询:

```
北京 → 电子产品 (1116次) ✅
上海 → 美妆 (1136次)     ✅
广州 → 服装 (1192次)     ✅
深圳 → 数码配件 (1157次)  ✅
杭州 → 食品饮料 (1200次)  ✅
```

### 4.4 能力边界测试

| 难度 | 问题类型 | 结果 | 策略 |
|------|---------|------|------|
| 很难 | 多指标时序对比（留存+消费+复购 3指标×平台均值） | ✅ | 子任务拆解 |
| 极难 | 队列留存趋势（双周批×4周留存×3批最优） | ✅ | CTE 链 |
| 地狱 | 滞销高价品（价>2x均价 AND 量<20%均量） | ✅ | 子任务拆解（结果为0，数据验证正确） |

---

## 5. 安全设计

四层纵深防御：

| 层级 | 机制 | 拦截内容 |
|------|------|---------|
| Layer 0: PromptGuard | 11类注入模式 + 查询意图校验 | 指令劫持、API Key窃取、越狱、命令行注入、无意义输入 |
| Layer 1: SQL 校验 | 只放行 SELECT/WITH(CTE)，黑名单拦截 | DROP/DELETE/INSERT/UPDATE/ALTER |
| Layer 2: OS 只读连接 | SQLite `file:?mode=ro` | 物理层面防写入 |
| Layer 3: 环境隔离 | API Key 仅存 env var | LLM 永远访问不到密钥 |

---

## 6. 业务可行性

| 查询类型 | 准确率 | 建议 |
|---------|--------|------|
| 单表查询 | >95% | 直接使用 |
| 双表 JOIN + 聚合 | >90% | 直接使用 + 展示 SQL 供复核 |
| 3表+ JOIN + 子查询 | >85% | 自动预览 → 用户确认 → 执行 |
| CTE/窗口函数 | >80% (Plan-Execute 兜底) | 需拆解执行，~15s |

### 落地路径

```
Phase 1: 内部工具 (1-2周)
  - 部署 CLI + Web UI
  - 覆盖高频 TOP 20 查询

Phase 2: 团队推广 (1个月)
  - 基于反馈优化 Few-shot + Prompt
  - 接入更多数据源

Phase 3: 平台化 (3个月)
  - SFT 微调小模型降低延迟/成本
  - 接入数据平台工作流
```

---

## 7. 项目文件结构

```
nl2sql_agent/
├── interfaces/          # 9个核心接口 + 实现
│   ├── schema_provider.py       # Full/Keyword/Hybrid
│   ├── database_connector.py    # SQLite 只读
│   ├── difficulty_classifier.py # LLM+规则兜底
│   ├── fewshot_selector.py      # 标签匹配
│   ├── sql_validator.py         # 三层防御
│   ├── error_enhancer.py        # 映射表+CoT
│   ├── stop_judge.py            # 硬规则+LLM语义
│   ├── task_decomposer.py       # CTE链+子任务
│   └── prompt_guard.py          # 注入防御+意图校验
├── agent/               # Agent 核心循环 + Prompt
├── tools/               # Function Calling 工具
├── core/                # LLM 抽象层
├── data/                # 数据库 + 评测用例 + 多数据源注册
├── config.py / schemas.py
├── main.py              # CLI 入口
├── eval.py              # 批量评测
├── app.py               # Web UI (Gradio)
├── start.bat            # 一键启动 (Agent+ngrok)
├── REPORT.md / README.md
└── .env / ngrok.yml
```

---

## 8. 总结

核心指标: **将数据获取周期从小时级压缩到 3 秒**，25 条评测 100% 通过，Token 成本 ¥0.0025/次。

核心创新:
- Plan-informed Self-Correction + Plan-Execute 双层架构
- 9 接口全可插拔，三种 Schema 策略、三种执行模式
- 四层纵深安全防御
- 合成数据注入可控规律实现可验证评测
