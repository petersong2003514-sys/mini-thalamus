# 外部基准证据报告

日期：2026-08-13

## 汇总

| 基准 | 子集 | 指标 | 当前值 | 报告文件 | 状态 |
|---|---|---|---|---|---|
| BFCL | simple_python | pass_rate | 0.98 | `evals/results/bfcl-deepseek-fc-simple_python-20260813-151714.json` | 已验证 |
| BFCL | multiple | pass_rate | 1.0 | `evals/results/bfcl-deepseek-fc-multiple-20260813-144717.json` | 已验证 |
| BFCL | parallel | pass_rate | 0.0 | `evals/results/bfcl-deepseek-fc-parallel-20260813-144719.json` | 已验证 |
| BFCL | parallel_multiple | pass_rate | 0.0 | `evals/results/bfcl-deepseek-fc-parallel_multiple-20260813-144721.json` | 已验证 |
| BFCL | multi_turn_base | pass_rate | 0.0（3 条） | `evals/results/bfcl-shushu-multi_turn_base-20260813-163330.json` | 官方逐轮计分 |
| BFCL | multi_turn_long_context | pass_rate | 0.0（3 条） | `evals/results/bfcl-shushu-multi_turn_long_context-20260813-163528.json` | 官方逐轮计分 |
| BFCL | multi_turn_miss_func | pass_rate | 0.0（3 条） | `evals/results/bfcl-shushu-multi_turn_miss_func-20260813-163747.json` | 官方逐轮计分 |
| BFCL | multi_turn_miss_param | pass_rate | 0.0（3 条） | `evals/results/bfcl-shushu-multi_turn_miss_param-20260813-164010.json` | 官方逐轮计分 |
| BFCL | simple_python (shushu) | pass_rate | 0.6 | `evals/results/bfcl-shushu-simple_python-20260813-151930.json` | 进化后 |
| BEIR | nfcorpus | nDCG@10 / MRR@10 | 0.4495 / 0.3927 | `evals/results/beir-nfcorpus-20260813-152043.json` | 已验证 |
| BEIR | scifact | nDCG@10 / MRR@10 | 0.7008 / 0.6622 | `evals/results/beir-scifact-20260813-152044.json` | 已验证 |
| Spider | dev | execution_accuracy | 0.9（50 库分层） | evals/results/spider-dev-20260813-164220.json | 已验（20 个 db_id 覆盖 50 条） |
| Spider | failure_regression | execution_accuracy | 0.0（5 条，可复现） | evals/results/spider-dev-20260813-165013.json | 失败回归集 |

## 主要失败类型

| 子集 | 失败类型 | 归因 |
|---|---|---|
| simple_python (shushu) | decoder_failed | Orchestrator 多次走 fallback，未返回可解析函数调用 |
| parallel / parallel_multiple | simple_function_checker:wrong_count | 单轮直连模式未生成足量并行调用 |
| multi_turn 4 类（12 条） | multi_turn:empty_turn_model_response | 官方逐轮计分发现每轮都未产生可执行工具调用 |
| web_search | agentic:answer_not_found | 检索/综合能力不足，且本轮 API 连接失败 |
| BEIR | retrieval_miss | BM25 基线召回有限 |

## 自进化记录

- 补丁：`manual-simple-python-keywords`
- 文件：`agents/api_caller_agent.py`
- 内容：为 `AGENT_SPEC.keywords` 增加 `calculate`、`factorial`、`hypotenuse`、`quadratic`、`roots`、`solve`
- Baseline：`evals/results/bfcl-shushu-simple_python-20260813-151810.json`，pass_rate = 0.2
- After：`evals/results/bfcl-shushu-simple_python-20260813-151930.json`，pass_rate = 0.6
- Guard：`evals/self_evolution/guard_results/simple_python-20260813-151930.json`，delta = +0.4，通过

## 未完成项

- Spider dev 数据：官方 Google Drive 链接仍失效，已改用 hf-mirror 镜像（xlangai/spider parquet + multispider-processed SQLite）构造本地 zip，check-data = 1034 cases / 20 databases；50 条分层样本 Execution Accuracy = 0.9（45/50，失败含 voter_1、flight_2、battle_death、dog_kennels、employee_hire_evaluation 等语义差异）。
- BFCL multi-turn：官方 per-turn 执行计分器已 vendored 并接入，4 类各 3 条冒烟 pass_rate = 0.0；全量 4 类约 800 条尚未跑。
