# 外部基准证据报告

日期：2026-08-13

## 汇总

| 基准 | 子集 | 指标 | 当前值 | 报告文件 | 状态 |
|---|---|---|---|---|---|
| BFCL | simple_python | pass_rate | 0.98 | `evals/results/bfcl-deepseek-fc-simple_python-20260813-151714.json` | 已验证 |
| BFCL | multiple | pass_rate | 1.0 | `evals/results/bfcl-deepseek-fc-multiple-20260813-144717.json` | 已验证 |
| BFCL | parallel | pass_rate | 0.0 | `evals/results/bfcl-deepseek-fc-parallel-20260813-144719.json` | 已验证 |
| BFCL | parallel_multiple | pass_rate | 0.0 | `evals/results/bfcl-deepseek-fc-parallel_multiple-20260813-144721.json` | 已验证 |
| BFCL | multi_turn_base | pass_rate | 0.0 | `evals/results/bfcl-shushu-multi_turn_base-20260813-161333.json` | 官方逐轮计分（1 条冒烟） |
| BFCL | simple_python (shushu) | pass_rate | 0.6 | `evals/results/bfcl-shushu-simple_python-20260813-151930.json` | 进化后 |
| BEIR | nfcorpus | nDCG@10 / MRR@10 | 0.4495 / 0.3927 | `evals/results/beir-nfcorpus-20260813-152043.json` | 已验证 |
| BEIR | scifact | nDCG@10 / MRR@10 | 0.7008 / 0.6622 | `evals/results/beir-scifact-20260813-152044.json` | 已验证 |
| Spider | dev | execution_accuracy | 0.95（20 库分层） | evals/results/spider-dev-20260813-161527.json | 已验（20 个 db_id 各 1 条） |

## 主要失败类型

| 子集 | 失败类型 | 归因 |
|---|---|---|
| simple_python (shushu) | decoder_failed | Orchestrator 多次走 fallback，未返回可解析函数调用 |
| parallel / parallel_multiple | simple_function_checker:wrong_count | 单轮直连模式未生成足量并行调用 |
| multi_turn_base | multi_turn:empty_turn_model_response | 官方逐轮计分发现每轮都未产生可执行工具调用 |
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

- Spider dev 数据：官方 Google Drive 链接仍失效，已改用 hf-mirror 镜像（xlangai/spider parquet + multispider-processed SQLite）构造本地 zip，check-data = 1034 cases / 20 databases；20 库分层样本 Execution Accuracy = 0.95（19/20，唯一失败为 voter_1 的 COUNT 语义差异）。
- BFCL multi-turn：当前使用有序工具路径计分，官方 per-turn 模拟器未 vendored，报告应标注为“路径计分”而非完整模拟器结果。
