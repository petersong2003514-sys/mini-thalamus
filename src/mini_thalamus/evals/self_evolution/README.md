# 自进化闭环

自进化信号全部来自外部权威基准（BFCL / BEIR / Spider），不使用自建
`capability_cases.jsonl` / `router_cases.jsonl` 作为进化依据。

## 一轮闭环

```powershell
python evals/bfcl_eval.py --category web_search --limit 5
python evals/self_evolution/collect.py --benchmark bfcl
python evals/self_evolution/feedback.py --collected evals/self_evolution/reports/collect-<timestamp>.json
python evals/self_evolution/apply.py --patches evals/self_evolution/patches/feedback-<timestamp>.json
```

`apply.py` 默认 dry-run，只打印 diff。确认后加 `--apply` 写入
`AGENT_SPEC.keywords` / `few_shots`。

重跑同一子集后执行门禁：

```powershell
python evals/self_evolution/guard.py --before <旧报告.json> --after <新报告.json> --threshold 0
```

门禁按基准取主指标：

- BFCL：`pass_rate`
- BEIR：`ndcg_at_10`
- Spider：`execution_accuracy`

`delta >= threshold` 才接受；不满足时退出码为 1，并给出回滚提示。

## 文件

- `collect.py`：扫描 `evals/results`，取每个基准/子集最新报告，输出指标和失败案例。
- `feedback.py`：把失败案例转成可审阅的补丁 JSON（路由 keywords 增补）。
- `apply.py`：默认 dry-run，把补丁写回 `agents/*_agent.py`。
- `guard.py`：重跑后对比指标，`delta` 低于阈值则拒绝。
