# BFCL v4 External Benchmark Status

Date: 2026-08-03

Harness: `evals/bfcl_eval.py`
Data: `evals/external/bfcl_data/` (official BFCL v4 files)
Checkers: `evals/external/bfcl_official/` (vendored official AST / agentic checkers)

## Run Status

### Raw DeepSeek native function calling (`deepseek-fc`, official AST protocol)

| Category | Pass rate | Cases | Avg latency |
|---|---:|---:|---:|
| live_simple | 89.15% | 258 | 1355.3 ms |
| live_parallel | 75.0% | 16 | 1535.6 ms |
| live_parallel_multiple | 66.67% | 24 | 1816.7 ms |

`live_multiple` (1053 cases) has not been run yet.

### Mini Thalamus full pipeline on official `web_search`

| Run | Pass rate | Cases | Avg latency |
|---|---:|---:|---:|
| Pre-fix (harness bug present) | 0.0% | 38 | 32.5 s |
| After query/Tavily/answer fixes | 0.0% | 10 | 38.1 s |
| After harness ground-truth fix | 20.0% | 10 | 50.1 s |
| After Step 2 call/fetch/placeholder fixes | 20.0% | 10 | 52.4 s |
| After shared extractor void-tag fix | 20.0% | 10 | 43.0 s |

The first 38-case run was invalid: `evals/bfcl_eval.py` checked web_search
answers against empty `ground_truth` instead of the official possible answers.
After passing the official possible answers to the checker, the first 10 cases
score 20% (web_search_2 and web_search_9 pass). Checkpoint reports are written
every 10 cases.

The Step 2 run removed the per-step LLM keyword calls, added a placeholder
guard, and added a page-fetch retry; the 10-case score was unchanged at 20%
(32 Tavily calls). The shared HTML extractor had a void-tag bug: `meta`/`link`
without closing tags inflated skip depth and swallowed all body text. After the
fix, most pages fetch 3000 chars, and the rerun dropped to 29 Tavily calls and
43.0 s/case while still scoring 20%. Only bot-blocked sites such as kbb.com,
reddit.com and zhihu.com remain unfetchable.

Observed root causes (fixed):

- Tavily returned HTTP 400 when the full `[Current task] ...` workflow context
  was sent as the search query. Fixed by cleaning the query before search and
  truncating it in `_provider_tavily` (`agents/websearch_agent.py`).
- Answers were Chinese or had the expected token glued to punctuation (for
  example `Black Tickets (1979)`), which the official substring checker rejects.
  Fixed by answering English questions in English and isolating the final answer.
- `evals/bfcl_eval.py` never passed the official `possible_answer` ground truth
  to the agentic checker for `web_search`, so every case failed with
  `answer_not_found`. Fixed by threading `ground_truth` into
  `run_mt_web_search`.

Remaining gaps:

- Page fetching is fixed for most sites (Wikipedia, news, Goodreads, etc.).
  Remaining failures are bot-blocked sites such as kbb.com, reddit.com and
  zhihu.com; a headless-browser fallback behind `MT_WEB_FETCH_BROWSER=1`
  does not bypass all of them.
- The remaining failures are multi-hop retrieval/synthesis errors: the wrong
  sub-question or entity gets picked (for example Burj Khalifa instead of the
  Bangkok building, Columbia instead of Zhejiang).

## Official-Protocol Adapter (DeepSeek native tool calling)

New: `evals/bfcl_web_search_adapter.py` implements the official BFCL v4
WebSearchAPI contract instead of the Mini Thalamus pipeline:

- Tools: `search_engine_query(keywords, max_results, region)` and
  `fetch_url_content(url, mode)` from the official function docs.
- The model runs multi-turn tool calls, then replies with plain text.
- The final non-tool-call message is scored with the official
  `agentic_checker`.

Run:

```powershell
python evals/bfcl_web_search_adapter.py --limit 10 --offset 0
```

Search uses the configured provider chain (Tavily/Serper/Bocha/SearXNG/Bing);
fetch uses the shared `url_fetcher`. Because the search backend is not the
official SerpAPI DuckDuckGo engine, scores are protocol-comparable but not
bit-identical to the leaderboard.
## BFCL v4 Reference (BenchmarkList import, sampled 2026-05-28)

Secondary citation; official leaderboard is
`https://gorilla.cs.berkeley.edu/leaderboard.html`. Model names on the import
page are abbreviated.

| Model | Overall | Non-live AST | Live | Web search |
|---|---:|---:|---:|---:|
| Opus 4.5 | 77.47% | 88.58% | 79.79% | 84.5% |
| Sonnet 4.5 | 73.24% | 88.65% | 81.13% | 81.0% |
| Gemini 3 | 72.51% | 90.65% | 83.12% | 80.0% |
| DeepSeek V3.2-Exp (FC) | 56.73% | 85.52% | 76.02% | 58.0% |
| GPT-5.2-2025-12-11 | 55.87% | 81.85% | 70.39% | 75.5% |
| GPT-5 Mini | 55.46% | 69.85% | 58.62% | 82.0% |

## Interpretation

The raw DeepSeek function-calling numbers were produced with the official
checker and sit in the same band as the DeepSeek V3.2-Exp reference rows,
so they are directly comparable.

After the harness fix, Mini Thalamus\'s current web_search baseline is 20% on the first
10 official cases (unchanged after Step 2 call-reduction fixes). That is well
below the published 75-85% range of frontier models, and the remaining failures
are mostly retrieval/factual errors rather than protocol errors.

## Next Steps

1. Improve page fetching / snippet quality so multi-hop facts come from real
   page content instead of snippets.
2. Harden multi-hop planner to substitute resolved values (no `[birthplace]`,
   `{city}` placeholders).
3. Run the full official 100-case `web_search` set with checkpoints.
4. Optionally run `live_multiple` (1053 cases) for a complete live AST picture.

## D1 Status (2026-08-13)

`bfcl_eval.py` now discovers categories from the data directory, supports
`--list-categories` and `--check-data`, and scores the official offline
subsets:

- `simple_python` (400), `multiple` (200), `parallel` (200),
  `parallel_multiple` (200), `irrelevance` (240), `memory` (155),
  `multi_turn_base` / `multi_turn_long_context` /
  `multi_turn_miss_func` / `multi_turn_miss_param` (200 each).
- `irrelevance` uses refusal detection (no tool call) because the official
  dataset has no `possible_answer` mirror.
- `multi_turn_*` files are vendored and scored with a lightweight ordered
  tool-path checker (`evals/multi_turn_checker.py`); the official per-turn
  simulator is not vendored, so scores are path-order evidence, not full
  simulator parity.
- Official raw data is downloaded by `evals/download_bfcl.py` from
  `HuanzhiMao/gorilla` -> `berkeley-function-call-leaderboard/bfcl_eval/data`.
- Smoke runs (1 case each) on 2026-08-13: `simple_python` pass, `multiple`
  pass, `parallel` fail, `parallel_multiple` fail, all with DeepSeek FC.

```powershell
python evals/bfcl_eval.py --check-data --category simple_python,multiple,parallel,parallel_multiple,irrelevance,memory
python evals/bfcl_eval.py --category simple_python --mode deepseek-fc --limit 1
```

## BEIR Status

`evals/beir_eval.py` supports official downloads, local BEIR layout, BM25 and
embedding retrieval, and nDCG@10/MRR@10 reports.  NFCorpus and SciFact are
vendored locally:

| Dataset | Corpus | Queries | Qrels | BM25 smoke (20) |
|---|---:|---:|---:|---:|
| nfcorpus | 3633 | 323 | 323 | nDCG@10 0.518, MRR@10 0.4867 |
| scifact | 5183 | 300 | 300 | nDCG@10 0.6631, MRR@10 0.6333 |

```powershell
python evals/beir_eval.py --dataset nfcorpus --limit 20 --bm25
python evals/beir_eval.py --smoke --bm25
```

## Spider Status

`evals/spider_eval.py` is implemented: it builds the NL2SQL peer agent with
`schema_strategy`, SmartDisambiguator and the agent's self-correction, then
reports Execution Accuracy by comparing SQLite result sets.  The official
Google Drive dataset link returned HTTP 404 on 2026-08-13, so the dev data
download is not complete in this workspace; `--check-data` and actual
Execution Accuracy runs require a local `spider/` directory or a working
official mirror.

## Self-Evolution Loop

`evals/self_evolution/` implements collect -> feedback -> apply(dry-run) ->
guard.  The loop only consumes external benchmark reports (`evals/results`),
not `capability_cases.jsonl` / `router_cases.jsonl`.  See
`evals/self_evolution/README.md`.

Sources:

- https://benchmarklist.com/benchmarks/bfcl_v4/
- https://gorilla.cs.berkeley.edu/leaderboard.html
- https://gorilla.cs.berkeley.edu/blogs/15_bfcl_v4_web_search.html
