"""Official BFCL v4 web_search protocol adapter.

Implements the official multi-turn WebSearchAPI contract:
  - tools: search_engine_query(keywords, max_results, region)
           fetch_url_content(url, mode)
  - the model calls tools until it emits a plain-text final message
  - the last non-tool-call message is scored with the official agentic checker

Search/fetch backends are pluggable: search uses the configured provider chain
(Tavily/Serper/Bocha/SearXNG/Bing), fetch uses the shared url_fetcher.
"""

import argparse
import json
import os
import re
import sys
import time


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVALS_ROOT = os.path.dirname(os.path.abspath(__file__))
EXTERNAL_ROOT = os.path.join(EVALS_ROOT, "external")
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, EVALS_ROOT)
sys.path.insert(0, EXTERNAL_ROOT)

from mini_thalamus.config import get_llm_config  # noqa: E402
from mini_thalamus.tools.url_fetcher import fetch_url  # noqa: E402
from bfcl_official.agentic_checker import agentic_checker  # noqa: E402


DATA_DIR = os.path.join(EVALS_ROOT, "external", "bfcl_data")
MAX_TURNS = 8
TOOL_RESULT_MAX_CHARS = 12000

EMBEDDED_TOOLS = [
    {
        "name": "search_engine_query",
        "description": (
            "This tool belongs to the Web Search API category. "
            "This function queries the search engine for the provided "
            "keywords and region."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "keywords": {
                    "type": "string",
                    "description": "The keywords to search for.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of search results.",
                    "default": 10,
                },
                "region": {
                    "type": "string",
                    "description": "Region code, defaults to wt-wt.",
                    "default": "wt-wt",
                },
            },
            "required": ["keywords"],
        },
    },
    {
        "name": "fetch_url_content",
        "description": (
            "This tool belongs to the Web Search API category. "
            "This function retrieves content from the provided URL and "
            "processes it based on the selected mode."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to fetch content from.",
                },
                "mode": {
                    "type": "string",
                    "description": "raw, markdown, or truncate.",
                    "default": "raw",
                    "enum": ["raw", "markdown", "truncate"],
                },
            },
            "required": ["url"],
        },
    },
]


def _load_official_tools() -> list[dict]:
    """Prefer the official function doc when the sparse source clone exists."""
    official = os.path.abspath(
        os.path.join(
            PROJECT_ROOT,
            "..",
            "..",
            "..",
            "..",
            "tmp",
            "gorilla_src",
            "berkeley-function-call-leaderboard",
            "bfcl_eval",
            "data",
            "multi_turn_func_doc",
            "web_search.json",
        )
    )
    if os.path.exists(official):
        tools = []
        with open(official, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    tools.append(json.loads(line))
        if len(tools) == 2:
            return tools
    return EMBEDDED_TOOLS


def _normalize_tools(tools: list[dict]) -> list[dict]:
    normalized = []
    for tool in tools:
        params = dict(tool.get("parameters", {}))
        params.pop("response", None)
        params["type"] = "object"
        normalized.append(
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": params,
                },
            }
        )
    return normalized


def load_entries(category="web_search"):
    with open(
        os.path.join(DATA_DIR, f"BFCL_v4_{category}.json"),
        encoding="utf-8",
    ) as handle:
        entries = [json.loads(line) for line in handle if line.strip()]
    answers = []
    answer_path = os.path.join(
        DATA_DIR, "possible_answer", f"BFCL_v4_{category}.json"
    )
    if os.path.exists(answer_path):
        with open(answer_path, encoding="utf-8") as handle:
            answers = [json.loads(line) for line in handle if line.strip()]
    return entries, answers


def search_engine_query(
    keywords: str,
    max_results: int = 10,
    region: str = "wt-wt",
) -> list[dict]:
    """Search using the configured provider chain; Bing is the last resort."""
    from mini_thalamus.agents.websearch_agent import WebSearchAgent

    agent = WebSearchAgent.__new__(WebSearchAgent)
    raw = agent._provider_search(keywords or "")
    if not raw:
        raw = agent._bing_search(keywords or "")
    results = []
    for item in raw[: max(1, min(int(max_results or 10), 10))]:
        results.append(
            {
                "title": item.get("title", ""),
                "href": item.get("url", item.get("href", "")),
                "body": item.get("snippet", item.get("body", "")),
            }
        )
    return results


def fetch_url_content(url: str, mode: str = "raw") -> dict:
    """Fetch a URL; map official modes onto the shared url_fetcher."""
    if not isinstance(url, str) or not url.startswith(("http://", "https://")):
        return {"error": f"Invalid URL: {url}"}
    fetch_mode = "raw" if mode in ("raw", "markdown") else "readability"
    result = fetch_url(
        url,
        mode=fetch_mode,
        max_chars=8000,
        timeout=15,
        allow_private_ips=False,
        verify_ssl=True,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/131.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
        },
    )
    if result.get("status") == "success":
        content = result.get("content", "")
        return {"content": content if isinstance(content, str) else str(content)}
    return {"error": result.get("error", "fetch failed")}


def _run_tool(name: str, arguments: dict) -> dict:
    if name == "search_engine_query":
        return {
            "results": search_engine_query(
                arguments.get("keywords", ""),
                arguments.get("max_results", 10),
                arguments.get("region", "wt-wt"),
            )
        }
    if name == "fetch_url_content":
        return fetch_url_content(
            arguments.get("url", ""),
            arguments.get("mode", "raw"),
        )
    return {"error": f"unknown tool {name}"}


def run_single(client, model, question, ground_truth, max_turns=MAX_TURNS):
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert web researcher. Use the provided search and "
                "fetch tools to find the answer. When you have enough evidence, "
                "reply with a plain-text answer containing the exact fact, "
                "without any tool call."
            ),
        },
        {"role": "user", "content": question},
    ]
    started = time.perf_counter()
    tool_calls = 0
    turns = 0
    answer = ""
    for _ in range(max(1, int(max_turns))):
        turns += 1
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=OFFICIAL_TOOLS_NORMALIZED,
            temperature=0.0,
            max_tokens=1024,
        )
        message = response.choices[0].message
        if not message.tool_calls:
            answer = (message.content or "").strip()
            break
        tool_calls += len(message.tool_calls)
        messages.append(
            {
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.function.name,
                            "arguments": call.function.arguments,
                        },
                    }
                    for call in message.tool_calls
                ],
            }
        )
        for call in message.tool_calls:
            try:
                arguments = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {}
            output = _run_tool(call.function.name, arguments)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(output, ensure_ascii=False)[
                        :TOOL_RESULT_MAX_CHARS
                    ],
                }
            )
    latency = time.perf_counter() - started
    checker = agentic_checker(answer, ground_truth)
    return checker, answer, tool_calls, turns, round(latency, 2)


def write_report(records, metrics, output_dir, base_offset=0):
    os.makedirs(output_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    stem = os.path.join(
        output_dir, f"bfcl-adapter-web_search-{stamp}"
    )
    with open(stem + ".json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "benchmark": "BFCL_v4_web_search_adapter",
                "metrics": metrics,
                "records": records,
                "base_offset": base_offset,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    lines = [
        "# BFCL v4 web_search Official Protocol Adapter",
        "",
        f"- Cases: {metrics['case_count']}",
        f"- Pass rate: {metrics['pass_rate']}",
        f"- Avg latency: {metrics['avg_latency_ms']} ms",
        f"- Avg tool calls: {metrics['avg_tool_calls']}",
        "",
        "| id | passed | latency_s | tool_calls | turns |",
        "|---|---:|---:|---:|---:|",
    ]
    for record in records:
        lines.append(
            f"| {record['id']} | {record['passed']} | "
            f"{record['latency_s']} | {record['tool_calls']} | "
            f"{record['turns']} |"
        )
    with open(stem + ".md", "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return stem


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-turns", type=int, default=MAX_TURNS)
    parser.add_argument(
        "--output-dir",
        default=os.path.join(EVALS_ROOT, "results"),
    )
    args = parser.parse_args(argv)

    config = get_llm_config()
    from openai import OpenAI

    client = OpenAI(api_key=config["api_key"], base_url=config["base_url"])
    model = config["model"]
    entries, answers = load_entries()
    if args.offset:
        entries = entries[args.offset:]
        answers = answers[args.offset:]
    if args.limit:
        entries = entries[: args.limit]
        answers = answers[: args.limit]

    records = []
    for index, entry in enumerate(entries):
        question = entry["question"][-1][-1]["content"]
        ground_truth = answers[index].get("ground_truth", []) if answers else []
        checker, answer, tool_calls, turns, latency = run_single(
            client,
            model,
            question,
            ground_truth,
            max_turns=args.max_turns,
        )
        records.append(
            {
                "id": entry["id"],
                "passed": bool(checker["valid"]),
                "latency_s": latency,
                "tool_calls": tool_calls,
                "turns": turns,
                "answer": answer[:500],
                "error_type": checker.get("error_type", "")
                if not checker["valid"]
                else "",
            }
        )
        print(
            json.dumps(
                {
                    "id": entry["id"],
                    "passed": records[-1]["passed"],
                    "latency_s": latency,
                    "tool_calls": tool_calls,
                    "turns": turns,
                },
                ensure_ascii=False,
            )
        )
        if (index + 1) % 10 == 0:
            write_report(
                records,
                aggregate(records),
                args.output_dir,
                base_offset=args.offset,
            )

    metrics = aggregate(records)
    stem = write_report(records, metrics, args.output_dir, args.offset)
    print(
        json.dumps(
            {"report": stem + ".json", "metrics": metrics},
            ensure_ascii=False,
        )
    )
    return 0


def aggregate(records):
    total = len(records)
    passed = sum(1 for r in records if r["passed"])
    latency = sum(r["latency_s"] for r in records)
    tools = sum(r["tool_calls"] for r in records)
    return {
        "case_count": total,
        "pass_rate": round(passed / total, 4) if total else 0,
        "avg_latency_ms": round(latency / total * 1000, 1) if total else 0,
        "avg_tool_calls": round(tools / total, 2) if total else 0,
    }


OFFICIAL_TOOLS_NORMALIZED = _normalize_tools(_load_official_tools())


if __name__ == "__main__":
    raise SystemExit(main())
