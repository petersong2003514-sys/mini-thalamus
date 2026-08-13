"""BFCL v4 external benchmark runner for the ShuShu Agent stack.

Uses the official BFCL v4 data files and the official AST / agentic
checkers (vendored under evals/external/bfcl_official).

Modes:
  deepseek-fc  raw DeepSeek native function calling (official protocol)
  shushu       full ShuShu orchestrator pipeline
"""

import argparse
import ast as py_ast
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MINICODE_ROOT = os.path.dirname(PROJECT_ROOT)
EXTERNAL_ROOT = os.path.join(PROJECT_ROOT, "external")
DATA_DIR = os.path.join(EXTERNAL_ROOT, "bfcl_data")
sys.path.insert(0, EXTERNAL_ROOT)

from bfcl_official.agentic_checker import agentic_checker  # noqa: E402
from bfcl_official.ast_checker import ast_checker  # noqa: E402
from bfcl_official.constants.enums import Language  # noqa: E402
from bfcl_official.constants.type_mappings import GORILLA_TO_OPENAPI  # noqa: E402
from multi_turn_checker import (  # noqa: E402
    score_multi_turn,
    score_official_multi_turn,
)


CATEGORIES = [
    "web_search",
    "live_simple",
    "live_multiple",
    "live_parallel",
    "live_parallel_multiple",
]

OFFLINE_CATEGORIES = [
    "simple_python",
    "simple_java",
    "simple_javascript",
    "multiple",
    "parallel",
    "parallel_multiple",
    "irrelevance",
    "multi_turn_base",
    "multi_turn_long_context",
    "multi_turn_miss_func",
    "multi_turn_miss_param",
    "memory",
]

# Text-answer subsets scored by the official agentic checker.
AGENTIC_CATEGORIES = {"web_search", "memory"}
CATEGORY_ALIASES = {
    "simple": "simple_python",
}

SYSTEM_PROMPT = (
    "You are an expert in composing functions. You are given a question and a set of "
    "possible functions. Based on the question, you will need to make one or more "
    "function/tool calls to achieve the purpose. If none of the functions can be used, "
    "point it out. If the given question lacks the parameters required by the function, "
    "also point it out.\n"
    "You should only return the function calls in your response.\n"
    "If you decide to invoke any of the function(s), you MUST put it in the format of "
    "[func_name1(params_name1=params_value1, params_name2=params_value2...), "
    "func_name2(params)]. You SHOULD NOT include any other text in the response.\n"
    "At each turn, you should try your best to complete the tasks requested by the user "
    "within the current turn. Continue to output functions to call until you have "
    "fulfilled the user's request to the best of your ability. Once you have no more "
    "functions to call, the system will consider the current turn complete and proceed "
    "to the next turn or task.\n"
    "Here is a list of functions in JSON format that you can invoke.\n{functions}\n"
)


def available_categories():
    """Return categories whose primary data files are present locally."""
    names = set()
    if os.path.isdir(DATA_DIR):
        for name in os.listdir(DATA_DIR):
            if not name.startswith("BFCL_v4_") or not name.endswith(".json"):
                continue
            names.add(name[len("BFCL_v4_"):-len(".json")])
    return sorted(names)


def load_entries(category):
    with open(
        os.path.join(DATA_DIR, f"BFCL_v4_{category}.json"),
        encoding="utf-8",
    ) as handle:
        entries = [json.loads(line) for line in handle if line.strip()]
    possible = []
    answer_path = os.path.join(
        DATA_DIR, "possible_answer", f"BFCL_v4_{category}.json"
    )
    if os.path.exists(answer_path):
        with open(answer_path, encoding="utf-8") as handle:
            possible = [json.loads(line) for line in handle if line.strip()]
    return entries, possible


def cast_properties(properties):
    for key, value in properties.items():
        if "type" not in value:
            value["type"] = "string"
        else:
            var_type = value["type"]
            if var_type == "float":
                value["format"] = "float"
            value["type"] = GORILLA_TO_OPENAPI.get(var_type, "string")
        if value["type"] in ("array", "object"):
            if "properties" in value:
                value["properties"] = cast_properties(value["properties"])
            elif "items" in value:
                value["items"]["type"] = GORILLA_TO_OPENAPI.get(
                    value["items"]["type"], "string"
                )
    return properties


def convert_to_tool(functions):
    import copy

    functions = copy.deepcopy(functions)
    tools = []
    for item in functions:
        if "." in item["name"]:
            item["name"] = re.sub(r"\.", "_", item["name"])
        item["parameters"]["type"] = "object"
        item["parameters"]["properties"] = cast_properties(
            item["parameters"]["properties"]
        )
        tools.append({"type": "function", "function": item})
    return tools


def deepseek_client():
    from openai import OpenAI

    sys.path.insert(0, MINICODE_ROOT)
    from config import get_llm_config

    cfg = get_llm_config()
    return OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"]), cfg["model"]


def call_deepseek_fc(client, model, messages, tools, temperature=0.001):
    last_error = None
    for attempt in range(4):
        try:
            return client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                temperature=temperature,
                max_tokens=2048,
            )
        except Exception as exc:
            last_error = exc
            time.sleep(2 * (attempt + 1))
    raise last_error


def decode_fc_response(response):
    try:
        tool_calls = response.choices[0].message.tool_calls
        if not tool_calls:
            return response.choices[0].message.content
        return [
            {call.function.name: call.function.arguments}
            for call in tool_calls
        ]
    except Exception:
        return response.choices[0].message.content


def decode_ast_output(text):
    """Parse BFCL python-format function calls from a model answer."""
    if isinstance(text, list):
        text = "".join(str(item) for item in text)
    text = (text or "").strip().strip("`").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    candidates = []
    if text.startswith("[") and text.endswith("]"):
        candidates.append(text)
    for match in re.finditer(r"\[[^\]]*\]", text):
        candidates.append(match.group(0))

    for candidate in candidates:
        try:
            tree = py_ast.parse(candidate, mode="eval")
            if isinstance(tree.body, py_ast.Call):
                calls = [tree.body]
            elif isinstance(tree.body, py_ast.List):
                calls = tree.body.elts
            else:
                continue
            return [
                {resolve_call_name(call): resolve_keywords(call)}
                for call in calls
                if isinstance(call, py_ast.Call)
            ]
        except SyntaxError:
            continue

    try:
        data = json.loads(text)
        if isinstance(data, list):
            result = []
            for item in data:
                if isinstance(item, dict) and "function" in item:
                    result.append({item["function"]: item.get("parameters", {})})
                elif isinstance(item, dict):
                    result.append(item)
            return result
    except (json.JSONDecodeError, TypeError):
        pass
    return None


def resolve_call_name(call):
    parts = []
    func = call.func
    while isinstance(func, py_ast.Attribute):
        parts.append(func.attr)
        func = func.value
    if isinstance(func, py_ast.Name):
        parts.append(func.id)
    return ".".join(reversed(parts))


def resolve_keywords(call):
    result = {}
    for arg in call.keywords:
        result[arg.arg] = resolve_value(arg.value)
    return result


def resolve_value(node):
    if isinstance(node, py_ast.Constant):
        return node.value
    if isinstance(node, py_ast.List):
        return [resolve_value(item) for item in node.elts]
    if isinstance(node, py_ast.Tuple):
        return [resolve_value(item) for item in node.elts]
    if isinstance(node, py_ast.Dict):
        return {
            resolve_value(k): resolve_value(v)
            for k, v in zip(node.keys, node.values)
        }
    if isinstance(node, py_ast.UnaryOp) and isinstance(
        node.op, (py_ast.USub, py_ast.UAdd)
    ):
        sign = -1 if isinstance(node.op, py_ast.USub) else 1
        value = node.operand.value
        return sign * value if isinstance(value, (int, float)) else node
    if isinstance(node, py_ast.Name):
        return node.id
    return None


def ast_literal(value):
    if isinstance(value, dict):
        return py_ast.Dict(
            keys=[ast_literal(k) for k in value],
            values=[ast_literal(v) for v in value.values()],
        )
    if isinstance(value, list):
        return py_ast.List(elts=[ast_literal(v) for v in value])
    return py_ast.Constant(value)


def render_decoded_call(name, kwargs):
    call = py_ast.Call(
        func=py_ast.Name(id=name),
        args=[],
        keywords=[
            py_ast.keyword(arg=key, value=ast_literal(value))
            for key, value in kwargs.items()
        ],
    )
    return py_ast.unparse(call)


def run_deepseek_fc(entry, ground_truth, client, model):
    messages = [
        dict(message)
        for turn in entry["question"]
        for message in turn
    ]
    tools = convert_to_tool(entry["function"])
    started = time.perf_counter()
    response = call_deepseek_fc(client, model, messages, tools)
    raw = decode_fc_response(response)
    latency = time.perf_counter() - started
    decoded = None
    if isinstance(raw, list):
        decoded = []
        for item in raw:
            try:
                name = list(item.keys())[0]
                args = json.loads(item[name]) if isinstance(item[name], str) else item[name]
                decoded.append({name: args})
            except (json.JSONDecodeError, TypeError, IndexError):
                return {
                    "valid": False,
                    "error": ["Failed to parse tool call JSON"],
                    "error_type": "decoder_failed",
                }, raw, latency
    else:
        decoded = decode_ast_output(raw)
    if decoded is None:
        return {
            "valid": False,
            "error": ["No function call found in model response"],
            "error_type": "decoder_failed",
        }, raw, latency
    return (
        ast_checker(
            entry["function"],
            decoded,
            ground_truth,
            Language.PYTHON,
            re.sub(r"_\d+-\d+-\d+$", "", entry["id"])
            if entry["id"].startswith("live_")
            else "live_simple",
            "deepseek-chat-fc",
        ),
        raw,
        latency,
    )


def build_prompting_message(entry):
    functions_json = json.dumps(entry["function"], indent=4)
    system_prompt = SYSTEM_PROMPT.format(functions=functions_json)
    user_question = entry["question"][-1][-1]["content"]
    return system_prompt + "\n\nUser request:\n" + user_question


def get_orchestrator():
    sys.path.insert(0, MINICODE_ROOT)
    from agent_runtime import get_runtime
    from orchestrator import Orchestrator

    runtime = get_runtime()
    agents = runtime.load_all()
    return Orchestrator(agents=agents, runtime=runtime)


def run_shushu_single(orchestrator, message, session_id, timeout=300):
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(
            orchestrator.process,
            message,
            user_id="bfcl-eval",
            session_id=session_id,
        )
        try:
            return future.result(timeout=timeout)
        except FutureTimeout:
            return {"status": "timeout", "answer": "", "data": {}}
        except Exception as exc:
            return {"status": "error", "answer": str(exc), "data": {}}


def run_shushu_web_search(entry, orchestrator, ground_truth=None):
    question = entry["question"][-1][-1]["content"]
    started = time.perf_counter()
    result = run_shushu_single(
        orchestrator, question, session_id="bfcl-" + entry["id"]
    )
    latency = time.perf_counter() - started
    answer = result.get("answer", "") or result.get("summary", "") or ""
    if not answer and result.get("status") == "error":
        answer = str(result.get("data", {}))
    checker = agentic_checker(
        answer,
        ground_truth if ground_truth is not None else entry.get("ground_truth", []),
    )
    return checker, answer, latency, result.get("status", "success")


def run_shushu_function_call(entry, ground_truth, orchestrator):
    message = build_prompting_message(entry)
    started = time.perf_counter()
    result = run_shushu_single(
        orchestrator, message, session_id="bfcl-fc-" + entry["id"]
    )
    latency = time.perf_counter() - started
    answer = result.get("answer", "") or result.get("summary", "") or ""
    decoded = decode_ast_output(answer)
    if decoded is None:
        return {
            "valid": False,
            "error": ["No function call found in ShuShu answer"],
            "error_type": "decoder_failed",
        }, answer, latency, result.get("status", "success")
    return (
        ast_checker(
            entry["function"],
            decoded,
            ground_truth,
            Language.PYTHON,
            re.sub(r"_\d+-\d+-\d+$", "", entry["id"]) or "live_simple",
            "shushu-prompting",
        ),
        answer,
        latency,
        result.get("status", "success"),
    )


def run_shushu_agentic(entry, ground_truth, orchestrator):
    """Run text-answer subsets (memory/irrelevance/web_search) via ShuShu."""
    question = entry["question"][-1][-1]["content"]
    if entry.get("scenario"):
        question = f"场景: {entry['scenario']}\n{question}"
    started = time.perf_counter()
    result = run_shushu_single(
        orchestrator, question, session_id="bfcl-agentic-" + entry["id"]
    )
    latency = time.perf_counter() - started
    answer = result.get("answer", "") or result.get("summary", "") or ""
    if not answer and result.get("status") == "error":
        answer = str(result.get("data", {}))
    checker = agentic_checker(
        answer,
        ground_truth if ground_truth is not None else entry.get("ground_truth", []),
    )
    return checker, answer, latency, result.get("status", "success")


def run_shushu_irrelevance(entry, orchestrator):
    """Irrelevance subset: a correct model refuses the unrelated tool."""
    question = entry["question"][-1][-1]["content"]
    started = time.perf_counter()
    result = run_shushu_single(
        orchestrator, question, session_id="bfcl-irrelevance-" + entry["id"]
    )
    latency = time.perf_counter() - started
    answer = result.get("answer", "") or result.get("summary", "") or ""
    decoded = decode_ast_output(answer)
    passed = decoded is None or len(decoded) == 0
    return {
        "valid": passed,
        "error_type": "irrelevance:false_positive_call" if not passed else "",
        "error": ["model returned an irrelevant tool call"] if not passed else [],
    }, answer, latency, result.get("status", "success")


def run_shushu_multi_turn(entry, ground_truth, orchestrator):
    """Run each turn and score with the official executable checker."""
    turns = entry.get("question", [])
    model_results = []
    statuses = []
    started = time.perf_counter()
    for turn_index, turn in enumerate(turns):
        question = "\n".join(
            message.get("content", "")
            for message in turn
            if message.get("content")
        )
        result = run_shushu_single(
            orchestrator,
            question or "请完成上述任务。",
            session_id=f"bfcl-multi-turn-{entry['id']}-{turn_index}",
        )
        statuses.append(result.get("status", "success"))
        answer = result.get("answer", "") or result.get("summary", "") or ""
        decoded = decode_ast_output(answer)
        calls = []
        if decoded:
            for item in decoded:
                for name, kwargs in item.items():
                    calls.append(render_decoded_call(name, kwargs or {}))
        model_results.append([calls])
    latency = time.perf_counter() - started
    checker = score_official_multi_turn(
        entry,
        ground_truth or [],
        model_results,
        category=entry["id"].rsplit("_", 1)[0],
        model_name="shushu",
    )
    raw = "\n".join(
        str(step_calls) for turn_calls in model_results for step_calls in turn_calls
    )
    status = (
        "success"
        if all(item == "success" for item in statuses)
        else ("timeout" if any(item == "timeout" for item in statuses) else "error")
    )
    return checker, raw, latency, status


def category_kind(category):
    if category.startswith("multi_turn_"):
        return "multi_turn"
    return "agentic" if category in AGENTIC_CATEGORIES else "ast"


def write_report(category, mode, records, metrics, output_dir, base_offset=0):
    os.makedirs(output_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    stem = os.path.join(output_dir, f"bfcl-{mode}-{category}-{stamp}")
    with open(stem + ".json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "benchmark": "BFCL_v4",
                "category": category,
                "mode": mode,
                "metrics": metrics,
                "records": records,
                "base_offset": base_offset,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    lines = [
        "# BFCL v4 External Evaluation",
        "",
        f"- Category: {category}",
        f"- Mode: {mode}",
        f"- Cases: {metrics['case_count']}",
        f"- Pass rate: {metrics['pass_rate']}",
        f"- Avg latency: {metrics['avg_latency_ms']} ms",
        f"- Error rate: {metrics['error_rate']}",
        "",
        "| id | passed | latency_s | error_type |",
        "|---|---:|---:|---|",
    ]
    for record in records:
        lines.append(
            f"| {record['id']} | {record['passed']} | "
            f"{record['latency_s']} | {record.get('error_type', '')} |"
        )
    with open(stem + ".md", "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return stem


def aggregate(records):
    total = len(records)
    passed = sum(1 for r in records if r["passed"])
    latency = sum(r["latency_s"] for r in records)
    errors = sum(1 for r in records if r.get("error_type"))
    return {
        "case_count": total,
        "pass_rate": round(passed / total, 4) if total else 0,
        "avg_latency_ms": round(latency / total * 1000, 1) if total else 0,
        "error_rate": round(errors / total, 4) if total else 0,
    }


def latest_checkpoint_offset(output_dir, category, mode):
    """Return the next entry offset from the newest checkpoint report, or 0."""
    pattern = re.compile(
        rf"^bfcl-{re.escape(mode)}-checkpoint-{re.escape(category)}-(\d{{8}}-\d{{6}})\.json$"
    )
    found = []
    if os.path.isdir(output_dir):
        for name in os.listdir(output_dir):
            match = pattern.match(name)
            if match:
                found.append((match.group(1), os.path.join(output_dir, name)))
    if not found:
        return 0
    found.sort()
    path = found[-1][1]
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        base = int(data.get("base_offset", 0) or 0)
        return base + int(data.get("metrics", {}).get("case_count", 0) or 0)
    except Exception as exc:
        print(json.dumps({"resume_warning": f"cannot parse {path}: {exc}"}))
        return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--category",
        default="web_search",
        help="comma-separated BFCL v4 categories, or 'all'",
    )
    parser.add_argument(
        "--mode",
        default="auto",
        choices=["auto", "deepseek-fc", "shushu", "bfcl-fc"],
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue from the newest checkpoint report",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(PROJECT_ROOT, "results"),
    )
    parser.add_argument(
        "--list-categories",
        action="store_true",
        help="print locally available categories and exit",
    )
    parser.add_argument(
        "--check-data",
        action="store_true",
        help="load categories and print case counts without invoking a model",
    )
    args = parser.parse_args(argv)

    if args.list_categories:
        print(json.dumps(
            {
                "available": available_categories(),
                "offline": OFFLINE_CATEGORIES,
                "aliases": CATEGORY_ALIASES,
            },
            ensure_ascii=False,
            indent=2,
        ))
        return 0

    categories = [c.strip() for c in args.category.split(",") if c.strip()]
    if categories == ["all"]:
        categories = available_categories()
    categories = [CATEGORY_ALIASES.get(c, c) for c in categories]

    if args.check_data:
        summary = {}
        for category in categories:
            entries, possible = load_entries(category)
            summary[category] = {
                "entries": len(entries),
                "possible_answers": len(possible),
            }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    known = set(available_categories())
    for category in categories:
        if category not in known:
            print(json.dumps(
                {
                    "error": f"unknown category {category}",
                    "available": sorted(known),
                    "hint": (
                        "run `python evals/download_bfcl.py --offline` "
                        "to fetch the official offline subsets"
                    ),
                },
                ensure_ascii=False,
            ))
            return 1

    client = None
    model = None
    orchestrator = None

    for category in categories:
        entries, possible = load_entries(category)

        mode = args.mode
        if mode == "auto":
            mode = (
                "shushu"
                if category_kind(category) == "agentic"
                else (
                    "bfcl-fc"
                    if category_kind(category) == "multi_turn"
                    else "deepseek-fc"
                )
            )
        run_offset = args.offset
        if args.resume:
            run_offset = latest_checkpoint_offset(args.output_dir, category, mode)
            print(
                json.dumps(
                    {
                        "resume": True,
                        "category": category,
                        "mode": mode,
                        "offset": run_offset,
                    },
                    ensure_ascii=False,
                )
            )
        if run_offset:
            entries = entries[run_offset:]
            possible = possible[run_offset:]
        if args.limit:
            entries = entries[: args.limit]
            possible = possible[: args.limit]

        if mode == "deepseek-fc":
            if client is None:
                client, model = deepseek_client()
        else:
            if orchestrator is None:
                orchestrator = get_orchestrator()

        records = []
        from agents.websearch_agent import SEARCH_PROVIDER_CALLS

        before_calls = dict(SEARCH_PROVIDER_CALLS)
        for index, entry in enumerate(entries):
            ground_truth = (
                possible[index].get("ground_truth", []) if possible else []
            )
            if mode == "deepseek-fc":
                checker, raw, latency = run_deepseek_fc(
                    entry, ground_truth, client, model
                )
                records.append(
                    {
                        "id": entry["id"],
                        "passed": bool(checker["valid"]),
                        "latency_s": round(latency, 2),
                        "error_type": checker.get("error_type", "")
                        if not checker["valid"]
                        else "",
                        "raw": raw,
                    }
                )
            elif category_kind(category) == "agentic":
                checker, answer, latency, status = run_shushu_agentic(
                    entry, ground_truth, orchestrator
                )
            elif mode == "bfcl-fc" and category_kind(category) == "multi_turn":
                from multi_turn_runner import (
                    DeepSeekFCModelStep,
                    run_multi_turn_entry,
                )

                if client is None:
                    client, model = deepseek_client()
                model_step = DeepSeekFCModelStep(client, model)
                checker, model_results, latency = run_multi_turn_entry(
                    entry,
                    model_step,
                    ground_truth,
                )
                answer = json.dumps(model_results, ensure_ascii=False)
                status = "success"
            elif category_kind(category) == "multi_turn":
                checker, answer, latency, status = run_shushu_multi_turn(
                    entry, ground_truth, orchestrator
                )
            elif category == "irrelevance":
                checker, answer, latency, status = run_shushu_irrelevance(
                    entry, orchestrator
                )
            else:
                checker, answer, latency, status = run_shushu_function_call(
                    entry, ground_truth, orchestrator
                )
            if mode != "deepseek-fc":
                records.append(
                    {
                        "id": entry["id"],
                        "passed": bool(checker["valid"]),
                        "latency_s": round(latency, 2),
                        "error_type": checker.get("error_type", "")
                        if not checker["valid"]
                        else "",
                        "status": status,
                        "answer": answer,
                        "question": entry["question"][-1][-1]["content"],
                    }
                )
            after_calls = dict(SEARCH_PROVIDER_CALLS)
            records[-1]["search_calls"] = {
                key: after_calls[key] - before_calls.get(key, 0)
                for key in after_calls
            }
            before_calls = after_calls
            print(
                json.dumps(
                    {
                        "id": entry["id"],
                        "passed": records[-1]["passed"],
                        "latency_s": records[-1]["latency_s"],
                    },
                    ensure_ascii=False,
                )
            )

            if (index + 1) % 10 == 0:
                write_report(
                    category,
                    f"{mode}-checkpoint",
                    records,
                    aggregate(records),
                    args.output_dir,
                    base_offset=run_offset,
                )
        metrics = aggregate(records)
        metrics["search_calls"] = dict(SEARCH_PROVIDER_CALLS)
        metrics["tavily_calls"] = int(SEARCH_PROVIDER_CALLS.get("tavily", 0))
        stem = write_report(
            category,
            mode,
            records,
            metrics,
            args.output_dir,
            base_offset=run_offset,
        )
        print(
            json.dumps(
                {
                    "report": stem + ".json",
                    "category": category,
                    "mode": mode,
                    "metrics": metrics,
                },
                ensure_ascii=False,
            )
        )
        tavily_quota = int(os.environ.get("SHUSHU_TAVILY_QUOTA", "1000"))
        if metrics["tavily_calls"] >= tavily_quota * 0.9:
            print(
                json.dumps(
                    {
                        "tavily_quota_warning": True,
                        "tavily_calls": metrics["tavily_calls"],
                        "quota": tavily_quota,
                    },
                    ensure_ascii=False,
                )
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
