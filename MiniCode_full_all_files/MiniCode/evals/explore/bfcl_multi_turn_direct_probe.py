"""Direct DeepSeek function-calling probe for BFCL multi-turn."""
import argparse
import ast
import copy
import json
import os
import sys
import time


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
EVALS_ROOT = os.path.join(PROJECT_ROOT, "evals")
DATA_DIR = os.path.join(EVALS_ROOT, "external", "bfcl_data")
DOC_DIR = os.path.join(DATA_DIR, "multi_turn_func_doc")
REPORT_DIR = os.path.join(EVALS_ROOT, "results")
sys.path.insert(0, EVALS_ROOT)
sys.path.insert(0, os.path.join(EVALS_ROOT, "external", "bfcl_official"))

DOC_FILE_MAPPING = {
    "GorillaFileSystem": "gorilla_file_system.json",
    "MathAPI": "math_api.json",
    "MessageAPI": "message_api.json",
    "TwitterAPI": "posting_api.json",
    "TicketAPI": "ticket_api.json",
    "TradingBot": "trading_bot.json",
    "TravelAPI": "travel_booking.json",
    "VehicleControlAPI": "vehicle_control.json",
    "WebSearchAPI": "web_search.json",
    "MemoryAPI_kv": "memory_kv.json",
    "MemoryAPI_vector": "memory_vector.json",
    "MemoryAPI_rec_sum": "memory_rec_sum.json",
}


def load_entries(category, limit=0):
    path = os.path.join(DATA_DIR, f"BFCL_v4_{category}.json")
    with open(path, encoding="utf-8") as handle:
        entries = [json.loads(line) for line in handle if line.strip()]
    if limit:
        entries = entries[:limit]
    return entries


def load_ground_truth(category, entry_id):
    path = os.path.join(
        DATA_DIR,
        "possible_answer",
        f"BFCL_v4_{category}.json",
    )
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            answer = json.loads(line)
            if answer.get("id") == entry_id:
                return answer.get("ground_truth", [])
    return []


def _openapi_parameters(parameters):
    params = copy.deepcopy(parameters)
    params["type"] = "object"
    for prop in params.get("properties", {}).values():
        prop_type = prop.get("type")
        if prop_type == "float":
            prop["type"] = "number"
        elif prop_type == "int":
            prop["type"] = "integer"
        elif prop_type == "list":
            prop["type"] = "array"
        elif prop_type == "dict":
            prop["type"] = "object"
    return params


def build_tools(involved_classes):
    tools = []
    for class_name in involved_classes:
        doc_file = DOC_FILE_MAPPING[class_name]
        with open(
            os.path.join(DOC_DIR, doc_file),
            encoding="utf-8",
        ) as handle:
            docs = [json.loads(line) for line in handle if line.strip()]
        for fn in docs:
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": fn["name"],
                        "description": fn["description"],
                        "parameters": _openapi_parameters(fn["parameters"]),
                    },
                }
            )
    return tools


def ast_literal(value):
    if isinstance(value, dict):
        return ast.Dict(
            keys=[ast_literal(k) for k in value],
            values=[ast_literal(v) for v in value.values()],
        )
    if isinstance(value, list):
        return ast.List(elts=[ast_literal(v) for v in value])
    return ast.Constant(value)


def render_call(name, args):
    call = ast.Call(
        func=ast.Name(id=name),
        args=[],
        keywords=[
            ast.keyword(arg=key, value=ast_literal(value))
            for key, value in (args or {}).items()
        ],
    )
    return ast.unparse(call)


def deepseek_client():
    from openai import OpenAI

    sys.path.insert(0, PROJECT_ROOT)
    from config import get_llm_config

    cfg = get_llm_config()
    return OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"]), cfg["model"]


def run_entry(entry, client, model):
    from multi_turn_checker import score_official_multi_turn
    from bfcl_official_impl.eval_checker.multi_turn_eval.multi_turn_utils import (
        execute_multi_turn_func_call,
    )

    tools = build_tools(entry.get("involved_classes", []))
    ground_truth = load_ground_truth(entry["id"].rsplit("_", 1)[0], entry["id"])
    model_results = []
    involved_classes = entry.get("involved_classes", [])
    initial_config = entry.get("initial_config", {})
    long_context = "long_context" in entry["id"]
    _, instances = execute_multi_turn_func_call(
        [],
        initial_config,
        involved_classes,
        "deepseek-fc-direct",
        entry["id"],
        long_context=long_context,
        is_evaL_run=False,
    )
    messages = [
        {
            "role": "system",
            "content": (
                "You are an agent that can call functions to complete each "
                "user request. Call the necessary functions and continue until "
                "the request is complete."
            ),
        }
    ]
    started = time.perf_counter()
    for turn in entry.get("question", []):
        question = "\n".join(
            message.get("content", "")
            for message in turn
            if message.get("content")
        )
        calls = []
        if not question:
            model_results.append([calls])
            continue
        messages.append({"role": "user", "content": question})
        for _ in range(10):
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                temperature=0.0,
                max_tokens=1024,
            )
            message = response.choices[0].message
            tool_calls = message.tool_calls or []
            if not tool_calls:
                break
            assistant_message = {
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments or "{}",
                        },
                    }
                    for tool_call in tool_calls
                ],
            }
            messages.append(assistant_message)
            call_strings = []
            for tool_call in tool_calls:
                name = tool_call.function.name
                try:
                    args = json.loads(tool_call.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                call_strings.append(render_call(name, args))
            execution_results, instances = execute_multi_turn_func_call(
                call_strings,
                initial_config,
                involved_classes,
                "deepseek-fc-direct",
                entry["id"],
                long_context=long_context,
                is_evaL_run=False,
            )
            calls.extend(call_strings)
            for tool_call, result in zip(tool_calls, execution_results):
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result,
                    }
                )
        model_results.append([calls])
    latency = time.perf_counter() - started
    checker = score_official_multi_turn(
        entry,
        ground_truth,
        model_results,
        category=entry["id"].rsplit("_", 1)[0],
        model_name="deepseek-fc-direct",
    )
    return checker, model_results, latency


def write_report(category, records, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    total = len(records)
    passed = sum(1 for record in records if record["passed"])
    metrics = {
        "case_count": total,
        "pass_rate": round(passed / total, 4) if total else 0,
        "avg_latency_ms": round(
            sum(record["latency_s"] for record in records) / total * 1000,
            1,
        )
        if total
        else 0,
    }
    path = os.path.join(
        output_dir,
        f"bfcl-direct-fc-{category}-{stamp}.json",
    )
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "benchmark": "BFCL_v4",
                "category": category,
                "mode": "deepseek-fc-direct",
                "metrics": metrics,
                "records": records,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    return path, metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", default="multi_turn_base")
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--output-dir", default=REPORT_DIR)
    args = parser.parse_args(argv)

    client, model = deepseek_client()
    entries = load_entries(args.category, limit=args.limit)
    records = []
    for entry in entries:
        checker, model_results, latency = run_entry(entry, client, model)
        record = {
            "id": entry["id"],
            "passed": bool(checker["valid"]),
            "latency_s": round(latency, 2),
            "error_type": checker.get("error_type", ""),
            "error_message": checker.get("error_message", ""),
            "model_results": model_results,
        }
        records.append(record)
        print(
            json.dumps(
                {
                    "id": entry["id"],
                    "passed": record["passed"],
                    "error_type": record["error_type"],
                },
                ensure_ascii=False,
            )
        )
    path, metrics = write_report(args.category, records, args.output_dir)
    print(json.dumps({"report": path, "metrics": metrics}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
