"""Reusable official-style multi-turn tool execution loop."""
import ast
import copy
import json
import os
import time


EVALS_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(EVALS_ROOT, "external", "bfcl_data")
DOC_DIR = os.path.join(DATA_DIR, "multi_turn_func_doc")

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


def build_tools(involved_classes):
    tools = []
    for class_name in involved_classes:
        with open(
            os.path.join(DOC_DIR, DOC_FILE_MAPPING[class_name]),
            encoding="utf-8",
        ) as handle:
            docs = [json.loads(line) for line in handle if line.strip()]
        for fn in docs:
            params = copy.deepcopy(fn["parameters"])
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
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": fn["name"],
                        "description": fn["description"],
                        "parameters": params,
                    },
                }
            )
    return tools


def _load_docs(class_name):
    with open(
        os.path.join(DOC_DIR, DOC_FILE_MAPPING[class_name]),
        encoding="utf-8",
    ) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def add_missed_functions(tools, missed_function, turn_index, docs_by_name):
    names = missed_function.get(str(turn_index), [])
    if not names:
        return tools
    existing = {tool["function"]["name"] for tool in tools}
    for name in names:
        if name in existing or name not in docs_by_name:
            continue
        doc = docs_by_name[name]
        params = copy.deepcopy(doc["parameters"])
        params["type"] = "object"
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": doc["description"],
                    "parameters": params,
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


class DeepSeekFCModelStep:
    def __init__(self, client, model):
        self.client = client
        self.model = model

    def step(self, messages, tools):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            temperature=0.0,
            max_tokens=1024,
        )
        return response.choices[0].message


def run_multi_turn_entry(entry, model_step, ground_truth):
    import sys

    sys.path.insert(0, os.path.join(EVALS_ROOT, "external", "bfcl_official"))
    from bfcl_official_impl.eval_checker.multi_turn_eval.multi_turn_utils import (
        execute_multi_turn_func_call,
    )
    from mini_thalamus.evals.multi_turn_checker import score_official_multi_turn

    tools = build_tools(entry.get("involved_classes", []))
    involved_classes = entry.get("involved_classes", [])
    docs_by_name = {
        doc["name"]: doc
        for class_name in involved_classes
        for doc in _load_docs(class_name)
    }
    initial_config = entry.get("initial_config", {})
    long_context = "long_context" in entry["id"]
    model_name = "mini-thalamus-fc-direct"
    execute_multi_turn_func_call(
        [],
        initial_config,
        involved_classes,
        model_name,
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
    model_results = []
    started = time.perf_counter()
    for turn_index, turn in enumerate(entry.get("question", [])):
        tools = add_missed_functions(
            tools,
            entry.get("missed_function", {}),
            turn_index,
            docs_by_name,
        )
        question = "\n".join(
            message.get("content", "")
            for message in turn
            if message.get("content")
        )
        if not question and str(turn_index) in entry.get("missed_function", {}):
            question = (
                "I have updated some more functions you can choose from. "
                "What about now?"
            )
        calls = []
        if question:
            messages.append({"role": "user", "content": question})
            for _ in range(10):
                message = model_step.step(messages, tools)
                tool_calls = message.tool_calls or []
                if not tool_calls:
                    break
                messages.append(
                    {
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
                )
                call_strings = []
                for tool_call in tool_calls:
                    name = tool_call.function.name
                    try:
                        args = json.loads(tool_call.function.arguments or "{}")
                    except json.JSONDecodeError:
                        args = {}
                    call_strings.append(render_call(name, args))
                execution_results, _ = execute_multi_turn_func_call(
                    call_strings,
                    initial_config,
                    involved_classes,
                    model_name,
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
        ground_truth or [],
        model_results,
        category=entry["id"].rsplit("_", 1)[0],
        model_name=model_name,
    )
    return checker, model_results, latency
