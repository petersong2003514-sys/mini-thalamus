import unittest

from base_agent import AgentResult
from agents.report_writer_agent import ReportWriterAgent
from route_schema import RouteDecision
from workflow_executor import WorkflowExecutor


SPECS = {
    "nl2sql": {
        "output_schema": {
            "exports": ["answer", "summary", "data.structured"],
        },
    },
    "code": {
        "output_schema": {
            "exports": ["answer", "summary", "data.artifacts"],
        },
    },
    "report_writer": {
        "output_schema": {
            "exports": ["answer", "summary", "data.report"],
        },
    },
}


class RecordingAgent:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def run(self, query, context=None):
        self.calls.append({"query": query, "context": context or {}})
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class FlakyAgent:
    def __init__(self):
        self.calls = 0

    def run(self, query, context=None):
        self.calls += 1
        if self.calls == 1:
            return {
                "status": "error",
                "answer": "temporary",
                "metadata": {"error_type": "transient_error"},
            }
        return {"status": "success", "answer": "recovered"}


class FakeRuntime:
    def __init__(self, agents):
        self.agents = agents

    def get_agent(self, agent_id):
        return self.agents[agent_id]


def decision(payload):
    return RouteDecision.from_dict(payload, agent_specs=SPECS)


def call(call_id, intent, agent, instruction, depends_on=None, input_refs=None,
         policy=None):
    value = {
        "call_id": call_id,
        "intent": intent,
        "function": {
            "name": agent,
            "arguments": {
                "instruction": instruction,
                "input_refs": input_refs or [],
            },
        },
        "depends_on": depends_on or [],
        "confidence": 0.9,
    }
    if policy is not None:
        value["policy"] = policy
    return value


def route_payload(calls):
    return {
        "mode": "single" if len(calls) == 1 else "workflow",
        "calls": calls,
        "candidates": [],
        "overall_confidence": 0.9,
        "method": "test",
        "reason": "test workflow",
        "trace_id": "trace-test",
    }


class WorkflowExecutorTests(unittest.TestCase):
    def test_memory_context_is_rendered_for_a_context_ignorant_real_agent(self):
        report = ReportWriterAgent.__new__(ReportWriterAgent)
        seen_topics = []

        def generate_outline(topic):
            seen_topics.append(topic)
            return {"title": "Report", "sections": []}

        report._generate_outline = generate_outline
        report._write_sections = lambda topic, outline: []
        executor = WorkflowExecutor(
            runtime=FakeRuntime({"report_writer": report})
        )
        route = decision(route_payload([
            call(
                "step_1",
                "report_generation",
                "report_writer",
                "Write the report",
            ),
        ]))

        result = executor.execute(
            route,
            original_query="Continue the prior report",
            context_provider=lambda route_call, results: {
                "memory": {
                    "current_query": "Write the report",
                    "relevant_history": [
                        {"q": "Prior question", "a": "Prior answer"},
                    ],
                    "tags": ["report"],
                    "long_term": {"language": "Chinese"},
                    "files": [],
                    "session_id": "s1",
                    "previous_result": {"summary": "Research complete"},
                    "conversation_history": "user: Earlier context",
                    "workflow_inputs": {
                        "step_0.data": {"value": 42},
                    },
                },
            },
        )

        self.assertEqual(result["status"], "success")
        rendered = seen_topics[0]
        self.assertIn("Write the report", rendered)
        self.assertIn("Prior answer", rendered)
        self.assertIn("Chinese", rendered)
        self.assertIn("Research complete", rendered)
        self.assertIn('"value": 42', rendered)

    def test_single_intent_executes_one_agent_with_atomic_instruction(self):
        nl2sql = RecordingAgent(AgentResult(
            answer="42",
            summary="done",
            data={"structured": {"rows": [[42]]}},
        ))
        executor = WorkflowExecutor(runtime=FakeRuntime({"nl2sql": nl2sql}))
        route = decision(route_payload([
            call("step_1", "database_query", "nl2sql", "查询销售额"),
        ]))

        result = executor.execute(route, original_query="原始复合描述")

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["answer"], "42")
        self.assertEqual(nl2sql.calls[0]["query"], "查询销售额")
        self.assertEqual(len(result["trace"]), 1)

    def test_dependency_outputs_are_resolved_by_input_refs(self):
        nl2sql = RecordingAgent({
            "status": "success",
            "answer": "data",
            "summary": "query done",
            "data": {"structured": {"headers": ["region"], "rows": [["华东"]]}},
        })
        code = RecordingAgent({
            "status": "success",
            "answer": "chart",
            "summary": "chart done",
            "data": {"artifacts": ["chart.png"]},
        })
        report = RecordingAgent({
            "status": "success",
            "answer": "report",
            "summary": "report done",
            "data": {"report": "final.md"},
        })
        executor = WorkflowExecutor(
            runtime=FakeRuntime({
                "nl2sql": nl2sql,
                "code": code,
                "report_writer": report,
            })
        )
        route = decision(route_payload([
            call("step_1", "database_query", "nl2sql", "查询销售额"),
            call(
                "step_2",
                "data_visualization",
                "code",
                "绘制图表",
                depends_on=["step_1"],
                input_refs=["step_1.data.structured"],
            ),
            call(
                "step_3",
                "report_generation",
                "report_writer",
                "生成报告",
                depends_on=["step_1", "step_2"],
                input_refs=[
                    "step_1.data.structured",
                    "step_2.data.artifacts",
                ],
            ),
        ]))

        result = executor.execute(route, original_query="查数据画图写报告")

        self.assertEqual(result["status"], "success")
        self.assertEqual(code.calls[0]["query"], "绘制图表")
        self.assertEqual(
            code.calls[0]["context"]["workflow_inputs"],
            {
                "step_1.data.structured": {
                    "headers": ["region"],
                    "rows": [["华东"]],
                },
            },
        )
        self.assertEqual(
            report.calls[0]["context"]["workflow_inputs"][
                "step_2.data.artifacts"
            ],
            ["chart.png"],
        )
        self.assertEqual(
            [entry["agent_id"] for entry in result["trace"]],
            ["nl2sql", "code", "report_writer"],
        )

    def test_failed_dependency_is_skipped_but_independent_call_runs(self):
        nl2sql = RecordingAgent(RuntimeError("database down"))
        code = RecordingAgent({
            "status": "success",
            "answer": "chart",
            "summary": "chart",
            "data": {"artifacts": []},
        })
        report = RecordingAgent({
            "status": "success",
            "answer": "independent report",
            "summary": "report",
            "data": {"report": "ok"},
        })
        executor = WorkflowExecutor(
            runtime=FakeRuntime({
                "nl2sql": nl2sql,
                "code": code,
                "report_writer": report,
            })
        )
        route = decision(route_payload([
            call("step_1", "database_query", "nl2sql", "查询"),
            call(
                "step_2",
                "data_visualization",
                "code",
                "绘图",
                depends_on=["step_1"],
                input_refs=["step_1.data.structured"],
            ),
            call("step_3", "report_generation", "report_writer", "写独立说明"),
        ]))

        result = executor.execute(route, original_query="test")

        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["results"]["step_1"]["status"], "error")
        self.assertEqual(result["results"]["step_2"]["status"], "skipped")
        self.assertEqual(result["results"]["step_3"]["status"], "success")
        self.assertEqual(code.calls, [])
        self.assertEqual(len(report.calls), 1)

    def test_high_risk_call_returns_checkpoint_before_execution(self):
        code = RecordingAgent({
            "status": "success",
            "answer": "done",
            "summary": "done",
            "data": {"artifacts": []},
        })
        executor = WorkflowExecutor(runtime=FakeRuntime({"code": code}))
        route = decision(route_payload([
            call(
                "step_1",
                "code_execution",
                "code",
                "执行代码",
                policy={
                    "risk_level": "workspace_write",
                    "requires_approval": True,
                    "reason": "writes files",
                },
            ),
        ]))

        result = executor.execute(route, original_query="执行代码")

        self.assertEqual(result["status"], "checkpoint")
        self.assertEqual(result["pending_call"]["call_id"], "step_1")
        self.assertEqual(code.calls, [])

    def test_transient_error_gets_one_bounded_retry(self):
        agent = FlakyAgent()
        executor = WorkflowExecutor(FakeRuntime({"nl2sql": agent}))
        route = decision(route_payload([
            call("step_1", "database_query", "nl2sql", "query"),
        ]))

        result = executor.execute(route, original_query="query")

        self.assertEqual(result["status"], "success")
        self.assertEqual(agent.calls, 2)
        self.assertEqual(result["trace"][0]["attempt"], 2)
        self.assertTrue(result["trace"][0]["retry"])



    def test_missing_agent_becomes_skipped_step_not_crash(self):
        executor = WorkflowExecutor(runtime=FakeRuntime({}))
        route = decision(route_payload([
            call("step_1", "database_query", "nl2sql", "query data"),
        ]))
        result = executor.execute(route, original_query="query data")
        self.assertEqual(result["status"], "error")
        self.assertIn("自动跳过", result.get("answer", ""))

    def test_replan_hook_replaces_pending_calls_after_failure(self):
        nl2sql = RecordingAgent({
            "status": "error",
            "answer": "db down",
            "summary": "db down",
        })
        report = RecordingAgent({
            "status": "success",
            "answer": "recovered report",
            "summary": "ok",
            "data": {"report": "final.md"},
        })
        executor = WorkflowExecutor(
            runtime=FakeRuntime({
                "nl2sql": nl2sql,
                "report_writer": report,
            })
        )
        route = decision(route_payload([
            call("step_1", "database_query", "nl2sql", "查询"),
            call(
                "step_2",
                "report_generation",
                "report_writer",
                "写报告",
                depends_on=["step_1"],
                input_refs=["step_1.data.structured"],
            ),
        ]))

        replan_infos = []

        def fake_replan(info):
            replan_infos.append(info)
            if info.get("attempt", 0) > 0:
                return None
            return decision(route_payload([
                call(
                    "rp0_step_2",
                    "report_generation",
                    "report_writer",
                    "基于已有结果重写报告",
                ),
            ]))

        result = executor.execute(
            route,
            original_query="查数据写报告",
            replan_fn=fake_replan,
        )

        # 失败步骤执行一次,重规划后的新步骤执行一次
        self.assertEqual(len(nl2sql.calls), 1)
        self.assertEqual(nl2sql.calls[0]["query"], "查询")
        self.assertEqual(len(report.calls), 1)
        self.assertEqual(report.calls[0]["query"], "基于已有结果重写报告")
        # 最终状态只按最终计划统计 → 成功
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["answer"], "recovered report")
        self.assertEqual(
            [entry["agent_id"] for entry in result["trace"]],
            ["nl2sql", "report_writer"],
        )
        # 钩子逐批询问是否重规划;第二次无失败应返回 None,不重复执行
        self.assertEqual(len(replan_infos), 2)
        self.assertIn("step_1", replan_infos[0]["batch_call_ids"])
        self.assertIn("rp0_step_2", replan_infos[1]["batch_call_ids"])
        self.assertEqual(
            result["results"]["step_1"]["status"], "error"
        )
        self.assertIn("rp0_step_2", result["results"])

    def test_replan_not_called_without_hook(self):
        nl2sql = RecordingAgent({
            "status": "error",
            "answer": "db down",
            "summary": "db down",
        })
        report = RecordingAgent({
            "status": "success",
            "answer": "independent",
            "summary": "ok",
        })
        executor = WorkflowExecutor(
            runtime=FakeRuntime({
                "nl2sql": nl2sql,
                "report_writer": report,
            })
        )
        route = decision(route_payload([
            call("step_1", "database_query", "nl2sql", "查询"),
            call(
                "step_2",
                "report_generation",
                "report_writer",
                "写独立说明",
                depends_on=["step_1"],
                input_refs=["step_1.data.structured"],
            ),
        ]))
        result = executor.execute(route, original_query="查询")
        # 无 replan_fn 时保持原有跳过行为
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["results"]["step_2"]["status"], "skipped")
        self.assertEqual(report.calls, [])




if __name__ == "__main__":
    unittest.main()
