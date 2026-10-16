import unittest
from unittest.mock import patch

from mini_thalamus.agent_adapter import StandardAgentAdapter, adapt_agent, normalize_agent_result
from mini_thalamus.agent_runtime import AgentRuntime
from mini_thalamus.base_agent import AgentContext, AgentResult


class FakeAgent:
    agent_id = "fake"

    def run(self, query, context=None):
        return {"answer": "ok", "data": {"x": 1}}

    def search(self, query):
        return ["result"]


class PartialAgent:
    def run(self, query, context=None):
        return {
            "status": "partial",
            "answer": "partial but usable",
            "data": {"sources": [1]},
        }


class ContextProbeAgent:
    def __init__(self):
        self.seen = None

    def run(self, query, context=None):
        self.seen = context
        return {"answer": "ok"}


class AgentAdapterTests(unittest.TestCase):
    def test_normalizes_plain_dict_to_standard_result(self):
        wrapped = adapt_agent(FakeAgent(), agent_id="fake")
        result = wrapped.run("q")

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["answer"], "ok")
        self.assertEqual(result["metadata"]["agent_id"], "fake")

    def test_usable_partial_becomes_success_with_original_status(self):
        wrapped = adapt_agent(PartialAgent(), agent_id="websearch")
        result = wrapped.run("q")

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["metadata"]["original_status"], "partial")
        self.assertTrue(result["metadata"]["usable_partial"])

    def test_agent_context_is_converted_to_dict(self):
        probe = ContextProbeAgent()
        wrapped = adapt_agent(probe, agent_id="ctx")

        wrapped.run("q", AgentContext(current_query="q", session_id="s"))

        self.assertIsInstance(probe.seen, dict)
        self.assertEqual(probe.seen["session_id"], "s")

    def test_unknown_attributes_are_forwarded(self):
        wrapped = adapt_agent(FakeAgent(), agent_id="fake")

        self.assertEqual(wrapped.search("q"), ["result"])

    def test_agent_result_instance_is_accepted(self):
        result = normalize_agent_result(
            AgentResult(answer="x"),
            agent_id="a",
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["metadata"]["agent_id"], "a")

    def test_runtime_returns_standard_adapter(self):
        runtime = AgentRuntime()

        # websearch is the sample agent here, and it builds an OpenAI client in
        # __init__. The assertion is about the adapter the runtime wraps it in,
        # so the client is stubbed and this runs without an API key. Both names
        # are patched: the loader may import the module itself, in which case
        # its `from openai import OpenAI` is what has to resolve to the stub.
        with patch("openai.OpenAI"), patch(
            "mini_thalamus.agents.websearch_agent.OpenAI"
        ):
            runtime.discover()
            agent = runtime.get_agent("websearch")

        self.assertIsInstance(agent, StandardAgentAdapter)


if __name__ == "__main__":
    unittest.main()
