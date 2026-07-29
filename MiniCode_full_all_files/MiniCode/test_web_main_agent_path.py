import unittest
from unittest.mock import AsyncMock, patch

import web_server


class WebMainAgentPathTests(unittest.IsolatedAsyncioTestCase):
    def test_failed_startup_cannot_leave_permanent_wait_state(self):
        original = dict(web_server._dynamic_cache)
        try:
            web_server._dynamic_cache.clear()
            web_server._dynamic_cache["initializing"] = True
            with patch.object(web_server._startup_ready, "wait", return_value=False):
                with self.assertRaises(RuntimeError):
                    web_server._ensure_startup()
            self.assertFalse(web_server._dynamic_cache["initializing"])
        finally:
            web_server._dynamic_cache.clear()
            web_server._dynamic_cache.update(original)

    async def test_manual_chat_routes_through_main_agent(self):
        routed = {"status": "success", "answer": "ok", "agent_used": "code"}
        with patch.object(web_server, "get_agents", return_value={"code": object()}), \
                patch.object(web_server, "_auto_route", AsyncMock(return_value=routed)) as auto:
            request = web_server.ChatRequest(
                message="执行计算",
                agent="code",
                approved_call_ids=["step_1"],
            )

            result = await web_server.chat(request)

        self.assertEqual(result, routed)
        auto.assert_awaited_once()
        args = auto.await_args.args
        kwargs = auto.await_args.kwargs
        self.assertEqual(args[0], "@code 执行计算")
        self.assertEqual(kwargs["approved_call_ids"], ["step_1"])

    async def test_sse_is_presentation_over_main_agent_checkpoint(self):
        checkpoint = {
            "status": "checkpoint",
            "answer": "approval required",
            "pending_call": {"call_id": "step_1"},
        }
        with patch.object(web_server, "get_agents", return_value={"code": object()}), \
                patch.object(web_server.memory, "get_files", return_value=[]), \
                patch.object(web_server.memory, "build_agent_context", return_value=""), \
                patch.object(web_server, "_auto_route", AsyncMock(return_value=checkpoint)) as auto:
            response = await web_server._chat_stream_impl(
                "执行计算",
                "code",
                "session1",
                approved_call_ids=[],
            )
            chunks = []
            async for chunk in response.body_iterator:
                chunks.append(chunk.decode() if isinstance(chunk, bytes) else chunk)

        payload = "".join(chunks)
        self.assertIn("approval_checkpoint", payload)
        self.assertIn("step_1", payload)
        auto.assert_awaited_once_with(
            "@code 执行计算",
            "session1",
            approved_call_ids=[],
        )


if __name__ == "__main__":
    unittest.main()
