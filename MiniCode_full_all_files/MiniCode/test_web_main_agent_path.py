import unittest
import sys
from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock, MagicMock, patch

from fastapi import HTTPException

import web_server


class WebMainAgentPathTests(unittest.IsolatedAsyncioTestCase):
    def test_authenticated_principal_uses_stable_subject(self):
        first = SimpleNamespace(
            scope={
                "user": SimpleNamespace(
                    is_authenticated=True,
                    subject="immutable-1",
                    display_name="same name",
                ),
            },
            state=SimpleNamespace(),
        )
        second = SimpleNamespace(
            scope={
                "user": SimpleNamespace(
                    is_authenticated=True,
                    subject="immutable-2",
                    display_name="same name",
                ),
            },
            state=SimpleNamespace(),
        )

        self.assertNotEqual(
            web_server._request_user_id(first),
            web_server._request_user_id(second),
        )

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

            http_request = SimpleNamespace(
                scope={},
                state=SimpleNamespace(authenticated_user_id="principal:alice"),
            )
            result = await web_server.chat(request, http_request)

        self.assertEqual(result, routed)
        auto.assert_awaited_once()
        args = auto.await_args.args
        kwargs = auto.await_args.kwargs
        self.assertEqual(args[0], "@code 执行计算")
        self.assertEqual(kwargs["user_id"], "principal:alice")
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
                patch.object(web_server, "_resolve_web_session") as resolve, \
                patch.object(web_server, "_auto_route", AsyncMock(return_value=checkpoint)) as auto:
            response = await web_server._chat_stream_impl(
                "执行计算",
                "code",
                "session1",
                request=SimpleNamespace(
                    scope={},
                    client=None,
                    state=SimpleNamespace(
                        authenticated_user_id="principal:alice"
                    ),
                ),
                approved_call_ids=[],
            )
            chunks = []
            async for chunk in response.body_iterator:
                chunks.append(chunk.decode() if isinstance(chunk, bytes) else chunk)

        payload = "".join(chunks)
        self.assertIn("approval_checkpoint", payload)
        self.assertIn("step_1", payload)
        resolve.assert_called_once_with("principal:alice", "session1")
        auto.assert_awaited_once_with(
            "@code 执行计算",
            "session1",
            user_id="principal:alice",
            approved_call_ids=[],
            on_stage=ANY,
        )

    async def test_cleanup_checks_owner_before_removing_session_files(self):
        request = SimpleNamespace(
            scope={},
            state=SimpleNamespace(authenticated_user_id="principal:alice"),
        )
        payload = web_server.ChatRequest(session_id="owned-session")
        with patch.object(web_server, "_resolve_web_session") as resolve, \
                patch.object(web_server.memory, "cleanup_session") as cleanup:
            result = await web_server.cleanup_session_endpoint(payload, request)

        self.assertEqual(result["status"], "ok")
        resolve.assert_called_once_with("principal:alice", "owned-session")
        cleanup.assert_called_once_with("owned-session")

    async def test_crystallize_rejects_foreign_session_before_reading_history(self):
        request = SimpleNamespace(
            scope={},
            state=SimpleNamespace(authenticated_user_id="principal:bob"),
        )
        payload = web_server.ChatRequest(session_id="alice-session")
        crystallize = MagicMock()
        fake_module = SimpleNamespace(crystallize=crystallize)
        denied = HTTPException(status_code=403, detail="Session access denied")

        with patch.object(
            web_server,
            "_resolve_web_session",
            side_effect=denied,
        ), patch.dict(sys.modules, {"crystallize": fake_module}):
            with self.assertRaises(HTTPException) as raised:
                await web_server.crystallize_session(payload, request)

        self.assertEqual(raised.exception.status_code, 403)
        crystallize.assert_not_called()


if __name__ == "__main__":
    unittest.main()
