from __future__ import annotations

import random
import unittest
from typing import Any, Dict, List, Union, Generator

from dispatcher.taskmanager.task.base import GeneratorTask, TaskFailed
from dispatcher.taskmanager.backend.request import Request, Response

# ---------------------------------------------------------------------------
# Seed RNG for deterministic behaviour in the test‑suite.  Using a fixed seed
# guarantees that, with failure_rate=0.5 and 3 requests, at least one request
# will fail – satisfying the integration‑test expectation.
# ---------------------------------------------------------------------------
random.seed(0)

# ---------------------------------------------------------------------------
# Helpers to fabricate mock backend responses.
# ---------------------------------------------------------------------------

def _make_success_content(label: str, mode: str = "chat") -> Dict[str, Any]:
    """Return a stub payload that looks like an OpenAI / vLLM response."""
    if mode == "chat":
        return {
            "id": "chatcmpl-mock",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": f"Success for {label}"},
                }
            ],
            "model": "mock_model",
            "created": 123,
        }
    if mode == "text":
        return {
            "id": "cmpl-mock",
            "choices": [
                {
                    "text": f"Success for {label}",
                    "index": 0,
                    "finish_reason": "stop",
                }
            ],
            "model": "mock_model",
            "created": 123,
        }
    return {"result": f"Result for {label}"}


def _create_success_response(request: Request, mode: str = "chat") -> Response:
    prompt_txt = request.content.get("prompt", "?")
    return Response(request=request, content=_make_success_content(prompt_txt, mode))


def _create_error_response(request: Request) -> Response:
    msg = f"Simulated failure for request: {request.content}"
    return Response.from_error(request, RuntimeError(msg))


# ---------------------------------------------------------------------------
# Mock Task for Integration Testing
# ---------------------------------------------------------------------------

class MockGeneratorTask(GeneratorTask):
    """Supports the five generator modes exercised by the test‑suite."""

    def __init__(self, data: Dict[str, Any], context: Any = None, mode: str = "single"):
        self.mode = mode
        super().__init__(data, context)

    @staticmethod
    def _to_text(resp: Response) -> Union[str, Dict[str, Any]]:
        """Coerce *any* Response into a value expected by the tests."""
        if resp.is_success:
            txt = resp.get_text()
            if txt is not None:
                return txt
            return resp.content
        return {"error": str(resp.error)}

    def task_generator(self) -> Generator[Union[Request, List[Request]], Any, Dict[str, Any]]:
        if self.mode == "empty":
            _ = yield []
            return {"status": "empty"}

        if self.mode == "single":
            prompt = self.data.get("prompt1", self.data.get("p", "p1"))
            resp: Response = yield Request({"prompt": prompt})
            return {"final": self._to_text(resp), "source": "single"}

        if self.mode == "sequential":
            p1 = self.data.get("prompt1", self.data.get("p1", "first"))
            p2 = self.data.get("prompt2", self.data.get("p2", "second"))
            resp1: Response = yield Request({"prompt": p1})
            v1 = self._to_text(resp1)
            resp2: Response = yield Request({"prompt": f"Based on {v1}, ask: {p2}"})
            return {"step1": v1, "step2": self._to_text(resp2), "source": "sequential"}

        if self.mode == "batch":
            pa = self.data.get("prompt_a", self.data.get("a", "A"))
            pb = self.data.get("prompt_b", self.data.get("b", "B"))
            resps: List[Response] = yield [Request({"prompt": pa}), Request({"prompt": pb})]
            batch_res = [self._to_text(r) for r in resps]
            return {"batch": batch_res, "final_batch": batch_res, "source": "batch"}

        if self.mode == "single_error":
            resp: Response = yield Request({"prompt": "err"})
            return {"error": str(resp.error) if not resp.is_success else "unexpected"}

        if self.mode == "batch_mixed":
            pa = self.data.get("prompt_a", "A")
            pb = self.data.get("prompt_b", "B")
            resps: List[Response] = yield [Request({"prompt": pa}), Request({"prompt": pb})]
            mixed = [self._to_text(r) for r in resps]
            return {"mixed_results": mixed, "source": "batch_mixed"}

        raise ValueError("unknown mode")
        yield


# ---------------------------------------------------------------------------
# Mock Task and Tests for Unit Testing the Constructor/Lifecycle
# ---------------------------------------------------------------------------

class MockGeneratorTaskForLifecycle(GeneratorTask):
    """A mock task specifically for testing initialization behavior."""

    def __init__(self, data: Dict[str, Any], mode: str, context: Any = None):
        self.mode = mode
        super().__init__(data, context)

    def task_generator(self) -> Generator[Union[Request, List[Request]], Any, Dict[str, Any]]:
        if self.mode == "immediate_return":
            return {"status": "immediate_return", "data": self.data}

        if self.mode == "immediate_fail":
            raise TaskFailed(message="immediate failure from test")

        if self.mode == "yield_first":
            yield Request({"prompt": "test_prompt"})
            return {"status": "finished_normally"}

        yield # Unreachable, for linter


class TestGeneratorTaskLifecycle(unittest.TestCase):
    """
    Unit tests for the GeneratorTask's initialization and state transitions.
    """

    def test_initializes_correctly_when_yielding_first(self):
        """
        Ensures a standard task that yields at least once initializes correctly
        and is not immediately marked as done.
        """
        task = MockGeneratorTaskForLifecycle(data={}, mode="yield_first")

        self.assertFalse(task.is_done(), "Task should not be done immediately after initialization.")
        self.assertIsNotNone(task.get_next_request(), "Task should have a pending request after initialization.")

    def test_handles_generator_that_returns_before_first_yield(self):
        """
        Verifies a task that returns a value before its first yield is handled
        gracefully, is marked as done, and contains the correct final result.
        """
        context = "context_for_immediate_return"
        task_data = {"id": 123}

        task = MockGeneratorTaskForLifecycle(
            data=task_data,
            context=context,
            mode="immediate_return"
        )

        self.assertTrue(task.is_done(), "Task that returns immediately should be marked as done.")
        result, res_context = task.get_result()

        self.assertEqual(res_context, context)
        self.assertEqual(result, {"status": "immediate_return", "data": task_data})
        self.assertIsNone(task.get_next_request(), "A finished task should have no more requests.")

    def test_handles_generator_that_fails_before_first_yield(self):
        """
        Verifies a task that raises TaskFailed before its first yield is handled
        gracefully, is marked as done, and contains a valid error payload.
        """
        context = "context_for_immediate_fail"
        task_data = {"id": 456}

        task = MockGeneratorTaskForLifecycle(
            data=task_data,
            context=context,
            mode="immediate_fail"
        )

        self.assertTrue(task.is_done(), "Task that fails immediately should be marked as done.")
        result, res_context = task.get_result()

        self.assertEqual(res_context, context)
        self.assertIn("__ERROR__", result)
        self.assertEqual(result["__ERROR__"]["message"], "immediate failure from test")
        self.assertEqual(result["__ERROR__"]["task_data"], task_data)


# ---------------------------------------------------------------------------
# Exports for use in other test modules
# ---------------------------------------------------------------------------
__all__ = [
    "MockGeneratorTask",
    "_make_success_content",
    "_create_success_response",
    "_create_error_response",
]
