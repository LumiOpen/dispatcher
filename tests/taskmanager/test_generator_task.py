from __future__ import annotations

import random
import unittest
from typing import Any, Dict, List, Union, Generator

from dispatcher.taskmanager.task.base import GeneratorTask, TaskFailed, TaskRetry
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

        if self.mode == "retry":
            raise TaskRetry("retry requested from test")

        if self.mode == "yield_first":
            yield Request({"prompt": "test_prompt"})
            return {"status": "finished_normally"}

        if self.mode == "retry_after_yield":
            _ = yield Request({"prompt": "test_prompt"})
            raise TaskRetry("retry after receiving response")

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

    def test_retry_before_first_yield(self):
        """TaskRetry raised before first yield marks task done with should_retry()."""
        task_data = {"id": 789}
        task = MockGeneratorTaskForLifecycle(data=task_data, mode="retry", context="ctx")

        self.assertTrue(task.is_done())
        self.assertTrue(task.should_retry())
        self.assertEqual(task.retry_reason, "retry requested from test")
        _, ctx = task.get_result()
        self.assertEqual(ctx, "ctx")

    def test_retry_after_yield(self):
        """TaskRetry raised mid-flow (after yield+response) marks task done with should_retry()."""
        task_data = {"id": 101}
        task = MockGeneratorTaskForLifecycle(data=task_data, mode="retry_after_yield", context="ctx2")

        self.assertFalse(task.is_done(), "Task should not be done before processing response.")
        self.assertFalse(task.should_retry())

        # Feed a response to advance the generator past the yield
        req = task.get_next_request()
        self.assertIsNotNone(req)
        task.process_result(_create_success_response(req))

        self.assertTrue(task.is_done())
        self.assertTrue(task.should_retry())
        self.assertEqual(task.retry_reason, "retry after receiving response")
        _, ctx = task.get_result()
        self.assertEqual(ctx, "ctx2")

    def test_should_retry_false_for_normal_completion(self):
        """should_retry() returns False for a task that completes normally."""
        task = MockGeneratorTaskForLifecycle(data={}, mode="immediate_return")
        self.assertTrue(task.is_done())
        self.assertFalse(task.should_retry())

    def test_should_retry_false_for_task_failed(self):
        """should_retry() returns False for a task that raises TaskFailed."""
        task = MockGeneratorTaskForLifecycle(data={}, mode="immediate_fail")
        self.assertTrue(task.is_done())
        self.assertFalse(task.should_retry())


# ---------------------------------------------------------------------------
# Tests for Task.is_last_retry_attempt() and Task.build_result()
# ---------------------------------------------------------------------------

class _RetryCtx:
    """Stand-in for dispatcher.models.WorkItem with retry metadata."""
    def __init__(self, retry_count=0, max_retries=None):
        self.retry_count = retry_count
        self.max_retries = max_retries


def _make_task(data=None, context=None):
    """Build a minimal initialized Task instance to exercise the helpers."""
    return MockGeneratorTaskForLifecycle(
        data=data if data is not None else {},
        mode="immediate_return",
        context=context,
    )


class TestIsLastRetryAttempt(unittest.TestCase):
    """is_last_retry_attempt() reads retry_count / max_retries off self.context."""

    def test_returns_false_when_metadata_unavailable(self):
        # context=None, FileTaskSource-style dict context, missing max_retries,
        # and an explicit None retry_count must all return False without crashing.
        for ctx in (
            None,
            {"line_number": 5},
            _RetryCtx(retry_count=2, max_retries=None),
            _RetryCtx(retry_count=None, max_retries=3),
        ):
            with self.subTest(ctx=ctx):
                self.assertFalse(_make_task(context=ctx).is_last_retry_attempt())

    def test_returns_false_for_unlimited_retries(self):
        # max_retries == -1 means infinite retries; no "last" attempt exists.
        task = _make_task(context=_RetryCtx(retry_count=99, max_retries=-1))
        self.assertFalse(task.is_last_retry_attempt())

    def test_threshold_behavior(self):
        # False before the last attempt, True at the threshold, True past it
        # (defensive), and True when max_retries=0 means "no retries allowed".
        cases = [
            (2, 3, False),
            (3, 3, True),
            (5, 3, True),
            (0, 0, True),
        ]
        for retry_count, max_retries, expected in cases:
            with self.subTest(retry_count=retry_count, max_retries=max_retries):
                task = _make_task(context=_RetryCtx(retry_count, max_retries))
                self.assertEqual(task.is_last_retry_attempt(), expected)


class TestBuildResult(unittest.TestCase):
    """build_result() assembles the dispatcher's standard result schema."""

    def test_default_success_with_payload(self):
        # success defaults to True; data is spread in; payload is merged.
        task = _make_task(data={"prompt": "hi"})
        self.assertEqual(
            task.build_result(translated="HI", score=0.9),
            {"prompt": "hi", "success": True, "translated": "HI", "score": 0.9},
        )
        # Empty data still yields a valid minimal result.
        self.assertEqual(_make_task().build_result(), {"success": True})

    def test_failure_with_error_and_partial_payload(self):
        task = _make_task(data={"prompt": "hi"})
        self.assertEqual(
            task.build_result(success=False, error="boom", partial="HE"),
            {"prompt": "hi", "success": False, "error": "boom", "partial": "HE"},
        )
        # error=None must be omitted, not serialized as null.
        self.assertNotIn("error", task.build_result(success=False))

    def test_retry_metadata_included_with_edge_values(self):
        # retry_count=0 (first attempt) and max_retries=-1 (unlimited) are both
        # meaningful and must surface in the result, not be dropped by a truthy check.
        for retry_count, max_retries in [(2, 3), (0, 3), (5, -1)]:
            with self.subTest(retry_count=retry_count, max_retries=max_retries):
                task = _make_task(context=_RetryCtx(retry_count, max_retries))
                result = task.build_result()
                self.assertEqual(result["retry_count"], retry_count)
                self.assertEqual(result["max_retries"], max_retries)

    def test_retry_metadata_omitted_when_context_lacks_attrs(self):
        for ctx in (None, {"line_number": 0}):
            with self.subTest(ctx=ctx):
                result = _make_task(context=ctx).build_result()
                self.assertNotIn("retry_count", result)
                self.assertNotIn("max_retries", result)

    def test_payload_overrides_data_and_standard_fields(self):
        # Last-write-wins: caller payload trumps both self.data and the auto-filled
        # retry fields, so a task can override defaults when it has better info.
        task = _make_task(
            data={"prompt": "original"},
            context=_RetryCtx(retry_count=2, max_retries=3),
        )
        result = task.build_result(prompt="overridden", retry_count=999)
        self.assertEqual(result["prompt"], "overridden")
        self.assertEqual(result["retry_count"], 999)

    def test_success_and_error_are_keyword_only(self):
        # Guards against positional misuse like build_result(False, "oops")
        # silently swapping argument meaning.
        task = _make_task()
        with self.assertRaises(TypeError):
            task.build_result(False)  # type: ignore[misc]
        with self.assertRaises(TypeError):
            task.build_result(True, "oops")  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Exports for use in other test modules
# ---------------------------------------------------------------------------
__all__ = [
    "MockGeneratorTask",
    "_make_success_content",
    "_create_success_response",
    "_create_error_response",
]
