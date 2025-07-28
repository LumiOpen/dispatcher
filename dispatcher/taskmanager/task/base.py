"""Core task abstractions.

`Task` is the minimal interface every user‑defined task must satisfy.
`GeneratorTask` is a convenience wrapper that drives a Python generator so the
user can `yield` requests and receive responses in an intuitive way.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union, Generator

from ..backend.request import Request, Response


class TaskFailed(Exception):
    """
    Exception to be raised by a task to signal a controlled failure.
    
    This allows a task to terminate itself gracefully and provide a structured
    error payload that will be recorded as its final result.
    """
    def __init__(self, message: str, error_type: str = "task_logic_error"):
        self.message = message
        self.error_type = error_type
        super().__init__(f"[{self.error_type}] {self.message}")


###############################################################################
# Abstract base task
###############################################################################

class Task(ABC):
    """Minimal contract any task must satisfy."""

    def __init__(self, data: Dict[str, Any], context: Any = None):
        self.data = data
        self.context = context

    # -- lifecycle ---------------------------------------------------------

    @abstractmethod
    def get_next_request(self) -> Optional[Request]:
        """Return one pending `Request` or *None* if no work is ready *now*."""

    @abstractmethod
    def process_result(self, response: Response) -> None:
        """Receive a `Response` from the backend."""

    @abstractmethod
    def is_done(self) -> bool:
        """True when the task has produced its final result."""

    @abstractmethod
    def get_result(self) -> Tuple[Dict[str, Any], Any]:
        """Return (result, original_context)."""

###############################################################################
# Generator‑powered task helper
###############################################################################

class GeneratorTask(Task):
    """Drive task logic with a generator interface.

    * Yield a single :class:`Request` or a list of requests.
    * Receive back a single :class:`Response` or list of responses (arrival order).
    * **Must** ``return`` a result dict; cases where no value is returned are
      treated as errors.
    """

    # ------------------------------------------------------------------
    # Construction & generator bootstrap
    # ------------------------------------------------------------------

    def __init__(self, data: Dict[str, Any], context: Any = None):
        super().__init__(data, context)

        self._pending_reqs: List[Request] = []       # queue for TaskManager
        self._awaiting_responses: int = 0            # outstanding count
        self._collected: List[Response] = []         # responses for current yield
        self._result: Optional[Dict[str, Any]] = None

        # Start the generator and get the first yielded request.
        self._gen = self.task_generator()
        try:
            first = next(self._gen)
            self._enqueue(first)
        except StopIteration as stop:
            # Handle the edge case where the generator returns immediately
            # without yielding any requests.
            if not hasattr(stop, "value") or stop.value is None:
                raise RuntimeError(
                    "task_generator finished on first call without returning a result dict"
                ) from None
            self._result = stop.value
        except TaskFailed as e:
            # Handle a deliberate failure during initialization.
            self._result = {
                "__ERROR__": {
                    "error": e.error_type,
                    "message": e.message,
                    "task_data": self.data
                }
            }


    # ------------------------------------------------------------------
    # User‑supplied coroutine signature
    # ------------------------------------------------------------------

    @abstractmethod
    def task_generator(self) -> Generator[Union[Request, List[Request]], Any, Dict[str, Any]]:  # noqa: D401
        """Implement the task flow using `yield`/`return`."""

    # ------------------------------------------------------------------
    # Task API consumed by TaskManager
    # ------------------------------------------------------------------

    def get_next_request(self) -> Optional[Request]:
        return self._pending_reqs.pop(0) if self._pending_reqs else None

    def process_result(self, response: Response) -> None:
        self._collected.append(response)
        self._awaiting_responses -= 1
        if self._awaiting_responses == 0 and self._result is None:
            self._advance_generator()

    def is_done(self) -> bool:
        """Generator has returned its final result."""
        return self._result is not None

    def get_result(self) -> Tuple[Dict[str, Any], Any]:
        if self._result is None:
            raise RuntimeError("Task finished without returning a result dictionary")
        return self._result, self.context

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _enqueue(self, yielded: Union[Request, List[Request], None]):
        if yielded is None:
            raise ValueError("GeneratorTask yielded None")
        if isinstance(yielded, list):
            if not yielded:
                raise ValueError("GeneratorTask yielded an empty list")
            self._pending_reqs.extend(yielded)
            self._awaiting_responses += len(yielded)
        else:
            self._pending_reqs.append(yielded)
            self._awaiting_responses += 1

    def _advance_generator(self):
        arg: Union[Response, List[Response]]
        if len(self._collected) == 1:
            arg = self._collected[0]
        else:
            arg = self._collected.copy()
        self._collected.clear()

        try:
            yielded = self._gen.send(arg)
            self._enqueue(yielded)
        except StopIteration as stop:
            if not hasattr(stop, "value") or stop.value is None:
                raise RuntimeError(
                    "task_generator finished without returning a result dict"
                ) from None
            self._result = stop.value
        except TaskFailed as e:
            # The task has deliberately failed. Set its final result to the error.
            self._result = {
                "__ERROR__": {
                    "error": e.error_type,
                    "message": e.message,
                    "task_data": self.data
                }
            }
