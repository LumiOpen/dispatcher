"""Example task – two responses + judge"""
from typing import Any, Dict, Generator, List, Union

from dispatcher.taskmanager.backend.request import Request, Response
from dispatcher.taskmanager.task import GeneratorTask, TaskFailed
import logging

__all__ = ["MultipleResponsesTask"]


class MultipleResponsesTask(GeneratorTask):
    logger = logging.getLogger(__name__)
    # Fixed generation hyper‑parameters for candidate answers
    GEN_PARAMS: Dict[str, Any] = {
        "temperature": 0.7,
        "top_p": 0.95,
        "max_tokens": 4096,
        "n": 16,  # response per request
    }

    # Deterministic parameters for the judge
    JUDGE_PARAMS: Dict[str, Any] = {
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 4096,
    }

    # --------------- generator ---------------
    def task_generator(self) -> Generator[Union[Request, List[Request]], Any, Dict[str, Any]]:
        # self.data is prepopulated with the data from the jsonl row being
        # processed
        messages = self.data.get("messages")

        # Step 1 – get n responses for prompt
        responses: List[Response] = yield [
            Request({"messages": messages, **self.GEN_PARAMS}),

        ]

        for i, resp in enumerate(responses):
            self.logger.info(f"\n\nRESP {i+1}: {resp}\n\n")

        return {
            "messages": messages,
            "responses": [resp.get_text() for resp in responses],
        }
