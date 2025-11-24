"""
Task description: Generates answers with a reasoning (thinking) model given the question as input.
"""
import logging
import os
from typing import Any, Dict, Generator, List, Union

from dispatcher.taskmanager.backend.request import Request, Response
from dispatcher.taskmanager.task.base import GeneratorTask

__all__ = ["ReasoningAnswerTask"]

class ReasoningAnswerTask(GeneratorTask):
    """Reasoning answer generation."""

    REASONING_ANSWER_GEN_PARAMS: Dict[str, Any] = {
        "temperature": 0.6,
        "top_p": 0.95,
        "max_tokens": 14336,  # Leave 2K tokens for the prompt
    }

    logger = logging.getLogger(__name__)

    def task_generator(self) -> Generator[Union[Request, List[Request]], Any, Dict[str, Any]]:
        # self.data is prepopulated with the data from the jsonl row being processed
        return_dict = self.data.copy()
        self.logger.info(f"Processing sample id: {self.data.get('id')}")

        input_messages = [
            {
                "role": "user",
                "content": self.data["generated_translation"]
            }
        ]

        # Request reasoning answer from vLLM backend
        req_dict = {
            "messages": input_messages,
            **self.REASONING_ANSWER_GEN_PARAMS,
        }
        reasoning_answer_resp: Response = yield Request(req_dict)
        return_dict["generated_reasoning_answer"] = reasoning_answer_resp.get_text().strip()
        return return_dict