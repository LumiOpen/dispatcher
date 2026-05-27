"""
Task description: Translates reasoning traces line-by-line from generated reasoning answers.
"""
from typing import Any, Dict, Generator, List, Union
from functools import lru_cache

from dispatcher.taskmanager.backend.request import Request, Response
from dispatcher.taskmanager.task.base import GeneratorTask
from dispatcher.taskmanager.task import TaskFailed, TaskRetry

import os
import logging
import re

__all__ = ["InferenceTask"]

class InferenceTask(GeneratorTask):
    """Simple inference task"""

    GEN_PARAMS: Dict[str, Any] = {
        "temperature": 0.8,
        "top_p": 0.95,
        "max_tokens": 32768,
    	"extra_body": {
        	"chat_template_kwargs": {"enable_thinking": True},
        	"skip_special_tokens": False
    	}
    }

    logger = logging.getLogger(__name__)
    
    # --------------- generator ---------------
    def task_generator(self) -> Generator[Union[Request, List[Request]], Any, Dict[str, Any]]:
        # self.data is prepopulated with the data from the jsonl row being processed
        # Get the generated reasoning answer
        self.logger.info(f"[InferenceTask] ID:{self.data.get('prompt_id')} Processing sample")
        input_messages = self.data.get("messages", [])

        resp: Response = yield Request({"messages": input_messages, **self.GEN_PARAMS})
       
        if not resp.is_success:
            self.logger.error(
                "[InferenceTask] ID:%s Backend request failed: %s",
                self.data.get("prompt_id"),
                resp.error,
            )
            raise TaskFailed(
                message=f"Backend request failed: {resp.error}",
                error_type="response_generation_error",
            )

        response_text = resp.get_text()
        if response_text is None:
            self.logger.error(
                "[InferenceTask] ID:%s Backend response had no extractable text payload",
                self.data.get("prompt_id"),
            )
            raise TaskFailed(
                message="Backend response had no extractable text payload",
                error_type="response_parsing_error",
            )

        response = response_text.strip()

        self.logger.info(
            f"[InferenceTask] ID:{self.data.get('prompt_id')} Finished processing sample"
        )

        output_messages = list(input_messages) + [{"role": "assistant", "content": response}]
        # build_result spreads self.data, sets success=True, then merges payload.
        return self.build_result(
            response=response,
            output_messages=output_messages,
        )
