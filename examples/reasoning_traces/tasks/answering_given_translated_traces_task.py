"""
Task description: Generate answers given the translated traces (generated in English by DeepSeek-R1 and translated to Finnish by DeepSeek-V3)
"""
import logging
import os
from functools import lru_cache
from typing import Any, Dict, Generator, List, Union

from dispatcher.taskmanager.backend.request import Request, Response
from dispatcher.taskmanager.task.base import GeneratorTask, TaskFailed

__all__ = ["AnsweringGivenTranslatedTracesTask"]

MODEL = os.environ.get("MODEL")

class AnsweringGivenTranslatedTracesTask(GeneratorTask):
    """Reasoning trace + answer generation."""

    ANSWER_GEN_PARAMS: Dict[str, Any] = {
        "temperature": 0.6,
        "top_p": 0.95,
        "max_tokens": 2048,  # Leave 63K tokens for the prompt+traces
    }

    logger = logging.getLogger(__name__)

    @staticmethod
    @lru_cache(maxsize=1)
    def get_tokenizer():
        # Import here to avoid global dependency if not used
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(
            MODEL,
            trust_remote_code=True,
        )

    def task_generator(self) -> Generator[Union[Request, List[Request]], Any, Dict[str, Any]]:
        # self.data is prepopulated with the data from the jsonl row being processed
        return_dict = self.data.copy()
        self.logger.info(f"Processing sample id: {self.data.get('id')}")
        tokenizer = self.get_tokenizer()
        answers = []
        for trace in self.data["translated_traces"]:
            input_messages = [
                {
                    "role": "user",
                    "content": self.data["generated_translation"] # Finnish question
                },
                {
                    "role": "assistant",
                    "reasoning_content": trace, # Finnish reasoning traces
                }
            ]
        
            # Render the chat template, leave the assistant turn open
            rendered = tokenizer.apply_chat_template(
                input_messages,
                tokenize=False,
                add_generation_prompt=False,
                # continue_final_message=True, # this argument exists in the API but qwen template does not use it
            )
            # Because the template does not implement continue_final_message we must trim the final hard-coded <|im_end|> token manually
            rendered = rendered.rsplit("<|im_end|>", 1)[0].rstrip()
            # print for debugging just the very first time
            self.logger.info(f"Rendered prompt: {rendered}")
            # Request text completion from vLLM backend
            req_dict = {
                "prompt": rendered,
                **self.ANSWER_GEN_PARAMS,
            }
            answer_resp: Response = yield Request(req_dict)
            answers.append(answer_resp.get_text().strip() if answer_resp.get_text() else "")
        return_dict["generated_solution_given_translated_traces"] = answers
        return return_dict