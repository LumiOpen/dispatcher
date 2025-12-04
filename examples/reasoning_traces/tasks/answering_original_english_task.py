"""
Task description: Generates answers with a reasoning (thinking) model given the original English question and the English reasoning traces (generated with R1) as input.
"""
import logging
import os
from functools import lru_cache
from typing import Any, Dict, Generator, List, Union

from dispatcher.taskmanager.backend.request import Request, Response
from dispatcher.taskmanager.task.base import GeneratorTask, TaskFailed

__all__ = ["AnsweringOriginalEnglishTask"]

MODEL = os.environ.get("MODEL")

class AnsweringOriginalEnglishTask(GeneratorTask):
    """Reasoning trace + answer generation."""

    ANSWER_GEN_PARAMS: Dict[str, Any] = {
        "temperature": 0.6,
        "top_p": 0.95,
        "max_tokens": 2048,  # Leave 30K tokens for the prompt+traces
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
        # preprocess - get R1 answers from "generated_reasoning_answer" list and split each answer by the end </think> tag - use the first part as the traces input
        traces = []
        for answer in self.data["generated_reasoning_answer"]:
            traces.append(answer.split("</think>")[0])
        if len(traces) == 0:
            raise TaskFailed(
                message=f"No traces found for sample {self.data.get('id')}",
                error_type="no_traces_found"
            )
        self.logger.info(f"Found {len(traces)} traces for sample {self.data.get('id')}")

        # Generate answers for each trace
        tokenizer = self.get_tokenizer()
        answers = []
        for trace in traces:
            input_messages = [
                {
                    "role": "user",
                    "content": self.data["text"] # English question
                },
                {
                    "role": "assistant",
                    "reasoning_content": trace, # English reasoning traces
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
            # print for debugging
            self.logger.info(f"Rendered prompt (sample {self.data.get('id')}): {rendered}")
            # Request text completion from vLLM backend
            req_dict = {
                "prompt": rendered,
                **self.ANSWER_GEN_PARAMS,
            }
            answer_resp: Response = yield Request(req_dict)
            answers.append(answer_resp.get_text())

        return_dict["generated_solution_given_original_english"] = answers
        return return_dict