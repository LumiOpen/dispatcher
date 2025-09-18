"""Task description: question + answer generation from documents. Prompts are taken from the EuroLLM technical report https://arxiv.org/abs/2506.04079"""
from typing import Any, Dict, Generator, List, Union

from dispatcher.taskmanager.backend.request import Request, Response
from dispatcher.taskmanager.task.base import GeneratorTask
from dispatcher.taskmanager.task import TaskFailed
from utils.lang_id import detect_language

import random
import os
import re
import logging

__all__ = ["LongMagpieTask"]

QWEN_USER_START = "\n<|im_start|>user\n"
QWEN_ASSISTANT_START = "\n<|im_start|>assistant\n"
QWEN_END = "<|im_end|>"

LLAMA_USER_START = "<|start_header_id|>user<|end_header_id|>\n\n"
LLAMA_ASSISTANT_START = "<|start_header_id|>assistant<|end_header_id|>\n\n"
LLAMA_END = "<|eot_id|>"

LANGUAGE=os.environ.get("LANGUAGE")
MODEL=os.environ.get("MODEL")
TURNS=int(os.environ.get("TURNS", 1))

MIN_DOC_LEN = 100
MAX_DOC_LEN = 30000

ERROR_TYPES = {
    "invalid_query": "model did not produce a valid query",
}



class LongMagpieTask(GeneratorTask):
    """Long Magpie implementation."""
    
    ANSWER_GEN_PARAMS: Dict[str, Any] = {
        "temperature": 0.8,
        "top_p": 0.9,
        "max_tokens": 4096,
    }
    
    INSTRUCTION_GEN_PARAMS: Dict[str, Any] = {
        "temperature": 0.9,
        "top_p": 0.9,
        "max_tokens": 4096,
    }
    
    logger = logging.getLogger(__name__)
    
    def create_query_generation_request(self, document: str, messages: List[Dict[str, str]] = None):
        if len(messages) == 0:
            if "qwen" in MODEL.lower():
                prompt_text = document + QWEN_USER_START
            else:
                prompt_text = document + LLAMA_USER_START
        else:
            prompt_text = document
            for turn, message in enumerate(messages):
                if "qwen" in MODEL.lower():
                    if message["role"] == "user":
                        prompt_text += QWEN_USER_START + message["content"] + QWEN_END
                    elif message["role"] == "assistant":
                        prompt_text += QWEN_ASSISTANT_START + message["content"] + QWEN_END
                    prompt_text += QWEN_USER_START
                else:
                    if message["role"] == "user":
                        prompt_text += LLAMA_USER_START + message["content"] + LLAMA_END
                    elif message["role"] == "assistant":
                        prompt_text += LLAMA_ASSISTANT_START + message["content"] + LLAMA_END
                    prompt_text += LLAMA_USER_START
        return prompt_text
    
    def create_response_generation_request(self, document: str, valid_query: str):
        input_messages = [
            {
                "role": "user",
                "content": document + "\n\n" + valid_query,
            },
        ]
        return input_messages
    
    # --------------- generator ---------------
    def task_generator(self) -> Generator[Union[Request, List[Request]], Any, Dict[str, Any]]:
        # self.data is prepopulated with the data from the jsonl row being processed
        document = self.data.get("text")
        doc_id = self.data.get("id")
        doc_url = self.data.get("url")
        return_dict = {
                        "id": doc_id,
                        "doc_url": doc_url,
                        "document": document,
                        # "query": "",
                        # "response": "",
                        "messages": [],
                    }
        for turn in range(TURNS):
            # Step 1 – Generate instruction from a document
            prompt = self.create_query_generation_request(document, return_dict["messages"])
            #self.logger.info(f"\n\nPROMPT: {prompt}\n")
            query_resp: Response = yield Request({"prompt": prompt, **self.INSTRUCTION_GEN_PARAMS})
            query_resp_text = query_resp.get_text()
            if query_resp_text is None:
                error_message = ERROR_TYPES['invalid_query']
                raise TaskFailed(
                    message=error_message,
                    error_type=ERROR_TYPES['invalid_query']
                )
            query_lines = query_resp_text.splitlines()
            self.logger.info(f"\n\nTURN {turn+1} RAW QUERY: {query_resp_text}\n")
            valid_query = query_lines[0].strip() if query_lines[0][-1] == '?' else ""
            if len(valid_query) == 0:
                error_message = ERROR_TYPES['invalid_query']
                raise TaskFailed(
                    message=error_message,
                    error_type=ERROR_TYPES['invalid_query']
                )
            self.logger.info(f"\n\nTURN {turn+1} QUERY: {valid_query}\n")
            return_dict["messages"].append({"role": "user", 
                                            "content": valid_query})
            # Step 2 – Generate response to the instruction
            input_messages = self.create_response_generation_request(document, valid_query)
            answer_resp: Response = yield Request({"messages": input_messages, **self.ANSWER_GEN_PARAMS})
            answer_text = answer_resp.get_text()
            return_dict["messages"].append({"role": "assistant", 
                                            "content": answer_text})
            self.logger.info(f"\n\nTURN {turn+1} RESPONSE: {answer_text}\n")
        return return_dict