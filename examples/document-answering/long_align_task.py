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

__all__ = ["LongAlignTask"]

LANGUAGE=os.environ.get("LANGUAGE")

ERROR_TYPES = {
    "invalid_query": "model did not produce a valid query",
}

TASK_TYPES = {
    1: "general",
    2: "summary",
    3: "reasoning",
    4: "information_extraction",
}

LANGUAGE_NAMES = {
    "bg": ["Bulgarian", "bul"],
    "cs": ["Czech", "ces"],
    "da": ["Danish", "dan"],
    "de": ["German", "deu"],
    "el": ["Greek", "ell"],
    "en": ["English", "eng"],
    "es": ["Spanish", "spa"],
    "et": ["Estonian", "est"],
    "fi": ["Finnish", "fin"],
    "fr": ["French", "fra"],
    "ga": ["Irish", "gle"],
    "hr": ["Croatian", "hrv"],
    "hu": ["Hungarian", "hun"],
    "it": ["Italian", "ita"],
    "lt": ["Lithuanian", "lit"],
    "lv": ["Latvian", "lav"],
    "mt": ["Maltese", "mlt"],
    "nl": ["Dutch", "nld"],
    "pl": ["Polish", "pol"],
    "pt": ["Portuguese", "por"],
    "ro": ["Romanian", "ron"],
    "sk": ["Slovak", "slk"],
    "sl": ["Slovenian", "slv"],
    "sv": ["Swedish", "swe"],
    "is": ["Icelandic", "isl"],
    "no": ["Norwegian", "nob"],
}

class LongAlignTask(GeneratorTask):
    """Long Align implementation."""
    
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
    
    def create_instruction_request(self, document: str, task_type: str):
        prompt_template = open(f"model_prompts/long_align_{task_type}_prompt.txt").read().strip()
        prompt_text = prompt_template.format(
                    document=document,
                    language=LANGUAGE_NAMES.get(LANGUAGE, ["English", "eng"])[0],
                    )
        input_messages = [
            {
                "role": "user",
                "content": prompt_text
            },
        ]
        return input_messages
    
    def create_response_generation_request(self, document: str, question: str):
        input_messages = [
            {
                "role": "user",
                "content": document + "\n\n" + question,
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
                        "query": "",
                        "response": "",
                    }
        # Step 0.5 - Randomly select the task type
        task_type = TASK_TYPES[random.randint(1, len(TASK_TYPES))]
        # self.logger.info(f"TASK: {task_type}")
        # Step 1 – Generate instruction from a document
        messages = self.create_instruction_request(document, task_type)
        query_resp: Response = yield Request({"messages": messages, **self.INSTRUCTION_GEN_PARAMS})
        query_resp_text = query_resp.get_text()
        if query_resp_text is None:
            error_message = ERROR_TYPES['invalid_query']
            raise TaskFailed(
                message=error_message,
                error_type=ERROR_TYPES['invalid_query']
            )
        questions = query_resp_text.splitlines()
        # Step 1.5 - Randomly select one question
        selected_question = questions[random.randint(0, len(questions)-1)]
        selected_question = re.sub(r'^\d+[:.]\s*', '', selected_question).strip()  # remove leading numbering
        return_dict["query"] = selected_question
        # Step 2 – Generate response to the instruction
        input_messages = self.create_response_generation_request(document, selected_question)
        answer_resp: Response = yield Request({"messages": input_messages, **self.ANSWER_GEN_PARAMS})
        answer_text = answer_resp.get_text()
        return_dict["response"] = answer_text
        self.logger.info(f"\n\nTASK: {task_type}\nQUESTION: {selected_question}\nANSWER: {answer_text}\n")
        return return_dict