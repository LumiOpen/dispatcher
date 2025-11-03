"""
Task description: Generates answers with a reasoning (thinking) model given the question and the ("pre-generated") reasoning traces as input.
"""
from typing import Any, Dict, Generator, List, Union

from dispatcher.taskmanager.backend.request import Request, Response
from dispatcher.taskmanager.task.base import GeneratorTask, TaskFailed

import os
import logging

__all__ = ["AnsweringWithTracesTask"]

LANGUAGE=os.environ.get("LANGUAGE")

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

class AnsweringWithTracesTask(GeneratorTask):
    """Reasoning trace + answer generation."""
    
    ANSWER_GEN_PARAMS: Dict[str, Any] = {
        "temperature": 0.6,
        "top_p": 0.95,
        "max_tokens": 2048, # Leave 14K tokens for the prompt+traces
    }

    logger = logging.getLogger(__name__)
    
    # --------------- generator ---------------
    def task_generator(self) -> Generator[Union[Request, List[Request]], Any, Dict[str, Any]]:
        # self.data is prepopulated with the data from the jsonl row being processed
        # We return the original data along with the generated answer
        return_dict = self.data.copy()

        # preprocess traces - extract from <think></think> tags
        # we have 4 traces generated for each sample
        traces = []
        for a in self.data["generated_answers"]:
            if '<think>' in a and '</think>' in a:
                trace = a.split('<think>')[1].split('</think>')[0]
                traces.append(trace)
            else:
                print("Missing <think> tags in generated answer.")
        if len(traces) == 0:
            raise TaskFailed(
                message=f"No traces found for sample {self.data.get("id")}",
                error_type="no_traces_found"
            )

        answers = []
        for t in traces:
            input_messages = [
                {
                    "role": "user", 
                    "content": self.data["generated_translation"]
                },
                {
                    "role": "assistant", 
                    "reasoning_content": traces,
                }
            ]
            
            answer_resp: Response = yield Request({"messages": input_messages, **self.ANSWER_GEN_PARAMS})
            answers.append(answer_resp.get_text().strip())

        return_dict["generated_answers_with_traces"] = answers
        return return_dict
