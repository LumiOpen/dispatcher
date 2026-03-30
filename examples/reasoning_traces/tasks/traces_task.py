"""
Task description: Generates (reasoning) traces and answers for existing prompts.
"""
from typing import Any, Dict, Generator, List, Union

from dispatcher.taskmanager.backend.request import Request, Response
from dispatcher.taskmanager.task.base import GeneratorTask
from dispatcher.taskmanager.task import TaskFailed

import os
import logging

__all__ = ["TracesTask"]

LANGUAGE=os.environ.get("LANGUAGE")

ERROR_TYPES = {
    "invalid_query": "model did not produce a valid query",
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

OPEN_R1_REASONING_PROMPT = """
You are a reasoning model to help users solve complex problems. Your role as an assistant involves thoroughly exploring questions through a systematic thinking process before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered thinking process. 
Please structure your response into two main sections: 
- Thought and Solution using the specified format: <think> [Thought section] </think> [Solution section]. 
- In the Thought section, detail your reasoning process in steps. Each step should include detailed considerations such as analysing questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps. 
- In the Solution section, based on various attempts, explorations, and reflections from the Thought section, systematically present the final solution that you deem correct. The Solution section should be logical, accurate, and concise and detail necessary steps needed to reach the conclusion. Do not use <solution></solution> tags or the word "Solution". Just append the answer after the Thought section.
- Since the question is in {language}, you MUST respond entirely in {language} for both the Thought and Solution sections.

Now, try to solve the following question through the above guidelines.
{question}
"""

class TracesTask(GeneratorTask):
    """Reasoning trace + answer generation."""
    
    TRACES_GEN_PARAMS: Dict[str, Any] = {
        "temperature": 0.6,
        "top_p": 0.95,
        "max_tokens": 14336, # Leave 2K tokens for the prompt
        "n": 4,
    }

    logger = logging.getLogger(__name__)

    
    # --------------- generator ---------------
    def task_generator(self) -> Generator[Union[Request, List[Request]], Any, Dict[str, Any]]:
        # self.data is prepopulated with the data from the jsonl row being processed
        # We return the original data along with the generated answer
        return_dict = self.data.copy()

        input_messages = [
            {
                "role": "user", 
                "content": OPEN_R1_REASONING_PROMPT.format(language=LANGUAGE_NAMES.get(LANGUAGE)[0], 
                                                          question=self.data["generated_translation"])
            }
        ]
        # We request multiple answers per prompt
        answer_resp: Response = yield Request({"messages": input_messages, **self.TRACES_GEN_PARAMS})
        return_dict["generated_traces"] = answer_resp.get_text(n=4)
        return return_dict
