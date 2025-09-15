"""Task description: question + answer generation from documents. Prompts are taken from the EuroLLM technical report https://arxiv.org/abs/2506.04079"""
from typing import Any, Dict, Generator, List, Union

from dispatcher.taskmanager.backend.request import Request, Response
from dispatcher.taskmanager.task.base import GeneratorTask
from dispatcher.taskmanager.task import TaskFailed

import random
import os
import re
import logging

__all__ = ["GenerateMathProblemsTask"]

ERROR_TYPES = {
    "problem_format": "missing_keyword",
}

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

class GenerateMathProblemsTask(GeneratorTask):
    """Generate math problems and solutions based on a persona."""
    
    ANSWER_GEN_PARAMS: Dict[str, Any] = {
        "temperature": 0.8,
        "top_p": 0.9,
        "max_tokens": 4096,
    }
    
    PROBLEM_GEN_PARAMS: Dict[str, Any] = {
        "temperature": 0.6,
        "top_p": 0.9,
        "max_tokens": 4096,
    }
    
    logger = logging.getLogger(__name__)
    
    def create_instruction_request(self, persona: str):
        prompt_template = open("model_prompts/generate_math_problem.txt").read().strip()
        prompt_text = prompt_template.format(
            persona=persona,
            language=LANGUAGE_NAMES.get(LANGUAGE, ["English", "eng"])[0],
        )
        input_messages = [
            {
                "role": "user",
                "content": prompt_text
            },
        ]
        return input_messages
    
    def extract_problem(self, problem_resp_text: str) -> tuple:
        # self.logger.info(f"\nPROBLEM RESPONSE:\n{problem_resp_text}\n")
        match = re.search(r'PROBLEM:\s*(.*)', problem_resp_text, re.DOTALL)
        if match is None:
            error_message = f"Could not find keyword PROBLEM"
            raise TaskFailed(
                message=error_message,
                error_type=ERROR_TYPES['problem_format']
            )
        problem_text = match.group(1).strip()
        # self.logger.info(f"\nEXTRACTED PROBLEM:\n{problem_text}\n")
        return problem_text
    

    def create_answer_request(self, problem: str):
        prompt_template = open("model_prompts/generate_math_answer.txt").read().strip()
        prompt_text = prompt_template.format(
                            math_problem=problem,
                            language=LANGUAGE_NAMES.get(LANGUAGE, ["English", "eng"])[0],
        )
        input_messages = [
            {
                "role": "user",
                "content": prompt_text,
            },
        ]
        return input_messages

    def extract_answer(self, answer_resp_text: str) -> tuple:
        match = re.search(r'FINAL ANSWER:\s*(.*)', answer_resp_text, re.DOTALL)
        if match is None:
            error_message = f"Could not find keyword FINAL ANSWER"
            raise TaskFailed(
                message=error_message,
                error_type=ERROR_TYPES['problem_format']
            )
        answer_text = match.group(1).strip()
        # self.logger.info(f"\nANSWER RESPONSE:\n{answer_resp_text}\n")
        # self.logger.info(f"\nEXTRACTED ANSWER:\n{answer_text}\n")
        return answer_text
    
    # --------------- generator ---------------
    def task_generator(self) -> Generator[Union[Request, List[Request]], Any, Dict[str, Any]]:
        # self.data is prepopulated with the data from the jsonl row being processed
        persona = self.data.get("professional_persona")
        persona_id = self.data.get("uuid") 
        return_dict = {
            "persona": persona,
            "persona_id": persona_id,
            "messages": [],
        }
        # Step 1 – Generate problem from persona
        input_messages = self.create_instruction_request(persona=persona)
        problem_resp: Response = yield Request({"messages": input_messages, **self.PROBLEM_GEN_PARAMS})
        # self.logger.info("\nPROBLEM TEXT:\n{problem_text}\n")
        problem_text = self.extract_problem(problem_resp.get_text())
        # Step 1.5 – Store the problem in the return_dict
        return_dict["messages"].append(
                    {
                        "role": "user",
                        "content": problem_text,
                    })
        # Step 2 – Generate answer to the problem
        input_messages = self.create_answer_request(problem=problem_text)
        answer_resp: Response = yield Request({"messages": input_messages, **self.ANSWER_GEN_PARAMS})
        answer_resp_text = answer_resp.get_text()
        # self.logger.info(f"\nANSWER RESPONSE:\n{answer_resp_text}\n")
        # Step 3.5 – Store the answer in the return_dict
        return_dict["messages"].append(
                    {
                        "role": "assistant",
                        "content": answer_resp_text,
                    })
        # # Step 4 - Judge answer quality
        # input_messages = self.create_judge_input(instruct_text, answer_text)
        # judge_resp = yield Request({"messages": input_messages, **self.JUDGE_PARAMS})
        # judge_resp_text = judge_resp.get_text()
        # answer_score = self.validate_judge_response(judge_resp_text)
        return return_dict