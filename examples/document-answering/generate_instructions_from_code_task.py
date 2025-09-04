"""Task description: question + answer generation from documents. Prompts are taken from the EuroLLM technical report https://arxiv.org/abs/2506.04079"""
from typing import Any, Dict, Generator, List, Union

from dispatcher.taskmanager.backend.request import Request, Response
from dispatcher.taskmanager.task.base import GeneratorTask
from dispatcher.taskmanager.task import TaskFailed

import random
import os
import re
import logging


__all__ = ["GenerateCodeProblemsTask"]

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


class GenerateCodeProblemsTask(GeneratorTask):
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

    def extract_code_from_text(self, text: str):
        pattern = re.compile(r"```(\w+)?\n(.*?)```", re.DOTALL)
        for lang, code in pattern.findall(text):
            if lang is not None and len(lang)>0 and code is not None and len(code)>0:
                # self.logger.info(f"\nEXTRACTED CODE:\n{code}\n")
                return code, lang
            else:
                error_message = f"Could not extract code from text"
                raise TaskFailed(
                    message=error_message,
                    error_type=ERROR_TYPES['problem_format']
                )
    
    def create_instruction_request(self, code: str, programming_lang: str):
        prompt_template = open("model_prompts/generate_instruction_from_code_prompt.txt").read().strip()
        prompt_text = prompt_template.format(
            language=LANGUAGE_NAMES.get(LANGUAGE, ["English", "eng"])[0],
            programming_lang=programming_lang,
            code=code
        )
        input_messages = [
            {
                "role": "user",
                "content": prompt_text
            },
        ]
        return input_messages
    
    def create_answer_request(self, problem: str, programming_lang: str):
        prompt_template = open("model_prompts/generate_code_answer.txt").read().strip()
        prompt_text = prompt_template.format(
                            code_problem=problem,
                            language=LANGUAGE_NAMES.get(LANGUAGE, ["English", "eng"])[0],
                            programming_lang=programming_lang
        )
        input_messages = [
            {
                "role": "user",
                "content": prompt_text,
            },
        ]
        return input_messages
    
    def create_judging_request(self, problem: str, solution: str, programming_lang: str):
        prompt_template = open("model_prompts/code_judging_prompt.txt").read().strip()
        prompt_text = prompt_template.format(
                            language=LANGUAGE_NAMES.get(LANGUAGE, ["English", "eng"])[0],
                            programming_lang=programming_lang,
                            problem=problem,
                            solution=solution
        )
        input_messages = [
            {
                "role": "user",
                "content": prompt_text,
            },
        ]
        return input_messages
    
    def extract_score_from_response(self, resp_text: str) -> int:
        score_match = re.search(r'Score:\s*(\d+)/\d+', resp_text)
        if not score_match:
            # self.logger.error(f"\nCould not find SCORE in judge response: {judge_resp_text}")
            error_message = "Could not find Score"
            raise TaskFailed(
                message=error_message,
                error_type=ERROR_TYPES['score_format']
            )
        score = int(score_match.group(1))
        return score
    
    
    # --------------- generator ---------------
    def task_generator(self) -> Generator[Union[Request, List[Request]], Any, Dict[str, Any]]:
        # self.data is prepopulated with the data from the jsonl row being processed
        original_question = self.data.get("question")
        text = self.data.get("answer")
        # sample_id = self.data.get("id")
        # metadata = self.data.get("metadata")
        return_dict = {
            # "sample_id": sample_id,
            "original_question": original_question,
            "original_answer": text,
            "messages": [],
        }
        # Step 1 - Extract code snippet from text
        if text is None or len(text)==0:
            error_message = "Empty text"
            raise TaskFailed(
                message=error_message,
                error_type=ERROR_TYPES['score_format']
            )
        code, programming_lang = self.extract_code_from_text(text)
        return_dict["original_code"] = code
        return_dict["programming_lang"] = programming_lang
        # Step 2 – Generate problem from code snippet
        input_messages = self.create_instruction_request(
                                                code=code,
                                                programming_lang=programming_lang)
        problem_resp: Response = yield Request({"messages": input_messages, **self.PROBLEM_GEN_PARAMS})
        problem_resp_text = problem_resp.get_text()
        # self.logger.info(f"\nPROBLEM:\n{problem_resp_text}\n")
        # Step 1.5 – Store the generated problem in the return_dict
        return_dict["messages"].append(
                    {
                        "role": "user",
                        "content": problem_resp_text,
                    })
        # Step 3 – Generate solution to the problem
        input_messages = self.create_answer_request(problem=problem_resp_text,
                                                    programming_lang=programming_lang)
        answer_resp: Response = yield Request({"messages": input_messages, **self.ANSWER_GEN_PARAMS})
        answer_resp_text = answer_resp.get_text()
        # self.logger.info(f"\nSOLUTION:\n{answer_resp_text}\n")
        # Step 3.5 - Store the proposed solution
        return_dict["messages"].append(
                {
                    "role": "assistant",
                    "content": answer_resp_text

                })
        # Step 4 - Judge quality of proposed solution
        input_messages = self.create_judging_request(
                                                    problem=problem_resp_text,
                                                    solution=answer_resp_text,
                                                    programming_lang=programming_lang)
        judge_resp: Response = yield Request({"messages": input_messages, **self.ANSWER_GEN_PARAMS})
        judge_resp_text = judge_resp.get_text()
        # Step 5 - Extract score from judging
        self.logger.info(f"\nJUDGE RESPONSE:\n{judge_resp_text}\n")
        answer_score = self.extract_score_from_response(judge_resp_text)
        self.logger.info(f"\nANSWER SCORE: {answer_score}\n")
        return_dict["answer_score"] = answer_score
        return return_dict