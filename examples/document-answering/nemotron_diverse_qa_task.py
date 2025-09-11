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

__all__ = ["GenerateNemotronDiverseQATask"]

LANGUAGE=os.environ.get("LANGUAGE")
MIN_DOC_LEN = 100
MAX_DOC_LEN = 30000


ERROR_TYPES = {
    "document_len": "document_length_error",
    "language": "language_error",
    "instruction_format": "instruction_format_error",
    "score_format": "score_format_error",
}




class GenerateNemotronDiverseQATask(GeneratorTask):
    """Generate a question from a document in some language, the have the model generate an answer to the question."""
    
    GEN_PARAMS: Dict[str, Any] = {
        "temperature": 0.9,
        "top_p": 0.9,
        "max_tokens": 4096,
    }

    ADVERSARIAL_GEN_PARAMS: Dict[str, Any] = {
        "temperature": 0.95,
        "max_tokens": 4096,
    }

    # Deterministic parameters for the judge
    JUDGE_PARAMS: Dict[str, Any] = {
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 256,
    }
    
    logger = logging.getLogger(__name__)

    
    def create_instruction_request(self, document: str):
        prompt_template = open("model_prompts/nemotron_diverse_qa_task.txt").read().strip()
        prompt_text = prompt_template.format(
                    document=document,
                    )
        input_messages = [
            {
                "role": "user",
                "content": prompt_text
            },
        ]
        return input_messages
    
    def extract_qa_pairs(self, instruct_resp_text: str) -> tuple:
        # self.logger.info(f"\n\nINSTRUCT_RESP_TEXT:\n{instruct_resp_text}\n")
        # Regex to capture QUESTION and ANSWER parts
        pattern = r'QUESTION:\s*(.*?)\s*ANSWER:\s*(.*)'
        pairs = re.findall(pattern, instruct_resp_text)
        questions_list = []
        answers_list = []
        for i, (question_text, answer_text) in enumerate(pairs):
            self.logger.info(f"\n\nQUESTION {i}:\n{question_text}\n")
            self.logger.info(f"\n\nANSWER {i}:\n{answer_text}\n")
            q_lang1, q_lang2 = detect_language(question_text)
            a_lang1, a_lang2 = detect_language(answer_text)
            if (q_lang1 == LANGUAGE or q_lang2 == LANGUAGE) and (a_lang1 == LANGUAGE or a_lang2 == LANGUAGE):
                # Only add the QA pair if both question and answer are in the expected language
                questions_list.append(question_text.strip())
                answers_list.append(answer_text.strip())
        return questions_list, answers_list

    def create_judge_input(self, instruct_text: str, answer_text: str):
        if answer_text is None:
            # judge the instruction: there is no answer text, so we judge the instruction only
            prompt_template = open("model_prompts/instruction_judging_prompt.txt").read()
            prompt_text = prompt_template.format(
                        instruction=instruct_text, 
                    )
        else:
            # judge the answer: answer text is present, so we judge the answer
            prompt_template = open("model_prompts/answer_judging_prompt.txt").read()
            prompt_text = prompt_template.format(
                        instruction=instruct_text, 
                        answer=answer_text
                    )
        input_messages = [
            {
                "role": "user",
                "content": prompt_text
            },
        ]
        return input_messages
    
    def validate_judge_response(self, judge_resp_text: str) -> int:
        score_match = re.search(r'Score:\s*(\d+)/\d+', judge_resp_text)
        if not score_match:
            # self.logger.error(f"\nCould not find SCORE in judge response: {judge_resp_text}")
            error_message = "Could not find Score"
            raise TaskFailed(
                message=error_message,
                error_type=ERROR_TYPES['score_format']
            )
        score = int(score_match.group(1))
        return score

    def validate_document(self, document: str) -> None:
        lang_id1, lang_id2 = detect_language(document)
        if len(document.split()) < MIN_DOC_LEN:
            error_message = f"Document too short. Document length is {len(document.split())}. Min length is {MIN_DOC_LEN}."
            raise TaskFailed(
                message=error_message,
                error_type=ERROR_TYPES["document_len"]
            )
        if len(document.split()) > MAX_DOC_LEN:
            error_message = f"Document too long. Document length is {len(document.split())}. Max length is {MAX_DOC_LEN}."
            raise TaskFailed(
                message=error_message,
                error_type=ERROR_TYPES["document_len"]
            )
        if lang_id1 != LANGUAGE and lang_id2 != LANGUAGE:
            error_message = f"Skipping this document. Document lang is {lang_id1.upper()} or {lang_id2.upper()}, but expected {LANGUAGE.upper()}"
            raise TaskFailed(
                message=error_message,
                error_type=ERROR_TYPES['language']
            )
    
    # --------------- generator ---------------
    def task_generator(self) -> Generator[Union[Request, List[Request]], Any, Dict[str, Any]]:
        # self.data is prepopulated with the data from the jsonl row being processed
        # self.logger.info(f"\nANSWER_FROM_DOCUMENT={ANSWER_FROM_DOCUMENT}\n")
        document = self.data.get("text")
        doc_id = self.data.get("id")
        # Step 0.1 - Validate the document lang and length   
        self.validate_document(document)
        # Step 0.2 - Draw a random category for each turn without replacement
        return_dict = {
                        "id": doc_id,
                        "questions": [],
                        "answers": [],
        }
        # Step 1 – Generate QA pairs from a document
        input_messages = self.create_instruction_request(document)
        instruct_resp: Response = yield Request({"messages": input_messages, **self.GEN_PARAMS})
        instruct_resp_text = instruct_resp.get_text()
        questions, answers = self.extract_qa_pairs(instruct_resp_text)
        return_dict["questions"].extend(questions)
        return_dict["answers"].extend(answers)
        return return_dict