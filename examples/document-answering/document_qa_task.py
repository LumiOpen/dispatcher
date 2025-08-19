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

__all__ = ["GenerateConversationJudgingFromDocumentsTask"]

LANGUAGE=os.environ.get("LANGUAGE")
MIN_DOC_LEN = 100
MAX_DOC_LEN = 30000
MAX_TURNS = 3

ERROR_TYPES = {
    "document_len": "document_length_error",
    "language": "language_error",
    "instruction_format": "instruction_format_error",
    "score_format": "score_format_error",
}

INSTRUCTION_CATEGORIES = [
    "Problem Solving",
    "Creative Tasks",
    "Information Processing",
    "Text Transformation",
    "Roleplay and Simulation",
    "Domain-Specific Knowledge",
    "Adversarial",
]

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

class GenerateConversationJudgingFromDocumentsTask(GeneratorTask):
    """Generate a question from a document in some language, the have the model generate an answer to the question."""
    
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

    def convert_conversation_to_string(self, messages):
        # messages = conversation['messages']
        string_conv = ""
        for msg in messages:
            if msg['role'] == 'assistant':
                string_conv += f"Assistant: {msg['content']}\n\n"
            else:
                string_conv += f"User: {msg['content']}\n"
        return string_conv.strip()
    
    def create_instruction_request(self, document: str, category: str, prior_messages: list, turn: int):
        if turn == 0:
            prompt_template = open("model_prompts/generate_complex_instruction_prompt.txt").read().strip()
            prompt_text = prompt_template.format(
                        language=LANGUAGE_NAMES.get(LANGUAGE, ["English", "eng"])[0],
                        document=document,
                        category=category
                        )
        else:
            existing_conv = self.convert_conversation_to_string(prior_messages)
            prompt_template = open("model_prompts/generate_next_turn_complex_instruction_prompt.txt").read()
            prompt_text = prompt_template.format(
                language=LANGUAGE_NAMES.get(LANGUAGE, ["English", "eng"])[0],
                document=document,
                conversation=existing_conv,
                category=category
            )
        input_messages = [
            {
                "role": "user",
                "content": prompt_text
            },
        ]
        # if turn >0:
        #     self.logger.info(f"\nINSTRUCTION PROMPT FOR TURN {turn+1}:\n{prompt_text}\n")
        return input_messages
    
    def validate_instruction(self, instruct_resp_text: str) -> tuple:
        match = re.search(r'INSTRUCTION\s*:?([\s\S]*?)CATEGORY', instruct_resp_text)
        if match is None:
            error_message = f"Could not find keyword INSTRUCTION for the first-turn"
            raise TaskFailed(
                message=error_message,
                error_type=ERROR_TYPES['instruction_format']
            )
        instruct_text = match.group(1).strip()
        match = re.search(r"CATEGORY:\s*(.*)", instruct_resp_text)
        if match is None:
            error_message = f"Could not find keyword CATEGORY for the first-turn"
            raise TaskFailed(
                message=error_message,
                error_type=ERROR_TYPES['instruction_format']
            )
        category_text = match.group(1).strip()
        lang_id1, lang_id2 = detect_language(instruct_text)
        if lang_id1 != LANGUAGE and lang_id2 != LANGUAGE:
            error_message = f"Skipping this Instruction. Instruction lang is {lang_id1.upper()} or {lang_id2.upper()}, but expected {LANGUAGE.upper()}"
            raise TaskFailed(
                message=error_message,
                error_type=ERROR_TYPES['language']
            )
        return instruct_text, category_text
    

    def create_answer_request(self, document: str, instruct_text: str, prior_messages: list, turn: int):
        if turn == 0:
            prompt_template = open("model_prompts/generate_answers_prompt_wo_documents.txt").read().strip()
            prompt_text = prompt_template.format(
                            language=LANGUAGE_NAMES.get(LANGUAGE, ["English", "eng"])[0],
                            # document=document,
                            instruction=instruct_text,
                    )
        else:
            existing_conv = self.convert_conversation_to_string(prior_messages)
            prompt_template = open("model_prompts/generate_next_turn_answers_prompt_wo_documents.txt").read().strip()
            prompt_text = prompt_template.format(
                            language=LANGUAGE_NAMES.get(LANGUAGE, ["English", "eng"])[0],
                            # document=document,
                            conversation=existing_conv
                    )
        input_messages = [
            {
                "role": "user",
                "content": prompt_text,
            },
        ]
        # if turn > 0:
        #     self.logger.info(f"\nANSWER PROMPT FOR TURN {turn+1}:\n{prompt_text}\n")
        return input_messages
    
    def validate_answer(self, answer_text: str) -> str:
        # print("Checking answer language")
        lang_id1, lang_id2 = detect_language(answer_text)
        if lang_id1 != LANGUAGE and lang_id2 != LANGUAGE:
            raise TaskFailed(
                message=ERROR_TYPES['language'],
                error_type=ERROR_TYPES['language']
            )
        return answer_text

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
        document = self.data.get("text")
        doc_id = self.data.get("id")
        # Step 0.1: Validate the document lang and length   
        self.validate_document(document)
        # Step 0.2: Draw a random category for each turn without replacement
        categories = random.sample(INSTRUCTION_CATEGORIES, MAX_TURNS)
        return_dict = {
                        "id": doc_id,
                        "messages": [],
                    }
        for turn in range(MAX_TURNS):
            # Step 1 – Generate instruction from a document
            input_messages = self.create_instruction_request(document, categories[turn], return_dict["messages"], turn)
            instruct_resp: Response = yield Request({"messages": input_messages, **self.INSTRUCTION_GEN_PARAMS})
            instruct_resp_text = instruct_resp.get_text()
            instruct_text, category_text = self.validate_instruction(instruct_resp_text)
            # Step 1.5 – Store the instruction in the return_dict
            return_dict["messages"].append(
                        {
                            "role": "user",
                            "content": instruct_text,
                        })
            # Step 2 - Judge instruction complexity
            input_messages = self.create_judge_input(instruct_text, None)
            judge_resp = yield Request({"messages": input_messages, **self.JUDGE_PARAMS})
            judge_resp_text = judge_resp.get_text()
            instruct_score = self.validate_judge_response(judge_resp_text)
            # Step 3 – Generate answer to the instruction
            input_messages = self.create_answer_request(document, instruct_text, return_dict["messages"], turn)
            answer_resp: Response = yield Request({"messages": input_messages, **self.ANSWER_GEN_PARAMS})
            answer_text = answer_resp.get_text()
            answer_score = self.validate_answer(answer_text)
            # Step 3.5 – Store the answer in the return_dict
            return_dict["messages"].append(
                        {
                            "role": "assistant",
                            "content": answer_text,
                        })
            # Step 4 - Judge answer quality
            input_messages = self.create_judge_input(instruct_text, answer_text)
            judge_resp = yield Request({"messages": input_messages, **self.JUDGE_PARAMS})
            judge_resp_text = judge_resp.get_text()
            answer_score = self.validate_judge_response(judge_resp_text)
            # Store the current turn's scores and category in the return_dict
            return_dict[f"turn_{turn+1}_instruction_score"] = instruct_score
            return_dict[f"turn_{turn+1}_answer_score"] = answer_score
            return_dict[f"turn_{turn+1}_category"] = category_text
        return return_dict