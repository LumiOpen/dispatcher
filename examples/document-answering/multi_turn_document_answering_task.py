"""Task description: question + answer generation from documents. Prompts are taken from the EuroLLM technical report https://arxiv.org/abs/2506.04079"""
from typing import Any, Dict, Generator, List, Union

from dispatcher.taskmanager.backend.request import Request, Response
from dispatcher.taskmanager.task.base import GeneratorTask
from utils.lang_id import detect_language

import random
import os
import re
import logging

__all__ = ["GenerateConversationFromDocumentsTask"]

SCORE_THRESH = 4  # Quality score threshold for the answer
LANGUAGE=os.environ.get("LANGUAGE")

INSTRUCTION_CATEGORIES = {
    "Problem Solving" : "Coding, Mathematical reasoning, Knowledge and reasoning", 
    "Creative Tasks" : "Creative writing, Brainstorming",
    "Information Processing" : "Summarization, Extraction, Classification, Translation", 
    "Question Answering": "Open-ended, Closed-ended, Multiple Choice",
    "Text Transformation" : "Rewriting",
    "Roleplay and Simulation": "Inhabiting a character/persona",
    "Advisory": "Asking for advice",
    "Domain-Specific Knowledge": "Humanity, history, and social studies, Other (specific domains could be added here as needed)",
    "General": ""
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

class GenerateConversationFromDocumentsTask(GeneratorTask):
    """Generate a question from a document in some language, the have the model generate an answer to the question."""

    # Fixed generation hyper‑parameters for candidate answers
    GEN_PARAMS: Dict[str, Any] = {
        "temperature": 0.6,
        "top_p": 0.9,
        "max_tokens": 4096,
    }
    
    # Deterministic parameters for the judge
    JUDGE_PARAMS: Dict[str, Any] = {
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 256,
    }
    
    logger = logging.getLogger(__name__)

    # --------------- generator ---------------
    def task_generator(self) -> Generator[Union[Request, List[Request]], Any, Dict[str, Any]]:
        # self.data is prepopulated with the data from the jsonl row being
        # processed
        return_message = {
                        "messages": []
                }
        document = self.data.get("text")
        lang_id1, lang_id2 = detect_language(document)
        # print("Checking document language")
        if lang_id1 != LANGUAGE and lang_id2 != LANGUAGE:
            error_message = f"Skipping this document. Document lang is {lang_id1.upper()} or {lang_id2.upper()}, but expected {LANGUAGE.upper()}"
            self.logger.error(error_message)
            return return_message["messages"].append({ "error": error_message })
        # draw two random categories without replacement
        categories = random.sample(list(INSTRUCTION_CATEGORIES.keys()), 2)
        gen_instruct_prompt_template = open("model_prompts/generate_instructions_prompt.txt").read().strip()
        gen_instruct_prompt_text = gen_instruct_prompt_template.format(
                        language=LANGUAGE_NAMES.get(LANGUAGE, ["English", "eng"])[0],
                        document=document,
                        category=categories[0]
                        )
        messages = [
            {
                "role": "user",
                "content": gen_instruct_prompt_text
            },
        ]

        # Step 1 – Generate instruction from a document
        instruct_resp: Response = yield Request({"messages": messages, **self.GEN_PARAMS})
        instruct_resp_text = instruct_resp.get_text()
        match = re.search(r'INSTRUCTION\s*:?([\s\S]*?)CATEGORY', instruct_resp_text)
        if match is None:
            error_message = f"Could not find keyword INSTRUCTION for the first-turn"
            self.logger.error(error_message)
            return return_message["messages"].append({ "error": error_message })
        instruct_text = match.group(1).strip()

        lang_id1, lang_id2 = detect_language(instruct_text)
        if lang_id1 != LANGUAGE and lang_id2 != LANGUAGE:
            error_message = f"Skipping this Instruction. Instruction lang is {lang_id1.upper()} or {lang_id2.upper()}, but expected {LANGUAGE.upper()}"
            self.logger.error(error_message)
            return return_message["messages"].append({ "error": error_message })
        gen_answer_prompt_template = open("model_prompts/generate_answers_prompt.txt").read().strip()
        gen_answer_prompt_text = gen_answer_prompt_template.format(
                        language=LANGUAGE_NAMES.get(LANGUAGE, ["English", "eng"])[0],
                        document=document,
                        instruction=instruct_text,
                )
        messages = [
            {
                "role": "user",
                "content": gen_answer_prompt_text
            },
        ]

        # Step 2 – Generate the answer to the instruction
        answer_resp: Response = yield Request({"messages": messages, **self.GEN_PARAMS})
        answer_text = answer_resp.get_text().strip()
        # print("Checking answer language")
        lang_id1, lang_id2 = detect_language(answer_text)
        if lang_id1 != LANGUAGE and lang_id2 != LANGUAGE:
            error_message = f"Skipping this answer. Answer lang is {lang_id1.upper()} or {lang_id2.upper()}, but expected {LANGUAGE.upper()}"
            self.logger.error(error_message)
            return return_message["messages"].append({ "error": error_message })
    
        # Step 3 - Generate second-turn instruction
        gen_next_turn_prompt_template = open("model_prompts/generate_next_turn_instruct_prompt.txt").read()
        gen_next_turn_text = gen_next_turn_prompt_template.format(
            language=LANGUAGE_NAMES.get(LANGUAGE, ["English", "eng"])[0],
            document=document,
            instruction=instruct_text,
            answer=answer_text,
            category=categories[1]
        )
        messages = [
            {
                "role": "user",
                "content": gen_next_turn_text
            }, 
        ]
        next_turn_instruct_resp: Response = yield Request({"messages": messages, **self.GEN_PARAMS})
        next_turn_instruct_resp_text = next_turn_instruct_resp.get_text()
        match = re.search(r'INSTRUCTION\s*:?([\s\S]*?)CATEGORY', next_turn_instruct_resp_text)
        if match is None:
            error_message = "Could not find keyword INSTRUCTION for the second turn"
            self.logger.error(error_message)
            return return_message["messages"].append({ "error": error_message })
        
        next_turn_instruct_text = match.group(1).strip()
        lang_id1, lang_id2 = detect_language(next_turn_instruct_text)
        if lang_id1 != LANGUAGE and lang_id2 != LANGUAGE:
            error_message = f"Skipping this instruction. Instruction lang is {lang_id1.upper()} or {lang_id2.upper()}, but expected {LANGUAGE.upper()}"
            self.logger.error(error_message)
            return return_message["messages"].append({ "error": error_message })

        # Step 4 - Generate answer to second-turn instruction
        gen_answer_prompt_template = open("model_prompts/generate_next_turn_answers_prompt.txt").read().strip()
        gen_answer_prompt_text = gen_answer_prompt_template.format(
                        language=LANGUAGE_NAMES.get(LANGUAGE, ["English", "eng"])[0],
                        document=document,
                        first_turn_instruction=instruct_text,
                        first_turn_answer=answer_text,
                        second_turn_instruction=next_turn_instruct_text
                )
        messages = [
            {
                "role": "user",
                "content": gen_answer_prompt_text
            },
        ]
        answer_resp: Response = yield Request({"messages": messages, **self.GEN_PARAMS})
        next_turn_answer_text = answer_resp.get_text().strip()
        lang_id1, lang_id2 = detect_language(next_turn_answer_text)
        if lang_id1 != LANGUAGE and lang_id2 != LANGUAGE:
            error_message = f"Skipping this answer. Answer lang is {lang_id1.upper()} or {lang_id2.upper()}, but expected {LANGUAGE.upper()}"
            self.logger.error(error_message)
            return return_message["messages"].append({ "error": error_message })
        else:
            return {
                "messages": [
                    {
                        "role": "user",
                        "content": instruct_text
                    },
                    {
                        "role": "assistant",
                        "content": answer_text
                    },
                    {
                        "role": "user",
                        "content": next_turn_instruct_text
                    },
                    {
                        "role": "assistant",
                        "content": next_turn_answer_text
                    }
                ],
            }