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
import numpy as np

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

INSTRUCTION_CATEGORIES = {
    "Problem Solving" : 0.1,
    "Creative Tasks": 0.1,
    "Information Processing": 0.1,
    "Text Transformation": 0.1,
    "Roleplay and Simulation": 0.1,
    "Domain-Specific Knowledge": 0.1,
    "Adversarial": 0.6,
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

class GenerateConversationJudgingFromDocumentsTask(GeneratorTask):
    """Generate a question from a document in some language, the have the model generate an answer to the question."""

    # Fixed generation hyper‑parameters for candidate answers
    ANSWER_GEN_PARAMS: Dict[str, Any] = {
        "temperature": 0.6,
        "top_p": 0.9,
        "max_tokens": 4096,
    }
    
    INSTRUCTION_GEN_PARAMS: Dict[str, Any] = {
        "temperature": 0.6,
        "top_p": 0.9,
        "max_tokens": 4096,
    }

    ADVERSARIAL_GEN_PARAMS: Dict[str, Any] = {
        "temperature": 0.9,
        "max_tokens": 4096,
    }

    # Deterministic parameters for the judge
    JUDGE_PARAMS: Dict[str, Any] = {
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 256,
    }
    
    logger = logging.getLogger(__name__)

    def convert_conversation_to_string(self, conversation):
        messages = conversation['messages']
        string_conv = ""
        for msg in messages:
            if msg['role'] == 'assistant':
                string_conv += f"Assistant: {msg['content']}\n\n"
            else:
                string_conv += f"User: {msg['content']}\n"
        return string_conv.strip()

    # --------------- generator ---------------
    def task_generator(self) -> Generator[Union[Request, List[Request]], Any, Dict[str, Any]]:
        # self.data is prepopulated with the data from the jsonl row being processed
        document = self.data.get("text")
        doc_id = self.data.get("id")
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
        
        # draw a random category for each turn without replacement
        prob = np.array(list(INSTRUCTION_CATEGORIES.values()))
        prob /= prob.sum()  # normalize to sum to 1
        categories = np.random.choice(list(INSTRUCTION_CATEGORIES.keys()), MAX_TURNS, p=prob)
        #self.logger.info(f"CATEGORIES: {categories}")
        return_dict = {
                        "id": doc_id,
                    }
        for turn in range(MAX_TURNS):
            if turn == 0:
                # Step 1 – Generate instruction from a document
                gen_instruct_prompt_template = open("model_prompts/generate_complex_instruction_prompt.txt").read().strip()
                gen_instruct_prompt_text = gen_instruct_prompt_template.format(
                            language=LANGUAGE_NAMES.get(LANGUAGE, ["English", "eng"])[0],
                            document=document,
                            category=categories[turn]
                            )
                messages = [
                    {
                        "role": "user",
                        "content": gen_instruct_prompt_text
                    },
                ]

                if categories[turn] != 'Adversarial':
                    instruct_resp: Response = yield Request({"messages": messages, **self.INSTRUCTION_GEN_PARAMS})
                else:
                    instruct_resp: Response = yield Request({"messages": messages, **self.ADVERSARIAL_GEN_PARAMS})
                instruct_resp_text = instruct_resp.get_text()
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
                
                # Step 1.5 - Judge instruction complexity
                judge_instruction_prompt_template = open("model_prompts/instruction_judging_prompt.txt").read()
                prompt_text = judge_instruction_prompt_template.format(
                            instruction=instruct_text, 
                        )
                messages = [
                    {
                        "role": "user",
                        "content": prompt_text
                    },
                ]
                judge_resp = yield Request({"messages": messages, **self.JUDGE_PARAMS})
                judge_resp_text = judge_resp.get_text()
                score_match = re.search(r'Score:\s*(\d+)/\d+', judge_resp_text)
                if not score_match:
                    # self.logger.error(f"\nCould not find SCORE in judge response: {judge_resp_text}")
                    error_message = "Could not find Score"
                    raise TaskFailed(
                        message=error_message,
                        error_type=ERROR_TYPES['score_format']
                    )
                first_turn_instruct_score = int(score_match.group(1))

                # Step 2 – Generate the answer to the instruction
                gen_answer_prompt_template = open("model_prompts/generate_answers_prompt.txt").read().strip()
                gen_answer_prompt_text = gen_answer_prompt_template.format(
                                language=LANGUAGE_NAMES.get(LANGUAGE, ["English", "eng"])[0],
                                document=document,
                                instruction=instruct_text,
                        )
                messages = [
                    {
                        "role": "user",
                        "content": gen_answer_prompt_text,
                    },
                ]

                answer_resp: Response = yield Request({"messages": messages, **self.ANSWER_GEN_PARAMS})
                answer_text = answer_resp.get_text().strip()
                # print("Checking answer language")
                lang_id1, lang_id2 = detect_language(answer_text)
                if lang_id1 != LANGUAGE and lang_id2 != LANGUAGE:
                    raise TaskFailed(
                        message=ERROR_TYPES['language'],
                        error_type=ERROR_TYPES['language']
                    )

                # Step 3 - Judge first-turn answer 
                judge_answer_prompt_template = open("model_prompts/answer_judging_prompt.txt").read()
                judge_answer_prompt_text = judge_answer_prompt_template.format(
                            instruction=instruct_text, 
                            answer=answer_text
                        )
                messages = [
                    {
                        "role": "user",
                        "content": judge_answer_prompt_text
                    },
                ]
                judge_resp = yield Request({"messages": messages, **self.JUDGE_PARAMS})
                judge_resp_text = judge_resp.get_text()
                score_match = re.search(r'Score:\s*(\d+)/\d+', judge_resp_text)
                if not score_match:
                    error_message="Could not find Score",
                    raise TaskFailed(
                        message=error_message,
                        error_type=ERROR_TYPES['score_format']
                    )
                first_turn_score = int(score_match.group(1))
                return_dict["messages"] = [
                            {
                                "role": "user",
                                "content": instruct_text,
                            },
                            {
                                "role": "assistant",
                                "content": answer_text,
                            },
                        ]
                return_dict[f"turn_{turn+1}_instruction_score"] = first_turn_instruct_score
                return_dict[f"turn_{turn+1}_answer_score"] = first_turn_score
                return_dict[f"turn_{turn+1}_category"] = category_text
            else: 
                # Step 4 - Generate instruction for the next turn
                existing_conv = self.convert_conversation_to_string(return_dict)
                gen_next_turn_prompt_template = open("model_prompts/generate_next_turn_complex_instruction_prompt.txt").read()
                gen_next_turn_text = gen_next_turn_prompt_template.format(
                    language=LANGUAGE_NAMES.get(LANGUAGE, ["English", "eng"])[0],
                    document=document,
                    conversation=existing_conv,
                    category=categories[turn]
                )
                # self.logger.info(f"gen_next_turn_text: {gen_next_turn_text}")
                messages = [
                    {
                        "role": "user",
                        "content": gen_next_turn_text
                    }, 
                ]
                if categories[turn] != 'Adversarial':
                    next_turn_instruct_resp: Response = yield Request({"messages": messages, **self.INSTRUCTION_GEN_PARAMS})
                else:
                    next_turn_instruct_resp: Response = yield Request({"messages": messages, **self.ADVERSARIAL_GEN_PARAMS})
                next_turn_instruct_resp_text = next_turn_instruct_resp.get_text()
                match = re.search(r'INSTRUCTION\s*:?([\s\S]*?)CATEGORY', next_turn_instruct_resp_text)
                if match is None:
                    error_message = f"Could not find keyword INSTRUCTION for the first-turn"
                    raise TaskFailed(
                        message=error_message,
                        error_type=ERROR_TYPES['instruction_format']
                    )
                next_turn_instruct_text = match.group(1).strip()

                match = re.search(r"CATEGORY:\s*(.*)", next_turn_instruct_resp_text)
                if match is None:
                    error_message = f"Could not find keyword CATEGORY for the first-turn"
                    raise TaskFailed(
                        message=error_message,
                        error_type=ERROR_TYPES['instruction_format']
                    )
                next_turn_category_text = match.group(1).strip()

                lang_id1, lang_id2 = detect_language(next_turn_instruct_text)
                if lang_id1 != LANGUAGE and lang_id2 != LANGUAGE:
                    error_message = f"Skipping this instruction. Instruction lang is {lang_id1.upper()} or {lang_id2.upper()}, but expected {LANGUAGE.upper()}"
                    raise TaskFailed(
                        message=error_message,
                        error_type=ERROR_TYPES['language']
                    )
                # Step 4.5 - Judge next-turn instruction complexity
                judge_instruction_prompt_template = open("model_prompts/instruction_judging_prompt.txt").read()
                prompt_text = judge_instruction_prompt_template.format(
                            instruction=next_turn_instruct_text, 
                        )
                messages = [
                    {
                        "role": "user",
                        "content": prompt_text
                    },
                ]
                judge_resp = yield Request({"messages": messages, **self.JUDGE_PARAMS})
                judge_resp_text = judge_resp.get_text()
                score_match = re.search(r'Score:\s*(\d+)/\d+', judge_resp_text)
                if not score_match:
                    raise TaskFailed(
                        message=ERROR_TYPES['score_format'],
                        error_type=ERROR_TYPES['score_format']
                    )
                next_turn_instruct_score = int(score_match.group(1))

                # Step 5 - Generate answer to the new turn's instruction
                return_dict["messages"].append(
                    {
                        "role": "user",
                        "content": next_turn_instruct_text,
                    }
                )
                existing_conv = self.convert_conversation_to_string(return_dict)
                gen_answer_prompt_template = open("model_prompts/generate_next_turn_answers_prompt.txt").read().strip()
                gen_answer_prompt_text = gen_answer_prompt_template.format(
                                language=LANGUAGE_NAMES.get(LANGUAGE, ["English", "eng"])[0],
                                document=document,
                                conversation=existing_conv
                        )
                messages = [
                    {
                        "role": "user",
                        "content": gen_answer_prompt_text
                    },
                ]
                answer_resp: Response = yield Request({"messages": messages, **self.ANSWER_GEN_PARAMS})
                next_turn_answer_text = answer_resp.get_text().strip()
                lang_id1, lang_id2 = detect_language(next_turn_answer_text)
                if lang_id1 != LANGUAGE and lang_id2 != LANGUAGE:
                    error_message = f"Skipping this answer. Answer lang is {lang_id1.upper()} or {lang_id2.upper()}, but expected {LANGUAGE.upper()}"
                    raise TaskFailed(
                        message=error_message,
                        error_type=ERROR_TYPES['language']
                    )
                
                # Step 6 - Judge second-turn answer
                judge_answer_prompt_text = judge_answer_prompt_template.format(
                            instruction=next_turn_instruct_text, 
                            answer=next_turn_answer_text
                        )
                messages = [
                    {
                        "role": "user",
                        "content": judge_answer_prompt_text
                    },
                ]

                judge_resp = yield Request({"messages": messages, **self.JUDGE_PARAMS})
                judge_resp_text = judge_resp.get_text()
                score_match = re.search(r'Score:\s*(\d+)/\d+', judge_resp_text)
                if not score_match:
                    error_message = "Could not find Score"
                    raise TaskFailed(
                        message=error_message,
                        error_type=ERROR_TYPES['score_format']
                    )
                next_turn_score = int(score_match.group(1))
                return_dict["messages"].append({
                    "role": "assistant",
                    "content": next_turn_answer_text,
                })
                return_dict[f"turn_{turn+1}_instruction_score"] = next_turn_instruct_score
                return_dict[f"turn_{turn+1}_answer_score"] = next_turn_score
                return_dict[f"turn_{turn+1}_category"] = next_turn_category_text
            
        return return_dict