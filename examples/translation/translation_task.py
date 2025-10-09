"""Task description: question + answer generation from documents. Prompts are taken from the EuroLLM technical report https://arxiv.org/abs/2506.04079"""
from typing import Any, Dict, Generator, List, Union

from dispatcher.taskmanager.backend.request import Request, Response
from dispatcher.taskmanager.task.base import GeneratorTask
from dispatcher.taskmanager.task import TaskFailed

import random
import os
import re
import logging

__all__ = ["TranslationTask"]

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

TRANSLATION_PROMPT = """
You are a professional translator. Your task is to translate the following math problem faithfully and accurately into {language}. Translate the problem only, DO NOT answer the problem.

Guidelines:

1. Preserve the original meaning, tone, and nuance.
2. Do not summarize, omit, or add information. 
3. Keep mathematical notations, numbers, technical terms unchanged unless a standard translation exists.
4. Maintain the style and register (formal/informal, literary/technical, etc.) of the source text.
5. If a phrase could be translated multiple ways, choose the one that best matches the author’s intent and note any ambiguity.

Do not explain your translation; output only the translated text unless asked otherwise.

Problem to translate:
{text}
"""


DEEPSEEK_R1_TRANSLATION_PROMPT = """
You are a professional translator specializing in mathematics and scientific texts. Your task is to translate the following content faithfully and accurately into {language}.

Guidelines:

1. Preserve all LaTeX code, equations, symbols, and formatting exactly as written.
2. Translate only the surrounding natural language, not the math expressions inside \( ... \), \[ ... \], or $$ ... $$.
3. Maintain the precise meaning, tone, and logical structure of the original text.
4. Use the standard mathematical terminology of the target language.
5. Do not simplify, interpret, or solve the math; your goal is linguistic translation only.
6. Keep variable names, constants, and notation unchanged.
7. If an English math term has multiple valid equivalents in the target language, choose the most widely accepted in academic usage.
8. Do not explain your translation; output only the translated text unless asked otherwise.

Text to translate:
{text}
"""

class TranslationTask(GeneratorTask):
    """Translation implementation."""
    
    TRANSLATION_GEN_PARAMS: Dict[str, Any] = {
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 8192,
    }
    
    logger = logging.getLogger(__name__)

    
    # --------------- generator ---------------
    def task_generator(self) -> Generator[Union[Request, List[Request]], Any, Dict[str, Any]]:
        # self.data is prepopulated with the data from the jsonl row being processed
        messages_to_translate = self.data.get("input")
        text_to_translate = messages_to_translate[0]['content']
        return_dict = {
                        "text": messages_to_translate[0]['content'],
                        "translation": "",
                    }
    
        input_messages = [
            {
                "role": "user", 
                "content": DEEPSEEK_R1_TRANSLATION_PROMPT.format(language=LANGUAGE_NAMES.get(LANGUAGE, ["English"])[0], 
                                                          text=text_to_translate)
            }
        ]
        resp: Response = yield Request({"messages": input_messages, **self.TRANSLATION_GEN_PARAMS})
        resp_text = resp.get_text()
        self.logger.info(f"\n\nTRANSLATION:\n{resp_text}\n")
        return_dict["translation"] = resp_text.strip()
        return return_dict
