"""
Task description: Translates reasoning traces line-by-line from generated reasoning answers.
"""
from typing import Any, Dict, Generator, List, Union

from dispatcher.taskmanager.backend.request import Request, Response
from dispatcher.taskmanager.task.base import GeneratorTask
from dispatcher.taskmanager.task import TaskFailed

import os
import logging

__all__ = ["ReasoningTranslationTask"]

LANGUAGE = os.environ.get("LANGUAGE")

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

DEEPSEEK_TRANSLATION_PROMPT = """
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


class ReasoningTranslationTask(GeneratorTask):
    """Translation of prompts + reasoning traces."""


    PROMPT_TRANSLATION_GEN_PARAMS: Dict[str, Any] = {
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 8192,
    }

    TRACES_TRANSLATION_GEN_PARAMS: Dict[str, Any] = {
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 32768,
    }

    logger = logging.getLogger(__name__)

    
    # --------------- generator ---------------
    def task_generator(self) -> Generator[Union[Request, List[Request]], Any, Dict[str, Any]]:
        # self.data is prepopulated with the data from the jsonl row being processed
        # Get the generated reasoning answer
        self.logger.info(f"Processing sample id: {self.data.get('id')}")
        return_dict = self.data.copy()

        # read prompt from input.content with "role"=="user"
        prompt = next((item for item in self.data.get("input", []) if item.get("role") == "user"), {}).get("content", "")

        # create the first translation request for the prompt
        input_messages = [
            {
                "role": "user", 
                "content": DEEPSEEK_TRANSLATION_PROMPT.format(
                    language=LANGUAGE_NAMES.get(LANGUAGE, ["Finnish"])[0], 
                    text=prompt
                )
            }
        ]
        # Request translation
        try:
            resp: Response = yield Request({"messages": input_messages, **self.PROMPT_TRANSLATION_GEN_PARAMS})
        except Exception as e:
            raise TaskFailed(
                message=f"Error translating prompt: {e}", 
                error_type="prompt_translation_error"
            )
        translated_prompt = resp.get_text().strip()

        # read traces from output
        traces = self.data.get("output", {})
        # Split by </think> tag and take only the first part (reasoning traces)
        traces_part = traces.split("</think>")[0]
        # create the second translation request for the traces
        input_messages = [
            {
                "role": "user", 
                "content": DEEPSEEK_TRANSLATION_PROMPT.format(
                    language=LANGUAGE_NAMES.get(LANGUAGE, ["Finnish"])[0], 
                    text=traces_part
                )
            }
        ]
        # Request translation
        try:
            resp: Response = yield Request({"messages": input_messages, **self.TRACES_TRANSLATION_GEN_PARAMS})
        except Exception as e:
            raise TaskFailed(
                message=f"Error translating traces: {e}", 
                error_type="traces_translation_error"
            )
        translated_traces = resp.get_text().strip()
        
        # Add to return dict
        return_dict["translated_prompt"] = translated_prompt
        return_dict["translated_traces"] = translated_traces
        return return_dict

