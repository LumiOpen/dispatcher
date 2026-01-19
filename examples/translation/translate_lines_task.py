"""Task description: translation task."""
from typing import Any, Dict, Generator, List, Union

from dispatcher.taskmanager.backend.request import Request, Response
from dispatcher.taskmanager.task.base import GeneratorTask
from dispatcher.taskmanager.task import TaskFailed

import random
import os
import re
import logging

from preprocess import preprocess_for_few_shot_translation, split_text_into_lines, classify_line, ContentType
from language_names import LANGUAGE_NAMES
from postprocess import reconstruct_translated_text, extract_translation

__all__ = ["TranslateLinesTask"]

LANGUAGE=os.environ.get("LANGUAGE")
FEW_SHOT_PROMPT=os.environ.get("FEW_SHOT_PROMPT", "false").lower() == "true"
N_SHOTS=int(os.environ.get("N_SHOTS", "5"))  # Default to 5 shots

ERROR_TYPES = {
    "invalid_query": "model did not produce a valid query",
}

TRANSLATION_PROMPT = """
You are a professional translator. Your task is to translate the following math problem faithfully and accurately into {language}. Translate the problem only, DO NOT answer the problem.

Guidelines:

1. Preserve the original meaning, tone, and nuance.
2. Do not summarize, omit, or add information. 
3. Keep numbers, technical terms unchanged unless a standard translation exists.
4. Do not translate mathematical notation, LaTeX syntax, or formulas; keep them exactly as in the source text. Mathematical notations are indicated by dollar signs ($...$) for inline math and double dollar signs ($$...$$) for display math. Some notations use escape characters (e.g., \\alpha, \\sum).
5. Maintain the style and register (formal/informal, literary/technical, etc.) of the source text.
6. If a phrase could be translated multiple ways, choose the one that best matches the author’s intent and note any ambiguity.

Do not explain your translation; output only the translated text unless asked otherwise.

Problem to translate:
{text}
"""


class TranslateLinesTask(GeneratorTask):
    """Translation implementation."""

    TRANSLATION_GEN_PARAMS: Dict[str, Any] = {
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 512,
    }
    
    logger = logging.getLogger(__name__)

    # --------------- generator ---------------
    def task_generator(self) -> Generator[Union[Request, List[Request]], Any, Dict[str, Any]]:
        # self.data is prepopulated with the data from the jsonl row being processed
        # messages_to_translate = self.data.get("input")
        # text_to_translate = messages_to_translate[0]['content']
        uuid = self.data.get("uuid", "unknown")
        messages_to_translate = self.data.get("messages", [])
        return_dict = {
            "uuid": uuid,
            "messages": [],
        }

        for message in messages_to_translate:
            role = message.get("role", "user")
            text_to_translate = message.get("content", "")
            translated_text = []
            lines, separators = split_text_into_lines(text_to_translate)
            for i, line in enumerate(lines):
                translated_line = []
                #self.logger.info(f"LINE TO TRANSLATE: {line}")
                content_type = classify_line(line)
                if content_type == ContentType.TRANSLATABLE:
                    # construct prompt
                    line_prompt = TRANSLATION_PROMPT.format(text=line, language=LANGUAGE_NAMES.get(LANGUAGE, LANGUAGE))
                    input_message = [
                        {"role": "user", "content": line_prompt}
                    ]
                    resp: Response = yield Request({"messages": input_message, **self.TRANSLATION_GEN_PARAMS})
                    # Extract translation from response
                    resp_text = resp.get_text()
                    #self.logger.info(f"Raw response for line {i+1}: {resp_text}")
                    translated_line = resp_text.strip()
                else:
                    translated_line = line  # Non-translatable lines are kept as is
                    self.logger.info(f"NON-TRANSLATABLE LINE, keeping as is.")
                translated_text.append(translated_line)
                translated_text.append(separators[i])
            full_translated_text = "".join(translated_text).strip()
            self.logger.info(f"\n\n------------ ROLE: {role} ------------\n\n")
            self.logger.info(f"ORIGINAL TEXT: {text_to_translate[:100]}")
            self.logger.info(f"FULL TRANSLATED TEXT: {full_translated_text[:100]}")
            return_dict["messages"].append({
                "role": role,
                "content": text_to_translate,
                "translation": full_translated_text,
            })
        return return_dict
    