"""Task description: translation task."""
from typing import Any, Dict, Generator, List, Union

from dispatcher.taskmanager.backend.request import Request, Response
from dispatcher.taskmanager.task.base import GeneratorTask
from dispatcher.taskmanager.task import TaskFailed

import random
import os
import re
import logging

from preprocess import preprocess_for_few_shot_translation
from language_names import LANGUAGE_NAMES
from postprocess import reconstruct_translated_text

__all__ = ["TranslationTask"]

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
        "max_tokens": 512,
    }
    
    logger = logging.getLogger(__name__)

    # --------------- generator ---------------
    def task_generator(self) -> Generator[Union[Request, List[Request]], Any, Dict[str, Any]]:
        # self.data is prepopulated with the data from the jsonl row being processed
        messages_to_translate = self.data.get("input")
        text_to_translate = messages_to_translate[0]['content']
        
        return_dict = {
            "text": text_to_translate,
            "translation": "",
        }

        # Check if few-shot prompting is enabled
        if FEW_SHOT_PROMPT:
            self.TRANSLATION_GEN_PARAMS["stop"] = ["##", "###"]  # Used in few-shot prompting as separator

            target_lang_name = LANGUAGE_NAMES.get(LANGUAGE, ["English"])[0]
            
            # Preprocess text for line-by-line translation
            line_prompts, structure_info = preprocess_for_few_shot_translation(
                text_to_translate,
                source_lang_code="eng",  # Always use English as source
                target_lang_code=LANGUAGE,
                target_lang_name=target_lang_name,
                n_shots=N_SHOTS
            )
            
            if not line_prompts:
                self.logger.warning("No lines to translate after preprocessing")
                return return_dict
            
            self.logger.info(f"Processing {len(line_prompts)} lines with few-shot prompting")
            
            # Yield requests for each line
            line_translations = []
            for i, line_prompt in enumerate(line_prompts):
                self.logger.info(f"Processing line {i+1}/{len(line_prompts)}")
                resp: Response = yield Request({"prompt": line_prompt, **self.TRANSLATION_GEN_PARAMS})
                
                # Extract translation from response
                resp_text = resp.get_text()
                self.logger.info(f"Raw response for line {i+1}: {resp_text}")
                line_translations.append(resp_text.strip())
            
            # Reconstruct the full translated text
            full_translation = reconstruct_translated_text(line_translations, structure_info)
            self.logger.info(f"FULL TRANSLATION:\n{full_translation}\n")
            return_dict["translation"] = full_translation
            return_dict["prompt"] = line_prompts
            
        else:
            # Use original instruction-based prompting for chat models
            prompt_content = TRANSLATION_PROMPT.format(
                language=LANGUAGE_NAMES.get(LANGUAGE, ["English"])[0], 
                text=text_to_translate
            )
            # prompt_content = DEEPSEEK_R1_TRANSLATION_PROMPT.format(
            #     language=LANGUAGE_NAMES.get(LANGUAGE, ["English"])[0], 
            #     text=text_to_translate
            # )
            input_messages = [
                {
                    "role": "user", 
                    "content": prompt_content
                }
            ]
            resp: Response = yield Request({"messages": input_messages, **self.TRANSLATION_GEN_PARAMS})
            
            resp_text = resp.get_text()
            self.logger.info(f"\n\nTRANSLATION:\n{resp_text}\n")
            return_dict["translation"] = resp_text.strip()
            return_dict["prompt"] = prompt_content.strip()
        
        return return_dict
    