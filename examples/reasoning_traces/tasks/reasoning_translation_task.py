"""
Task description: Translates reasoning traces line-by-line from generated reasoning answers.
"""
from typing import Any, Dict, Generator, List, Union
from functools import lru_cache

from dispatcher.taskmanager.backend.request import Request, Response
from dispatcher.taskmanager.task.base import GeneratorTask
from dispatcher.taskmanager.task import TaskFailed

import os
import logging
import re

__all__ = ["ReasoningTranslationTask"]

LANGUAGE = os.environ.get("LANGUAGE")
MODEL = os.environ.get("MODEL")

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
9. Preserve the `<think>` and `</think>` tags in the resulting text in the same positions as in the original input.

Text to translate:
{text}
"""


class ReasoningTranslationTask(GeneratorTask):
    """Translation of prompts + reasoning traces."""


    PROMPT_TRANSLATION_GEN_PARAMS: Dict[str, Any] = {
        "temperature": 0.7,
        "top_p": 0.95,
        "max_tokens": 8192,
    }

    TRACES_TRANSLATION_GEN_PARAMS: Dict[str, Any] = {
        "temperature": 0.7,
        "top_p": 0.95,
        "max_tokens": 32768,
    }

    logger = logging.getLogger(__name__)

    @staticmethod
    @lru_cache(maxsize=1)
    def get_tokenizer():
        """Load tokenizer once per worker."""
        # Import here to avoid global dependency if not used
        from transformers import AutoTokenizer
        if MODEL is None:
            raise ValueError("MODEL environment variable must be set to use tokenizer")
        return AutoTokenizer.from_pretrained(
            MODEL,
            trust_remote_code=True,
        )

    def _check_redacted_reasoning_tag(self, translation: str) -> bool:
        """Check if translation starts with <think> tag."""
        # Check if translation starts with <think> (allowing for whitespace)
        pattern = r'^\s*<think>'
        return bool(re.match(pattern, translation))

    def _check_token_count(self, original: str, translation: str) -> tuple[bool, int, int, int]:
        """Check if token count delta is acceptable (delta should not be more than 5000).
        
        Returns:
            tuple: (is_valid, original_tokens, translation_tokens, delta)
        """
        tokenizer = self.get_tokenizer()
        original_tokens = len(tokenizer.encode(original))
        translation_tokens = len(tokenizer.encode(translation))
        delta = original_tokens - translation_tokens
        # Delta should not be more than 5000 (translation should not be significantly shorter)
        is_valid = delta <= 5000
        return (is_valid, original_tokens, translation_tokens, delta)

    
    # --------------- generator ---------------
    def task_generator(self) -> Generator[Union[Request, List[Request]], Any, Dict[str, Any]]:
        # self.data is prepopulated with the data from the jsonl row being processed
        # Get the generated reasoning answer
        self.logger.info(f"[ReasoningTranslationTask] Processing sample id: {self.data.get('id')}")
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
        
        # Post-process the trace translation with retry logic
        max_retries = 5
        translated_traces = None
        resp = None
        issues = []
        
        for attempt in range(max_retries):
            # create the second translation request for the traces
            input_messages = [
                {
                    "role": "user", 
                    "content": DEEPSEEK_TRANSLATION_PROMPT.format(
                        language=LANGUAGE_NAMES.get(LANGUAGE, ["Finnish"])[0], 
                        text=traces
                    )
                }
            ]
            # Request translation
            try:
                resp: Response = yield Request({"messages": input_messages, **self.TRACES_TRANSLATION_GEN_PARAMS})
            except Exception as e:
                if attempt == max_retries - 1:
                    raise TaskFailed(
                        message=f"Error translating traces after {max_retries} attempts: {e}", 
                        error_type="traces_translation_error"
                    )
                self.logger.error(f"[ReasoningTranslationTask] Error translating traces (attempt {attempt + 1}/{max_retries}): {e}")
                continue
            
            translated_traces = resp.get_text().strip()
            finish_reason = resp.content['choices'][0]["finish_reason"]
            # Post-process checks:
            # 1. Check if translation starts with <think> tag
            # 2. Check if token count delta is acceptable (delta should not be more than 5000)
            tag_check = self._check_redacted_reasoning_tag(translated_traces)
            token_check, original_tokens, translation_tokens, delta = self._check_token_count(str(traces), translated_traces)
            
            if tag_check and token_check:
                # Both checks passed, break out of retry loop
                break
            else:
                if not tag_check:
                    issues.append(f"missing <think> tag at start (finish reason: {finish_reason})")
                if not token_check:
                    issues.append(f"token count delta too large (delta: {delta}, original tokens: {original_tokens}, translation tokens: {translation_tokens}) finish reason: {finish_reason}")
                
                if attempt < max_retries - 1:
                    self.logger.warning(
                        f"[ReasoningTranslationTask] Translation validation failed (attempt {attempt + 1}/{max_retries}): {', '.join(issues)}. Retrying..."
                    )
                else:
                    # Last attempt, log warning but proceed
                    self.logger.warning(
                        f"[ReasoningTranslationTask] Translation validation failed after {max_retries} attempts: {', '.join(issues)}. Proceeding with current translation."
                    )
        
        # Add to return dict
        return_dict["translated_prompt"] = translated_prompt
        return_dict["translated_traces"] = translated_traces
        if issues:
            return_dict["translation_issues"] = issues

        return return_dict

