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


class TranslationIssueType:
    """Standardized issue types for translation validation."""
    MISSING_OPEN_THINK_TAG = "missing_open_think_tag"
    INVALID_OPEN_THINK_TAG_COUNT = "invalid_open_think_tag_count"
    INVALID_CLOSE_THINK_TAG_COUNT = "invalid_close_think_tag_count"
    NO_CONTENT_AFTER_CLOSE_THINK_TAG = "no_content_after_close_think_tag"
    TOKEN_COUNT_DELTA_TOO_LARGE = "token_count_delta_too_large"


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

    def _check_redacted_reasoning_tag(self, translation: str, issues: list[dict]) -> bool:
        """Check if translation starts with <think> tag. Appends issue to list if not. Returns True if pass."""
        pattern = r'^\s*<think>'
        if not re.match(pattern, translation):
            issues.append({"type": TranslationIssueType.MISSING_OPEN_THINK_TAG, "params": {}})
            return False
        return True

    def _check_think_tags_structure(self, translation: str, issues: list[dict]) -> bool:
        """Check that translation has exactly one <think> and one </think> tag,
        and that there is non-whitespace content after the closing </think> tag.
        Appends issues to list for each failed check. Returns True if all pass."""
        open_count = len(re.findall(r'<think>', translation))
        close_count = len(re.findall(r'</think>', translation))

        if open_count != 1:
            issues.append({"type": TranslationIssueType.INVALID_OPEN_THINK_TAG_COUNT, "params": {}})
            return False
        if close_count != 1:
            issues.append({"type": TranslationIssueType.INVALID_CLOSE_THINK_TAG_COUNT, "params": {}})
            return False

        # Only check content after </think> if exactly one closing tag exists
        if close_count == 1:
            after_close = translation.split('</think>', 1)[1]
            if not after_close.strip():
                issues.append({"type": TranslationIssueType.NO_CONTENT_AFTER_CLOSE_THINK_TAG, "params": {}})
                return False
        return True

    def _check_token_count(self, original: str, translation: str, issues: list[dict]) -> bool:
        """Check if token count delta is acceptable (delta should not be more than 5000).
        Appends issue with params to list if delta exceeds threshold. Returns True if pass."""
        tokenizer = self.get_tokenizer()
        original_tokens = len(tokenizer.encode(original))
        translation_tokens = len(tokenizer.encode(translation))
        delta = original_tokens - translation_tokens
        if delta > 5000:
            issues.append({"type": TranslationIssueType.TOKEN_COUNT_DELTA_TOO_LARGE, "params": {
                "delta": delta,
                "original_tokens": original_tokens,
                "translation_tokens": translation_tokens,
            }})
            return False
        return True

    
    # --------------- generator ---------------
    def task_generator(self) -> Generator[Union[Request, List[Request]], Any, Dict[str, Any]]:
        # self.data is prepopulated with the data from the jsonl row being processed
        # Get the generated reasoning answer
        self.logger.info(f"[ReasoningTranslationTask] ID:{self.data.get('id')} Processing sample")
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
        success = False
        attempts_made = 0
        
        for attempt in range(max_retries):
            attempts_made = attempt + 1
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
                continue
            
            translated_traces = resp.get_text().strip()
            # Run validation checks — short-circuit on first failure
            attempt_issues = []
            if (self._check_redacted_reasoning_tag(translated_traces, attempt_issues)
                    and self._check_think_tags_structure(translated_traces, attempt_issues)
                    and self._check_token_count(str(traces), translated_traces, attempt_issues)):
                success = True
                break
            else:
                issues.extend(attempt_issues)
        
        self.logger.info(
            f"[ReasoningTranslationTask] ID:{self.data.get('id')} Finished processing sample; "
            f"translated_traces_success={success}, attempts={attempts_made}/{max_retries}"
        )

        return_dict["translated_prompt"] = translated_prompt
        return_dict["translated_traces"] = translated_traces
        return_dict["translated_traces_success"] = success
        return_dict["translated_traces_issues"] = issues

        return return_dict
