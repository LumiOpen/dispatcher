"""
Task description: Translates reasoning traces and final answers separately.
"""
from typing import Any, Dict, Generator, List, Union
from functools import lru_cache

from dispatcher.taskmanager.backend.request import Request, Response
from dispatcher.taskmanager.task.base import GeneratorTask
from dispatcher.taskmanager.task import TaskRetry

import os
import logging
import re
import uuid

__all__ = ["ReasoningTranslationSplitTracesTask"]

LANGUAGE = os.environ.get("LANGUAGE")
MODEL = os.environ.get("MODEL")


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value else default


PROMPT_TRANSLATION_MAX_TOKENS = _env_int("PROMPT_TRANSLATION_MAX_TOKENS", 8192)
TRACE_TRANSLATION_MAX_TOKENS = _env_int("TRACE_TRANSLATION_MAX_TOKENS", 32768)
ANSWER_TRANSLATION_MAX_TOKENS = _env_int("ANSWER_TRANSLATION_MAX_TOKENS", 8192)
THINK_BLOCK_PATTERN = re.compile(r"^\s*<think>(?P<traces>.*?)</think>(?P<answer>.*)\s*$", re.DOTALL)

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
9. Do not add `<think>` or `</think>` tags unless they are present in the input text.

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
    UNABLE_TO_SPLIT_THINK_BLOCK = "unable_to_split_think_block"


class ReasoningTranslationSplitTracesTask(GeneratorTask):
    """Translation of prompts plus split reasoning trace and answer bodies."""

    PROMPT_TRANSLATION_GEN_PARAMS: Dict[str, Any] = {
        "temperature": 0.7,
        "top_p": 0.95,
        "max_tokens": PROMPT_TRANSLATION_MAX_TOKENS,
    }

    TRACES_TRANSLATION_GEN_PARAMS: Dict[str, Any] = {
        "temperature": 0.7,
        "top_p": 0.95,
        "max_tokens": TRACE_TRANSLATION_MAX_TOKENS,
    }

    ANSWER_TRANSLATION_GEN_PARAMS: Dict[str, Any] = {
        "temperature": 0.7,
        "top_p": 0.95,
        "max_tokens": ANSWER_TRANSLATION_MAX_TOKENS,
    }

    logger = logging.getLogger(__name__)

    def _failed_result(self, *, error_type: str, message: str, **payload: Any) -> Dict[str, Any]:
        """Wrap Task.build_result so every failure call site logs uniformly."""
        self.logger.warning(
            "[ReasoningTranslationSplitTracesTask] ID:%s Dumping unsuccessful record after final retry: %s",
            self.data.get("id"),
            message,
        )
        return self.build_result(
            success=False,
            error=message,
            error_type=error_type,
            **payload,
        )

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

    @staticmethod
    def _split_traces_and_answer(output: str) -> tuple[str, str] | None:
        match = THINK_BLOCK_PATTERN.match(output)
        if match is None:
            return None
        return match.group("traces").strip(), match.group("answer").strip()

    @staticmethod
    def _reconstruct_traces(translated_trace_body: str, translated_answer: str) -> str:
        return f"<think>{translated_trace_body}</think>{translated_answer}"

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

    def _translation_request(self, text: str, gen_params: Dict[str, Any], *, sample_id: str, turn_index: int, part: str) -> Request:
        messages = [
            {
                "role": "user",
                "content": TRANSLATION_PROMPT.format(
                    language=LANGUAGE_NAMES.get(LANGUAGE, ["Finnish"])[0],
                    text=text,
                ),
            }
        ]
        return Request(
            {"messages": messages, **gen_params},
            context={
                "task": "reasoning_translation_split_traces",
                "sample_id": sample_id,
                "turn_index": turn_index,
                "part": part,
            },
        )

    def _extract_text(self, response: Response, label: str) -> tuple[bool, str, str]:
        """Validate one response and pull out its text. Returns (ok, text-or-error-message, error_type)."""
        if not response.is_success:
            return False, f"{label} translation request failed: {response.error}", f"{label}_translation_error"
        text = response.get_text()
        if text is None:
            return False, f"{label} translation response had no extractable text payload", f"{label}_translation_response_parsing_error"
        return True, text.strip(), ""

    # --------------- generator ---------------
    def task_generator(self) -> Generator[Union[Request, List[Request]], Any, Dict[str, Any]]:
        # self.data is prepopulated with the data from the jsonl row being processed
        sample_id = self.data.get("id") or str(uuid.uuid4())
        # Mutate self.data so build_result spreads the resolved id.
        self.data["id"] = sample_id
        self.logger.info(f"[ReasoningTranslationSplitTracesTask] ID:{sample_id} Processing sample")

        # Canonical conversation to translate: the messages list, plus (for the legacy
        # schema where the final assistant reply lives in a top-level "output" field
        # instead of being embedded in "messages") the output appended as the final
        # assistant turn.
        turns = list(self.data.get("messages", []))
        legacy_output = self.data.get("output")
        if legacy_output is not None and (not turns or turns[-1].get("role") != "assistant"):
            turns.append({"role": "assistant", "content": str(legacy_output)})

        if not turns:
            message = "No conversation turns found (empty messages and no output)"
            self.logger.error("[ReasoningTranslationSplitTracesTask] ID:%s %s", sample_id, message)
            if self.is_last_retry_attempt():
                return self._failed_result(error_type="no_conversation_turns", message=message)
            raise TaskRetry(message=message)

        last_assistant_idx = max(
            (i for i, turn in enumerate(turns) if turn.get("role") == "assistant"),
            default=None,
        )
        if last_assistant_idx is None:
            message = "No assistant turn found to translate (empty messages/output)"
            self.logger.error("[ReasoningTranslationSplitTracesTask] ID:%s %s", sample_id, message)
            if self.is_last_retry_attempt():
                return self._failed_result(error_type="no_assistant_turn", message=message)
            raise TaskRetry(message=message)

        # Build the translation plan: one entry per turn, describing how it will be
        # translated, plus the Request(s) needed to do it.
        turn_plans: List[Dict[str, Any]] = []
        requests: List[Request] = []

        for i, turn in enumerate(turns):
            role = turn.get("role")
            content = str(turn.get("content", ""))

            if role == "user":
                turn_plans.append({"index": i, "kind": "prompt", "original": content})
                requests.append(
                    self._translation_request(
                        content, self.PROMPT_TRANSLATION_GEN_PARAMS,
                        sample_id=sample_id, turn_index=i, part="prompt",
                    )
                )
                continue

            if role != "assistant":
                # No other roles observed in practice; pass through untranslated.
                turn_plans.append({"index": i, "kind": "asis", "original": content})
                continue

            split_output = self._split_traces_and_answer(content)
            if split_output is None:
                if i == last_assistant_idx:
                    # The final assistant turn is expected to always carry a reasoning
                    # trace - mirrors the original single-turn strictness check.
                    message = "Unable to split output into <think> trace body and answer"
                    self.logger.error("[ReasoningTranslationSplitTracesTask] ID:%s %s", sample_id, message)
                    if self.is_last_retry_attempt():
                        return self._failed_result(
                            error_type=TranslationIssueType.UNABLE_TO_SPLIT_THINK_BLOCK,
                            message=message,
                        )
                    raise TaskRetry(message=message)

                # Earlier assistant turns in a multi-turn conversation may legitimately
                # be plain text with no reasoning trace - translate as-is.
                turn_plans.append({"index": i, "kind": "plain", "original": content})
                requests.append(
                    self._translation_request(
                        content, self.ANSWER_TRANSLATION_GEN_PARAMS,
                        sample_id=sample_id, turn_index=i, part="full",
                    )
                )
                continue

            trace_body, answer = split_output
            turn_plans.append({
                "index": i, "kind": "split", "original": content,
                "trace_body": trace_body, "answer": answer,
            })
            requests.append(
                self._translation_request(
                    trace_body, self.TRACES_TRANSLATION_GEN_PARAMS,
                    sample_id=sample_id, turn_index=i, part="trace_body",
                )
            )
            requests.append(
                self._translation_request(
                    answer, self.ANSWER_TRANSLATION_GEN_PARAMS,
                    sample_id=sample_id, turn_index=i, part="answer",
                )
            )

        responses = yield requests
        responses_by_key = {
            (response.request.context["turn_index"], response.request.context["part"]): response
            for response in responses
        }

        # Phase 1: validate every response and extract its translated text.
        texts_by_key: Dict[tuple[int, str], str] = {}
        for plan in turn_plans:
            i = plan["index"]
            parts = {"prompt": ["prompt"], "plain": ["full"], "split": ["trace_body", "answer"]}.get(plan["kind"], [])
            for part in parts:
                response = responses_by_key[(i, part)]
                ok, text_or_message, error_type = self._extract_text(response, f"turn{i}_{part}")
                if not ok:
                    self.logger.error("[ReasoningTranslationSplitTracesTask] ID:%s %s", sample_id, text_or_message)
                    if self.is_last_retry_attempt():
                        return self._failed_result(error_type=error_type, message=text_or_message)
                    raise TaskRetry(message=text_or_message)
                texts_by_key[(i, part)] = text_or_message

        # Phase 2: reconstruct each turn's translated content and validate token counts.
        translated_by_turn: Dict[int, str] = {}
        for plan in turn_plans:
            i = plan["index"]
            kind = plan["kind"]

            if kind == "asis":
                translated_by_turn[i] = plan["original"]
                continue

            if kind == "prompt":
                translated_by_turn[i] = texts_by_key[(i, "prompt")]
                continue

            if kind == "plain":
                translated_content = texts_by_key[(i, "full")]
            else:  # split
                translated_content = self._reconstruct_traces(
                    texts_by_key[(i, "trace_body")],
                    texts_by_key[(i, "answer")],
                )

            issues: list = []
            if not self._check_token_count(plan["original"], translated_content, issues):
                issue_types = ", ".join(issue["type"] for issue in issues)
                message = f"Turn {i} translation validation failed: {issue_types}"
                if self.is_last_retry_attempt():
                    error_type = "trace_translation_validation_failed" if kind == "split" else "translation_validation_failed"
                    return self._failed_result(error_type=error_type, message=message)
                raise TaskRetry(message=message)

            translated_by_turn[i] = translated_content

        translated_messages = [
            {"role": turns[i].get("role"), "content": translated_by_turn[i]}
            for i in range(len(turns))
        ]

        self.logger.info(
            f"[ReasoningTranslationSplitTracesTask] ID:{sample_id} Finished processing sample"
        )

        return self.build_result(translated_messages=translated_messages)
