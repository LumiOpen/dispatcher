"""
Task description: Translates prompt and ground_truth fields from JSONL records.
"""
from typing import Any, Dict, Generator, List, Union

from dispatcher.taskmanager.backend.request import Request, Response
from dispatcher.taskmanager.task.base import GeneratorTask
from dispatcher.taskmanager.task import TaskRetry

import logging
import os
import uuid

__all__ = ["PromptTranslationTask"]

LANGUAGE = os.environ.get("LANGUAGE")


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value else default


PROMPT_TRANSLATION_MAX_TOKENS = _env_int("PROMPT_TRANSLATION_MAX_TOKENS", 4096)
GROUND_TRUTH_TRANSLATION_MAX_TOKENS = _env_int(
    "GROUND_TRUTH_TRANSLATION_MAX_TOKENS",
    _env_int("ANSWER_TRANSLATION_MAX_TOKENS", 1024),
)

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
5. Do not simplify, interpret, solve, or expand the math; your goal is linguistic translation only.
6. Keep variable names, constants, notation, numeric answers, and short answer labels unchanged when translation is not needed.
7. Do not explain your translation; output only the translated text.

Text to translate:
{text}
"""


class PromptTranslationTask(GeneratorTask):
    """Translation of Dolci prompt and ground_truth fields."""

    PROMPT_TRANSLATION_GEN_PARAMS: Dict[str, Any] = {
        "temperature": 0.7,
        "top_p": 0.95,
        "max_tokens": PROMPT_TRANSLATION_MAX_TOKENS,
    }

    GROUND_TRUTH_TRANSLATION_GEN_PARAMS: Dict[str, Any] = {
        "temperature": 0.7,
        "top_p": 0.95,
        "max_tokens": GROUND_TRUTH_TRANSLATION_MAX_TOKENS,
    }

    logger = logging.getLogger(__name__)

    def _failed_result(self, *, error_type: str, message: str, **payload: Any) -> Dict[str, Any]:
        """Wrap Task.build_result so every failure call site logs uniformly."""
        self.logger.warning(
            "[PromptTranslationTask] ID:%s Dumping unsuccessful record after final retry: %s",
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
    def _target_language() -> str:
        return LANGUAGE_NAMES.get(LANGUAGE, ["Finnish"])[0]

    @classmethod
    def _messages_for_text(cls, text: str) -> List[Dict[str, str]]:
        return [
            {
                "role": "user",
                "content": TRANSLATION_PROMPT.format(
                    language=cls._target_language(),
                    text=text,
                ),
            }
        ]

    @staticmethod
    def _normalize_ground_truth(ground_truth: Any) -> tuple[List[str], bool]:
        if ground_truth is None:
            return [], True
        if isinstance(ground_truth, list):
            return ["" if item is None else str(item) for item in ground_truth], True
        return [str(ground_truth)], False

    @staticmethod
    def _restore_ground_truth_shape(translated_items: List[str], was_list: bool) -> Any:
        if was_list:
            return translated_items
        return translated_items[0] if translated_items else ""

    def _response_text_or_retry(
        self,
        response: Response,
        sample_id: str,
        part: str,
    ) -> str:
        if not response.is_success:
            self.logger.error(
                "[PromptTranslationTask] ID:%s %s translation request failed: %s",
                sample_id,
                part,
                response.error,
            )
            raise TaskRetry(message=f"{part} translation request failed: {response.error}")

        text = response.get_text()
        if text is None:
            self.logger.error(
                "[PromptTranslationTask] ID:%s %s translation response had no extractable text payload",
                sample_id,
                part,
            )
            raise TaskRetry(
                message=f"{part} translation response had no extractable text payload"
            )

        return text.strip()

    # --------------- generator ---------------
    def task_generator(self) -> Generator[Union[Request, List[Request]], Any, Dict[str, Any]]:
        sample_id = (
            self.data.get("id")
            or self.data.get("custom_id")
            or self.data.get("prompt_id")
            or str(uuid.uuid4())
        )
        task_uuid = str(uuid.uuid4())
        # Mutate self.data so build_result spreads the resolved id/task_uuid.
        self.data["id"] = sample_id
        self.data["task_uuid"] = task_uuid
        self.logger.info(
            "[PromptTranslationTask] ID:%s UUID:%s Processing sample",
            sample_id,
            task_uuid,
        )

        prompt = "" if self.data.get("prompt") is None else str(self.data.get("prompt"))
        ground_truth_items, ground_truth_was_list = self._normalize_ground_truth(
            self.data.get("ground_truth")
        )

        requests = [
            Request(
                {
                    "messages": self._messages_for_text(prompt),
                    **self.PROMPT_TRANSLATION_GEN_PARAMS,
                },
                context={
                    "task": "prompt_translation",
                    "sample_id": sample_id,
                    "task_uuid": task_uuid,
                    "part": "prompt",
                },
            )
        ]
        requests.extend(
            Request(
                {
                    "messages": self._messages_for_text(item),
                    **self.GROUND_TRUTH_TRANSLATION_GEN_PARAMS,
                },
                context={
                    "task": "prompt_translation",
                    "sample_id": sample_id,
                    "task_uuid": task_uuid,
                    "part": "ground_truth",
                    "index": index,
                },
            )
            for index, item in enumerate(ground_truth_items)
        )

        responses = yield requests
        if isinstance(responses, Response):
            responses = [responses]
        prompt_response = next(
            response for response in responses if response.request.context["part"] == "prompt"
        )

        try:
            translated_prompt = self._response_text_or_retry(
                prompt_response,
                sample_id,
                "Prompt",
            )
        except TaskRetry as exc:
            if self.is_last_retry_attempt():
                return self._failed_result(
                    error_type="prompt_translation_error",
                    message=str(exc),
                )
            raise

        translated_ground_truth_items: List[str] = []
        ground_truth_responses = sorted(
            (
                response
                for response in responses
                if response.request.context["part"] == "ground_truth"
            ),
            key=lambda response: response.request.context["index"],
        )
        for response in ground_truth_responses:
            try:
                translated_ground_truth_items.append(
                    self._response_text_or_retry(
                        response,
                        sample_id,
                        f"Ground truth {response.request.context['index']}",
                    )
                )
            except TaskRetry as exc:
                if self.is_last_retry_attempt():
                    return self._failed_result(
                        error_type="ground_truth_translation_error",
                        message=str(exc),
                        translated_prompt=translated_prompt,
                        translated_ground_truth=translated_ground_truth_items,
                    )
                raise

        self.logger.info(
            "[PromptTranslationTask] ID:%s UUID:%s Finished processing sample",
            sample_id,
            task_uuid,
        )
        return self.build_result(
            translated_prompt=translated_prompt,
            translated_ground_truth=self._restore_ground_truth_shape(
                translated_ground_truth_items,
                ground_truth_was_list,
            ),
        )
