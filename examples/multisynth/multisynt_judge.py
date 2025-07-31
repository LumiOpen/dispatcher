"""Example task – two responses + judge"""
import re
import threading
from typing import Any, Dict, Generator, List, Union

from dispatcher.taskmanager.backend.request import Request, Response
from dispatcher.taskmanager.task.base import GeneratorTask, TaskFailed
from transformers import AutoTokenizer, PreTrainedTokenizerBase

__all__ = ["MultiSyntJudge"]

thread_local = threading.local()

def get_tokenizer() -> PreTrainedTokenizerBase:
    """Tokenizers are not thread safe.  We will keep one copy of the tokenizer
per thread using thread local storage."""
    if not hasattr(thread_local, "tokenizer"):
        thread_local.tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-27b-it")
    return thread_local.tokenizer


def is_short_enough(text: str, tokenizer: PreTrainedTokenizerBase, max_length: int) -> bool:
    tokens = tokenizer.encode(text, add_special_tokens=True)
    return len(tokens) < max_length


class MultiSyntJudge(GeneratorTask):
    """Generate two answers, have the model judge, and return preferred vs dispreferred."""

    # Deterministic parameters for the judge
    JUDGE_PARAMS: Dict[str, Any] = {
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 256,
    }

    # --------------- generator ---------------
    def task_generator(self) -> Generator[Union[Request, List[Request]], Any, Dict[str, Any]]:
        source_text = self.data.get("source_text")
        target_text = self.data.get("target_text")
        source_language = self.data.get("source_language")
        target_language = self.data.get("target_language")

        tokenizer = get_tokenizer()
        if not is_short_enough(source_text, tokenizer, 4096) or not is_short_enough(target_text, tokenizer, 4096):
            raise TaskFailed("Dcoument(s) too long to judge")

        prompt = f"""Please judge the following translation using the provided scoring methodology.
The original text is in <original_text></original_text> tags, and the
translation is in <translation></translation> tags.

The source language is {source_language}.  The target language is
{target_language}.

Scoring methodology: You must score the text in each of the following categories using the method described.

- correct_language category: is the translated text completely in {target_language}, with no other languages mixed in? If the text is completely in the correct language assign a 1, otherwise assign a 0.
- avoid_boilerplate category: does the target text avoid any boilerplate phrases ("sure, here is your translation") or extraneous text beyond the translation itself? If the translation avoids the addition of extraneous text assign a 1, otherwise assign a 0.
- complete category: is the target text a complete translation of the original, or is it truncated? If the text is complete assign a 1, but if the document is truncated assign 0.
- well_formed category: Is the translated text in {target_language} well-formed and free of obvious grammatical errors, aside from potential truncation, which is covered by the previous category?  Additionally, is the translated text completely in the target language {target_language}, with no other languages mixed in? If so, assign a 1, otherwise assign a 0.
- accurate category: Leaving aside any issues arising from truncation, is the translated text an otherwise accurate translation of the original document?  If the translation is accurate assign a 1, otherwise assign a 0.

Here is the original text:
<original_text>{source_text}</original_text>

Here is the translation:
<translation>{target_text}</translation>

Please apply the above scoring methodology to score this translation. Your
response should be in the following format, and inside each category tag there
must only be a 1 or a 0 indicating the score for that category. Include only the
category scores in the following format, and nothing else.
```
<correct_language>[1|0]</correct_language>
<avoid_boilerplate>[1|0]</avoid_boilerplate>
<complete>[1|0]</complete>
<well_formed>[1|0]</well_formed>
<accurate>[1|0]</accurate>
``` """

        response: List[Response] = yield [Request({
            "messages": [{"role": "user", "content": prompt}],
            **self.JUDGE_PARAMS,
        })]

        judge_text = response.get_text().strip()
        def extract_score(tag: str) -> Union[int, None]:
            match = re.search(f"<{tag}>([01])</{tag}>", judge_text, re.IGNORECASE)
            return int(match.group(1)) if match else None

        scores = {
            "correct_language": extract_score("correct_language"),
            "avoid_boilerplate": extract_score("avoid_boilerplate"),
            "well_formed": extract_score("well_formed"),
            "complete": extract_score("complete"),
            "accurate": extract_score("accurate"),
        }

        if None in scores.values():
            raise TaskFailed(f"judge failure")

        result = self.data.copy()
        result.update({
            "judge_full_response": judge_text,
            "judge_model": response.model_name,
            "judge_correct_language": scores["correct_language"],
            "judge_avoid_boilerplate": scores["avoid_boilerplate"],
            "judge_well_formed": scores["well_formed"],
            "judge_complete": scores["complete"],
            "judge_accurate": scores["accurate"],
        })
        return result
