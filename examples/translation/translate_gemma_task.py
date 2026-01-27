from typing import Any, Dict, Generator, List, Union, Tuple

from dispatcher.taskmanager.backend.request import Request, Response
from dispatcher.taskmanager.task.base import GeneratorTask
from dispatcher.taskmanager.task import TaskFailed

import random
import os
import re
import logging
from collections import Counter
from langdetect import detect



__all__ = ["TranslationTask"]

LANGUAGE=os.environ.get("LANGUAGE")

ERROR_TYPES = {
    "invalid_query": "model did not produce a valid query",
}

LANGUAGE_NAMES = {
    "bg": ["Bulgarian", "bg-BG"],
    "cs":  ["Czech", "cs-CZ"],
    "da": ["Danish", "da-DK"],
    "de": ["German", "de-DE"],
    "el": ["Greek", "el-GR"],
    "en": ["English", "en-US"],
    "es": ["Spanish", "es-ES"],
    "et": ["Estonian", "et-EE"],
    "fi": ["Finnish", "fi-FI"],
    "fr":  ["French", "fr-FR"],
    "ga": ["Irish", "ga-IE"],
    "hr": ["Croatian", "hr-HR"],
    "hu": ["Hungarian", "hu-HU"],
    "it": ["Italian", "it-IT"],
    "lt": ["Lithuanian", "lt-LT"],
    "lv": ["Latvian", "lv-LV"],
    "mt": ["Maltese", "mt-MT"],
    "nl": ["Dutch", "nl-NL"],
    "pl": ["Polish", "pl-PL"],
    "pt": ["Portuguese", "pt-PT"],
    "ro": ["Romanian", "ro-RO"],
    "sk": ["Slovak", "sk-SK"],
    "sl": ["Slovenian", "sl-SI"],
    "sv":  ["Swedish", "sv-SE"],
    "is": ["Icelandic", "is-IS"],
    "no": ["Norwegian", "nb-NO"],
}

GEMMA_TRANSLATION_PROMPT = """
You are a professional {source_lang} ({src_lang_code}) to {target_lang} ({tgt_lang_code}) translator. Your goal is to accurately convey the meaning and \
nuances of the original {source_lang} text while adhering to {target_lang} grammar, vocabulary, and cultural sensitivities. Produce only the {target_lang} \
translation, without any additional explanations or commentary. DO NOT translate XML tags such as <think></think>. Please translate the following {source_lang} text into {target_lang}:\n\n\n{text}
"""



class TranslationTask(GeneratorTask):
    """Translation implementation."""
    
    TRANSLATION_GEN_PARAMS: Dict[str, Any] = {
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 32768,
    }
    
    logger = logging.getLogger(__name__)
    max_length_ratio: float = 1.5  # maximum allowed length ratio between translation and original
    min_lenth_ratio: float = 0.5  # minimum allowed length ratio between translation and original
    min_word_count: int = 3  # minimum word count for repeating phrase detection

    def check_length(self, original: str, translation: str) -> float:
        """
        Check if translation length is reasonable compared to original.
        
        Returns:
            Tuple of (is_valid, ratio)
        """
        if not original or not translation or len(original) == 0 or len(translation) == 0:
            return 0
        
        length_ratio = len(translation) / len(original)
        # is_valid = ratio <= self.max_length_ratio
        return length_ratio

    def extract_ngrams(self, text: str, n: int) -> List[str]:
        """Extract n-grams from text (sequences of n words)."""
        words = text.lower().split()
        return [' '.join(words[i:i+n]) for i in range(len(words) - n + 1)]
    
    def check_repeating_phrases(self, text: str) -> int:
        """
        Check for repeating phrases in text.
        
        Returns:
            Tuple of (is_valid, list_of_repeated_phrases)
            where list_of_repeated_phrases contains (phrase, count) for phrases appearing 2+ times
        """
        text = text.replace("\n", " ")
        repeating_phrases = []
        
        # Check for repeating sequences of varying lengths
        for n in range(self.min_word_count, min(len(text.split()) // 2 + 1, 6)):
            ngrams = self.extract_ngrams(text, n)
            ngram_counts = Counter(ngrams)
            
            # Find ngrams that appear more than once
            for phrase, count in ngram_counts.items():
                if count > 1:
                    repeating_phrases.append((phrase, count))
        
        # Remove duplicates and keep only the longest phrases
        if repeating_phrases:
            repeating_phrases = sorted(set(repeating_phrases), key=lambda x: len(x[0]), reverse=True)
        
        # is_valid = len(repeating_phrases) < 5
        return len(repeating_phrases)
    
    # --------------- generator ---------------
    def task_generator(self) -> Generator[Union[Request, List[Request]], Any, Dict[str, Any]]:
        # self.data is prepopulated with the data from the jsonl row being processed
        uuid = self.data.get("uuid", "unknown")
        messages_to_translate = self.data.get("messages", [])
        return_dict = {
            "uuid": uuid,
            "messages": [],
        }

        for message in messages_to_translate:
            role = message.get("role", "")
            text_to_translate = message.get("content", "")
            if len(text_to_translate.strip()) == 0:
                continue
            translator_prompt = GEMMA_TRANSLATION_PROMPT.format(
                source_lang=LANGUAGE_NAMES["en"][0],
                src_lang_code=LANGUAGE_NAMES["en"][1],
                target_lang=LANGUAGE_NAMES["fi"][0],
                tgt_lang_code=LANGUAGE_NAMES["fi"][1],
                text=text_to_translate)
            # self.logger.info(f"TRANSLATOR PROMPT:\n{translator_prompt}\n")
            input_message = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            # "source_lang_code": "cs",
                            # "target_lang_code": "de-DE",
                            "text": translator_prompt,
                        }
                    ],
                }
            ]
            resp: Response = yield Request({"messages": input_message, **self.TRANSLATION_GEN_PARAMS})
            # Extract translation from response
            translated_text = resp.get_text().strip()
            self.logger.info(f"---"*50)
            # self.logger.info(f"\nTRANSLATED TEXT:\n{translated_text}\n")
            # lang_detected = detect(translated_text.replace("\n", " ")) # predict top 1 language
            # self.logger.info(f"\nDETECTED LANGUAGE: {lang_detected}\n\n")
            length_ratio = self.check_length(text_to_translate, translated_text)
            # num_repeating_phrases = self.check_repeating_phrases(translated_text)
            return_dict["messages"].append({
                "role": role,
                "content": text_to_translate,
                "translation": translated_text,
                # "translation_language": lang_detected,
                "length_ratio": length_ratio,
                # "repetitions": num_repeating_phrases,
            })
        return return_dict
