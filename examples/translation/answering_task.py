"""
Task description: Generates (reasoning) answers for existing prompts.
This task is the second step in our prompt translation evaluation pipeline.
We generate multiple answers for each prompt.
"""
from typing import Any, Dict, Generator, List, Union

from dispatcher.taskmanager.backend.request import Request, Response
from dispatcher.taskmanager.task.base import GeneratorTask
from dispatcher.taskmanager.task import TaskFailed

import random
import os
import re
import logging

__all__ = ["TranslationAnsweringTask"]

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

GENERAL_TRANSLATION_PROMPT = """
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


DEEPSEEK_TRANSLATION_PROMPT = """
You are a professional translator specializing in mathematics and scientific texts. Your task is to translate the following content faithfully and accurately into {language}.

Guidelines:

1. Preserve all LaTeX code, equations, symbols, and formatting exactly as written.
2. Translate only the surrounding natural language, not the math expressions inside ( ... ), [ ... ], or $$ ... $$.
3. Maintain the precise meaning, tone, and logical structure of the original text.
4. Use the standard mathematical terminology of the target language.
5. Do not simplify, interpret, or solve the math; your goal is linguistic translation only.
6. Keep variable names, constants, and notation unchanged.
7. If an English math term has multiple valid equivalents in the target language, choose the most widely accepted in academic usage.
8. Do not explain your translation; output only the translated text unless asked otherwise.

Text to translate:
{text}
"""

OPEN_R1_REASONING_PROMPT = """
You are a reasoning model to help users solve complex problems. Your role as an assistant involves thoroughly exploring questions through a systematic thinking process before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered thinking process. 
Please structure your response into two main sections: 
- Thought and Solution using the specified format: <think> [Thought section] </think> [Solution section]. 
- In the Thought section, detail your reasoning process in steps. Each step should include detailed considerations such as analysing questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps. 
- In the Solution section, based on various attempts, explorations, and reflections from the Thought section, systematically present the final solution that you deem correct. The Solution section should be logical, accurate, and concise and detail necessary steps needed to reach the conclusion. Do not use <solution></solution> tags or the word "Solution". Just append the answer after the Thought section.
- Since the question is in {language}, you MUST respond entirely in {language} for both the Thought and Solution sections.

Now, try to solve the following question through the above guidelines.
{question}
"""

class AnsweringTask(GeneratorTask):
    """Translation and reasoning trace generation."""
    
    TRANSLATION_GEN_PARAMS: Dict[str, Any] = {
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 8192,
    }
    
    ANSWER_GEN_PARAMS: Dict[str, Any] = {
        "temperature": 0.6,
        "top_p": 0.95,
        "max_tokens": 14336, # Leave 2K tokens for the prompt
    #    'n': 16, # We generate multiple answers for each prompt. # FIXME: This doesn't seem to work
    }

    logger = logging.getLogger(__name__)

    
    # --------------- generator ---------------
    def task_generator(self) -> Generator[Union[Request, List[Request]], Any, Dict[str, Any]]:
        # self.data is prepopulated with the data from the jsonl row being processed
        # We return the original data along with the generated answer
        return_dict = self.data.copy()

        input_messages = [
            {
                "role": "user", 
                "content": OPEN_R1_REASONING_PROMPT.format(language=LANGUAGE_NAMES.get(LANGUAGE)[0], 
                                                          question=self.data["generated_translation"])
            }
        ]
        # We request multiple answers per prompt
        answers = []
        for _ in range(4):
            answer_resp: Response = yield Request({"messages": input_messages, **self.ANSWER_GEN_PARAMS})
            answers.append(answer_resp.get_text())
        return_dict["generated_answers"] = answers
        return return_dict
