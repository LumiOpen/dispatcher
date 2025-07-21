"""Task descript: question + answer generation from documents. Prompts are taken from the EuroLLM technical report https://arxiv.org/abs/2506.04079"""
from typing import Any, Dict, Generator, List, Union

from dispatcher.taskmanager.backend.request import Request, Response
from dispatcher.taskmanager.task.base import GeneratorTask

__all__ = ["GenerateSamplesFromDocumentsTask"]


import random
import os

LANGUAGE=os.environ.get("LANGUAGE")

INSTRUCTION_CATEGORIES = {
    "Problem Solving" : "Coding, Mathematical reasoning, Knowledge and reasoning", 
    "Creative Tasks" : "Creative writing, Brainstorming",
    "Information Processing" : "Summarization, Extraction, Classification, Translation 4. Question Answering: Open-ended, Closed-ended, Multiple Choice",
    "Text Transformation" : "Rewriting",
    "Roleplay and Simulation": "Inhabiting a character/persona",
    "Advisory": "Asking for advice",
    "Domain-Specific Knowledge": "Humanity, history, and social studies, Other (specific domains could be added here as needed)",
    "General / Miscellaneous": ""
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

class GenerateSamplesFromDocumentsTask(GeneratorTask):
    """Generate a question from a document in some language, the have the model generate an answer to the question."""

    # Fixed generation hyper‑parameters for candidate answers
    GEN_PARAMS: Dict[str, Any] = {
        "temperature": 0.7,
        "top_p": 0.95,
        "max_tokens": 4096,
    }

    # --------------- generator ---------------
    def task_generator(self) -> Generator[Union[Request, List[Request]], Any, Dict[str, Any]]:
        # self.data is prepopulated with the data from the jsonl row being
        # processed
        document = self.data.get("document")
        prompt_template = open("model_prompts/generate_instructions_prompt.txt").read().strip()
        gen_instruction_prompt_text = prompt_template.format(
                        document=document,
                        language=LANGUAGE_NAMES.get(LANGUAGE, ["English", "eng"])[0],
                        category=random.choice(list(INSTRUCTION_CATEGORIES.keys())),
                ),
        print(f"\ngen_instruction_prompt_text: {gen_instruction_prompt_text}")
        messages = [
            {
                "role": "user",
                "content": gen_instruction_prompt_text
            },
        ]

        # Step 1 – Generate instruction from a document
        instruct_resp: Response = yield Request({"messages": messages, **self.GEN_PARAMS})
        instruction_text = instruct_resp.get_text()
        print(f"\ninstruction_text: {instruction_text}\n-------------")

        gen_answer_prompt_template = open("model_prompts/generate_answers_prompt.txt").read().strip()
        gen_answer_prompt_text = gen_answer_prompt_template.format(
                        language=LANGUAGE_NAMES.get(LANGUAGE, ["English", "eng"])[0],
                        document=document,
                        instruction=instruction_text,
                )
        print(f"\ngen_answer_prompt_text: {gen_answer_prompt_text}")
        messages = [
            {
                "role": "user",
                "content": gen_answer_prompt_text
            },
        ]

        # Step 2 – Generate the answer to the instruction
        answer_resp: Response = yield Request({"messages": messages, **self.GEN_PARAMS})
        answer_text = answer_resp.get_text().strip()
        print(f"\nanswer_text: {answer_text}\n-------------")


        # return dict can contain anything you wish to record from this task.
        return {
            "messages": [
                {
                    "role": "user",
                    "content": instruction_text
                },
                {
                    "role": "assistant",
                    "content": answer_text
                },
            ],
        }
