"""Task description: question + answer generation from documents. Prompts are taken from the EuroLLM technical report https://arxiv.org/abs/2506.04079"""
from typing import Any, Dict, Generator, List, Union

from dispatcher.taskmanager.backend.request import Request, Response
from dispatcher.taskmanager.task.base import GeneratorTask
from dispatcher.taskmanager.task import TaskFailed

import random
import os
import re
import logging
import ast
from pygments.lexers import guess_lexer
from pygments.util import ClassNotFound
from tree_sitter import Parser
from tree_sitter_languages import get_language

__all__ = ["GenerateCodeProblemsTask"]

ERROR_TYPES = {
    "problem_format": "missing_keyword",
}

LANGUAGE=os.environ.get("LANGUAGE")

PROGRAMMING_LANGUAGES = ['python', 'javascript', 'c#', 'sql', 'java', 'c++', 'go', 'c',
       'php', 'typescript', 'rust', 'r', 'bash']

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


class GenerateCodeProblemsTask(GeneratorTask):
    """Generate math problems and solutions based on a persona."""
    
    ANSWER_GEN_PARAMS: Dict[str, Any] = {
        "temperature": 0.8,
        "top_p": 0.9,
        "max_tokens": 4096,
    }
    
    PROBLEM_GEN_PARAMS: Dict[str, Any] = {
        "temperature": 0.6,
        "top_p": 0.9,
        "max_tokens": 4096,
    }
    
    logger = logging.getLogger(__name__)

    
    def create_instruction_request(self, persona: str, programming_lang: str):
        prompt_template = open("model_prompts/generate_code_problem.txt").read().strip()
        prompt_text = prompt_template.format(
            persona=persona,
            language=LANGUAGE_NAMES.get(LANGUAGE, ["English", "eng"])[0],
            programming_lang=programming_lang
        )
        input_messages = [
            {
                "role": "user",
                "content": prompt_text
            },
        ]
        return input_messages
    
    def extract_problem(self, problem_resp_text: str) -> tuple:
        # self.logger.info(f"\nPROBLEM RESPONSE:\n{problem_resp_text}\n")
        match = re.search(r'QUESTION:\s*(.*)', problem_resp_text, re.DOTALL)
        if match is None:
            error_message = f"Could not find keyword PROBLEM"
            raise TaskFailed(
                message=error_message,
                error_type=ERROR_TYPES['problem_format']
            )
        problem_text = match.group(1).strip()
        # self.logger.info(f"\nEXTRACTED PROBLEM:\n{problem_text}\n")
        return problem_text
    

    def create_answer_request(self, problem: str, programming_lang: str):
        prompt_template = open("model_prompts/generate_code_answer.txt").read().strip()
        prompt_text = prompt_template.format(
                            code_problem=problem,
                            language=LANGUAGE_NAMES.get(LANGUAGE, ["English", "eng"])[0],
                            programming_lang=programming_lang
        )
        input_messages = [
            {
                "role": "user",
                "content": prompt_text,
            },
        ]
        return input_messages
    
    def verify_python_code(self, code_snippet: str):
        valid_code = False
        pattern = re.compile(r"```(\w+)?\n(.*?)```", re.DOTALL)
        for lang, code in pattern.findall(code_snippet):
            try:
                ast.parse(code)
                valid_code = True
                self.logger.info("VALID PYTHON CODE")
            except:
                error_message = "Could not parse python code"
                raise TaskFailed(
                    message=error_message,
                    error_type=ERROR_TYPES['problem_format']
                )
        return valid_code
    
    def verify_other_code_pygments(self, code_snippet: str, target_code_lang: str):
        valid_code = False
        pattern = re.compile(r"```(\w+)?\n(.*?)```", re.DOTALL)
        # self.logger.info(f"\nCODE SNIPPET:\n{code_snippet}\n")
        for lang, code in pattern.findall(code_snippet):
            try:
                self.logger.info(f"\nEXTRACTED CODE:\n{code}\n")
                lexer = guess_lexer(code)
                # print(lexer.name)  # e.g. "Java"
                self.logger.info(f"| IDENTIFIED LANG: {lexer.name} | TARGET LANG: {target_code_lang} |\n")
                # self.logger.info(f"VALID {lang.upper()} code")
            except ClassNotFound:
                error_message = "Could not determine language"
                raise TaskFailed(
                    message=error_message,
                    error_type=ERROR_TYPES['problem_format']
                )
        return valid_code

    def count_errors_tree_sitter(self, node):
        """Recursively count ERROR nodes in a tree-sitter AST."""
        errors = 1 if node.type == "ERROR" else 0
        for child in node.children:
            errors += self.count_errors_tree_sitter(child)
        return errors

    def verify_code_tree_sitter(self, code_snippet: str, prog_lang: str) -> str:
        """
        Detects the best matching language for a code snippet using Tree-sitter.
        Returns a language key from SUPPORTED_LANGS, or 'unknown'.
        """
        best_lang = None
        min_errors = 2
        pattern = re.compile(r"```(\w+)?\n(.*?)```", re.DOTALL)
        # self.logger.info(f"\nCODE SNIPPET:\n{code_snippet}\n")
        for target_lang, code in pattern.findall(code_snippet):
            self.logger.info(f"\nEXTRACTED CODE:\n{code}\n")
            for lang in PROGRAMMING_LANGUAGES:
                try:
                    parser = Parser()
                    parser.set_language(get_language(lang))
                    tree = parser.parse(code)
                    errors = self.count_errors_tree_sitter(tree.root_node)
                    self.logger.info(f"\nERRORS: {errors}\n")
                    if errors < min_errors:
                        min_errors = errors
                        best_lang = lang
                        self.logger.info(f"\nIDENTIFIED LANG: {best_lang} | EXPECTED LANG: {target_lang}\n")
                except Exception:
                    self.logger.info("\nCould not identify programming language\n")
                    error_message = "Could not identify programming language"
                    raise TaskFailed(
                        message=error_message,
                        error_type=ERROR_TYPES['problem_format']
                    )
        return best_lang
    
    # --------------- generator ---------------
    def task_generator(self) -> Generator[Union[Request, List[Request]], Any, Dict[str, Any]]:
        # self.data is prepopulated with the data from the jsonl row being processed
        persona = self.data.get("professional_persona")
        persona_id = self.data.get("uuid") 
        return_dict = {
            "persona": persona,
            "persona_id": persona_id,
            "messages": [],
        }

        # Step 0.5 - Sample programming language
        programming_lang = random.choice(PROGRAMMING_LANGUAGES)  
        self.logger.info(f"\nPROGRAMMING LANG: {programming_lang}\n")
        # Step 1 – Generate problem from persona
        input_messages = self.create_instruction_request(persona=persona,
                                                         programming_lang=programming_lang
                                            )
        problem_resp: Response = yield Request({"messages": input_messages, **self.PROBLEM_GEN_PARAMS})
        problem_resp_text = problem_resp.get_text()
        problem_text = self.extract_problem(problem_resp_text)
        # Step 1.5 – Store the instruction in the return_dict
        return_dict["messages"].append(
                    {
                        "role": "user",
                        "content": problem_text,
                    })
        # Step 2 – Generate answer to the instruction
        input_messages = self.create_answer_request(problem=problem_text,
                                                    programming_lang=programming_lang)
        answer_resp: Response = yield Request({"messages": input_messages, **self.ANSWER_GEN_PARAMS})
        answer_resp_text = answer_resp.get_text()
        if programming_lang == "python":    
            valid_code = self.verify_python_code(answer_resp_text)
        else:
            valid_code = self.verify_other_code_pygments(answer_resp_text, programming_lang)
            # valid_code = self.verify_code_tree_sitter(answer_resp_text, programming_lang)
        # self.logger.info(f"\nANSWER RESPONSE:\n{answer_resp_text}\n")
        # Step 3.5 – Store the answer in the return_dict
        return_dict["messages"].append(
                    {
                        "role": "assistant",
                        "content": answer_resp_text,
                    })
        # # Step 4 - Judge answer quality
        # input_messages = self.create_judge_input(instruct_text, answer_text)
        # judge_resp = yield Request({"messages": input_messages, **self.JUDGE_PARAMS})
        # judge_resp_text = judge_resp.get_text()
        # answer_score = self.validate_judge_response(judge_resp_text)
        return return_dict