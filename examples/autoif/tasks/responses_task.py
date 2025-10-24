"""Example task – two responses + judge"""
from typing import Any, Dict, Generator, List, Union
import re
import os

from dispatcher.taskmanager.backend.request import Request, Response
from dispatcher.taskmanager.task.base import GeneratorTask, TaskFailed

from src.utils.text_utils import format_instructions_with_conjunctions
from src.keyword_handler import KeywordHandler
from src.scoring_handler import ScoringHandler
from src.verification_handler import VerificationHandler

__all__ = ["GenerateQueryResponsesTask"]

def is_no_followup_case(queries: List[str]) -> bool:
    """Check if this is a no_followup case (queries after first are empty)."""
    if len(queries) <= 1:
        return False
    # Check if all queries after the first one are empty
    return all(query.strip() == "" for query in queries[1:])

class GenerateQueryResponsesTask(GeneratorTask):
    """Generate query responses with verifiers"""

    # Fixed generation hyper‑parameters for candidate answers
    GEN_PARAMS: Dict[str, Any] = {
        "temperature": 0.7,
        "top_p": 0.95,
        "max_tokens": 4096,
    }
    
    # Score threshold for accepting responses
    SCORE_THRESHOLD = int(os.environ.get("SCORE_THRESHOLD", "4"))

    # --------------- generator ---------------
    def task_generator(self) -> Generator[Union[Request, List[Request]], Any, Dict[str, Any]]:
        # self.data is prepopulated with the data from the jsonl row being
        # processed
        # these are queries with verifiers of format
        # {
        #     'instruction_ids': [[instruction_ids_turn1], [instruction_ids_turn2], ...],
        #     'instructions': [[instructions_turn1], [instructions_turn2], ...],
        #     'instruction_categories': [[categories_turn1], [categories_turn2], ...],
        #     'queries': [query1, query2, ...],
        #     'queries_responses': [response1, response2, ...],
        #     'query_metadata': query_metadata,
        #     'eval_funcs': [["def evaluate():...",], ["def evaluate():...",], ...],
        #     'prompts': [prompt1, prompt2, ...]
        # }
        # Get data fields
        prompts = self.data.get("prompts", [])
        queries = self.data.get("queries", [])
        queries_responses = self.data.get("queries_responses", [])
        instruction_ids = self.data.get("instruction_ids", [])
        instructions = self.data.get("instructions", [])
        instruction_categories = self.data.get("instruction_categories", [])
        eval_funcs = self.data.get("eval_funcs", [])
        
        num_turns = len(prompts)
        queries_messages = []
        all_responses = []
        all_scores = []
        all_scoring_responses = []
        final_messages = []
        final_instructions = []
        final_prompts = []
        
        # Determine if this is a no_followup case
        # meaning next turns after the first one will ask to rephrase the previous response 
        # instead of following up with another user question
        is_no_followup = is_no_followup_case(queries)
        
        # Process each turn
        for turn_idx in range(num_turns):
            current_prompt = prompts[turn_idx]
            # Step 0 (optional) - Create per-turn keyword handler and process keywords if needed
            keyword_handler = KeywordHandler(
                turn_idx=turn_idx,
                instruction_categories=instruction_categories[turn_idx] if turn_idx < len(instruction_categories) else [],
                instructions=instructions[turn_idx] if turn_idx < len(instructions) else [],
                instruction_ids=instruction_ids[turn_idx] if turn_idx < len(instruction_ids) else [],
                query=queries[0] if is_no_followup else queries[turn_idx] if turn_idx < len(queries) else "" # for no-followup case we want to potentially generate keywords for all turns that are related only to the first (and only) query
            )

            # Process keyword generation for this turn if needed
            if keyword_handler.has_keyword_instructions():
                yield from keyword_handler.process_keyword_generation(self.GEN_PARAMS)
                current_prompt = keyword_handler.apply_keyword_modifications_to_prompt(current_prompt)
            
            # Store the final prompt (potentially modified with keywords)
            final_prompts.append(current_prompt)
            
            # Add user message for this turn (potentially modified with new keywords)
            queries_messages.append({
                "role": "user",
                "content": current_prompt
            })
            
            # Step 1 – get response for the current turn
            queries_resp: Response = yield Request({"messages": queries_messages, **self.GEN_PARAMS})
            response_text = queries_resp.get_text()
            all_responses.append(response_text)
            
            # Step 2 - verify response
            verification_handler = VerificationHandler(
                turn_idx=turn_idx,
                instruction_ids=instruction_ids,
                instructions=instructions,
                eval_funcs=eval_funcs,
                instruction_categories=instruction_categories,
                keyword_handler=keyword_handler
            )
            verification_handler.verify_response(response_text)

            # Step 3 - score the response for this turn
            scoring_handler = ScoringHandler(
                turn_idx=turn_idx,
                is_no_followup=is_no_followup,
                instruction_ids=instruction_ids,
                instructions=instructions,
                queries=queries,
                all_responses=all_responses
            )

            scoring_messages = scoring_handler.construct_scoring_messages(response_text)
            scored_resp: Response = yield Request({"messages": scoring_messages, **self.GEN_PARAMS})
            scoring_text = scored_resp.get_text()

            # Extract score and check threshold
            score = scoring_handler.extract_and_check_score(scoring_text)

            all_scores.append(score)
            all_scoring_responses.append(scoring_text)
            
            # Add assistant response to conversation for next turn
            queries_messages.append({
                "role": "assistant",
                "content": response_text
            })
            
            # Build final messages format for this turn
            query = queries[turn_idx] if turn_idx < len(queries) else ""
            
            # Get final instructions with keyword replacements applied (if any)
            turn_final_instructions = keyword_handler.get_final_instructions()
            final_instructions.append(turn_final_instructions)
            
            # Format instructions with proper conjunctions
            instructions_text = format_instructions_with_conjunctions(turn_final_instructions)
            
            # Construct user_content consistently for both keyword and non-keyword cases
            if query.strip():
                if not re.search(r'[.!?]$', query):
                    query += "."
                user_content = f"{query} {instructions_text}"
            else: # rephrasing message - only instructions without any "query"
                user_content = instructions_text
            
            final_messages.extend([
                {
                    "role": "user", 
                    "content": user_content
                },
                {
                    "role": "assistant", 
                    "content": response_text
                }
            ])
        
        # Dump eval_funcs as dicts with instruction_ids as keys
        eval_funcs_dict = {}
        for turn_idx, turn_instruction_ids in enumerate(instruction_ids):
            for instr_idx, instruction_id in enumerate(turn_instruction_ids):
                if instruction_id not in eval_funcs_dict and turn_idx < len(eval_funcs) and instr_idx < len(eval_funcs[turn_idx]):
                    eval_funcs_dict[instruction_id] = eval_funcs[turn_idx][instr_idx]

        # Return results
        return {
            'instruction_ids': instruction_ids,
            'instructions': final_instructions,
            'instruction_categories': instruction_categories,
            'queries': queries,
            'queries_responses': queries_responses,
            'query_metadata': self.data.get("query_metadata"),
            'responses': all_responses,
            'eval_funcs': eval_funcs_dict,
            'prompts': final_prompts,
            'messages': final_messages,
            'scores': all_scores,
            'scoring_responses': all_scoring_responses
        }
