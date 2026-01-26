"""Example task – two responses + judge"""
from typing import Any, Dict, Generator, List, Union
import re
import os
import hashlib
import logging

from dispatcher.taskmanager.backend.request import Request, Response
from dispatcher.taskmanager.task.base import GeneratorTask, TaskFailed

logger = logging.getLogger(__name__)

from src.utils.text_utils import format_instructions_with_conjunctions
from src.utils.lang_id import LANG_MAP
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
        "max_tokens": 8192,
    }

    @staticmethod
    def _get_language_name() -> str:
        """Get the full language name from LANGUAGE env variable."""
        lang_code = os.environ.get('LANGUAGE', 'en').lower().strip()
        return LANG_MAP.get(lang_code, 'English')

    @staticmethod
    def _generate_task_id(instruction_ids: List[List[str]]) -> str:
        """Generate a unique task ID as a hash of the instruction_ids list."""
        # Flatten and sort instruction_ids for consistent hashing
        flat_ids = []
        for turn_ids in instruction_ids:
            flat_ids.extend(sorted(turn_ids))
        ids_str = "|".join(flat_ids)
        return hashlib.sha256(ids_str.encode()).hexdigest()[:16]

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
        # }
        # Get data fields
        queries = self.data.get("queries", [])
        queries_responses = self.data.get("queries_responses", [])
        instruction_ids = self.data.get("instruction_ids", [])
        instructions = self.data.get("instructions", [])
        instruction_categories = self.data.get("instruction_categories", [])
        eval_funcs = self.data.get("eval_funcs", [])
        
        # Generate unique task ID from instruction_ids for logging
        task_id = self._generate_task_id(instruction_ids)
        
        num_turns = len(queries)
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
        MAX_RETRIES = 5
        for turn_idx in range(num_turns):
            # Save state before attempting turn (for rollback on retry)
            saved_state = {
                'final_prompts_len': len(final_prompts),
                'queries_messages_len': len(queries_messages),
                'all_responses_len': len(all_responses),
                'all_scores_len': len(all_scores),
                'all_scoring_responses_len': len(all_scoring_responses),
                'final_instructions_len': len(final_instructions),
                'final_messages_len': len(final_messages),
            }
            
            turn_succeeded = False
            for retry_attempt in range(MAX_RETRIES):
                try:
                    # Restore state on retry (rollback any partial changes from failed attempt)
                    if retry_attempt > 0:
                        final_prompts[:] = final_prompts[:saved_state['final_prompts_len']]
                        queries_messages[:] = queries_messages[:saved_state['queries_messages_len']]
                        all_responses[:] = all_responses[:saved_state['all_responses_len']]
                        all_scores[:] = all_scores[:saved_state['all_scores_len']]
                        all_scoring_responses[:] = all_scoring_responses[:saved_state['all_scoring_responses_len']]
                        final_instructions[:] = final_instructions[:saved_state['final_instructions_len']]
                        final_messages[:] = final_messages[:saved_state['final_messages_len']]
                    
                    # Get raw query for this turn
                    raw_query = queries[turn_idx] if turn_idx < len(queries) else ""
                    
                    # Step 0a - Format the query with LLM (if there's a query)
                    if raw_query.strip():
                        with open("model_prompts/format_query_prompt.txt", "r") as f:
                            format_query_prompt = f.read().strip()
                        format_query_prompt = format_query_prompt.format(query=raw_query)
                        logger.info(f"[{task_id}] Turn {turn_idx} format query prompt: {format_query_prompt}")
                        
                        format_query_messages = [{"role": "user", "content": format_query_prompt}]
                        format_query_resp: Response = yield Request({"messages": format_query_messages, **self.GEN_PARAMS})
                        formatted_query = format_query_resp.get_text().strip()
                        logger.info(f"[{task_id}] Turn {turn_idx} formatted query: {formatted_query}")
                    else:
                        formatted_query = ""
                    
                    # Step 0b - Create per-turn keyword handler and generate keywords if needed
                    # Keywords are generated BEFORE building the prompt, so instructions already
                    # have keywords applied when we construct the prompt (no regex substitution needed)
                    keyword_handler = KeywordHandler(
                        turn_idx=turn_idx,
                        instruction_categories=instruction_categories[turn_idx] if turn_idx < len(instruction_categories) else [],
                        instructions=instructions[turn_idx] if turn_idx < len(instructions) else [],
                        instruction_ids=instruction_ids[turn_idx] if turn_idx < len(instruction_ids) else [],
                        query=queries[0] if is_no_followup else queries[turn_idx] if turn_idx < len(queries) else ""  # for no-followup case we want to potentially generate keywords for all turns that are related only to the first (and only) query
                    )

                    # Process keyword generation for this turn if needed
                    if keyword_handler.has_keyword_instructions():
                        yield from keyword_handler.process_keyword_generation(self.GEN_PARAMS)
                    
                    # Get final instructions with keyword replacements already applied
                    turn_final_instructions = keyword_handler.get_final_instructions()
                    
                    # Format instructions as bullet-point list with "-" prefix
                    instructions_bullet_list = "\n".join([f"- {instr}" for instr in turn_final_instructions])
                    
                    # Get full language name from env variable
                    language = self._get_language_name()
                    
                    # Step 0c - Build prompt from scratch using appropriate template
                    # Instructions already have keywords applied, so no post-processing needed
                    if num_turns == 1:
                        # Single turn uses the original template
                        template_file = "model_prompts/generate_response_prompt.txt"
                        with open(template_file, "r") as f:
                            prompt_template = f.read().strip()
                        current_prompt = prompt_template.format(
                            query=formatted_query,
                            instructions=instructions_bullet_list,
                            language=language
                        )
                    elif turn_idx == 0:
                        # First turn of multi-turn conversation
                        template_file = "model_prompts/generate_response_turn1_prompt.txt"
                        with open(template_file, "r") as f:
                            prompt_template = f.read().strip()
                        current_prompt = prompt_template.format(
                            query=formatted_query,
                            instructions=instructions_bullet_list,
                            language=language
                        )
                    else:
                        # Subsequent turns of multi-turn conversation
                        if is_no_followup:
                            # Use rephrase prompt (no query needed)
                            template_file = "model_prompts/rephrase_response_turnN_prompt.txt"
                            with open(template_file, "r") as f:
                                prompt_template = f.read().strip()
                            current_prompt = prompt_template.format(
                                instructions=instructions_bullet_list,
                                language=language
                            )
                        else:
                            # Use regular turnN prompt with query
                            template_file = "model_prompts/generate_response_turnN_prompt.txt"
                            with open(template_file, "r") as f:
                                prompt_template = f.read().strip()
                            current_prompt = prompt_template.format(
                                query=formatted_query,
                                instructions=instructions_bullet_list,
                                language=language
                            )
                    
                    logger.info(f"[{task_id}] Turn {turn_idx} prompt: {current_prompt}")
                    
                    # Store the final prompt
                    final_prompts.append(current_prompt)
                    
                    # Add user message for this turn
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
                    # Store final instructions (already computed with keyword replacements)
                    final_instructions.append(turn_final_instructions)

                    # Format instructions with conjunctions for final user_content
                    formatted_instructions = format_instructions_with_conjunctions(turn_final_instructions)
                    
                    # Build user_content by concatenating formatted_query (already computed) with formatted_instructions
                    if formatted_query:
                        user_content = f"{formatted_query} {formatted_instructions}"
                    else:
                        # No query (constraints-only turn): just use formatted instructions
                        user_content = formatted_instructions
                    
                    logger.info(f"[{task_id}] Turn {turn_idx} user_content: {user_content}")
                    
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
                    
                    # Turn succeeded, exit retry loop
                    turn_succeeded = True
                    break
                    
                except TaskFailed as tf:
                    # Store the exception for potential re-raise
                    last_exception = tf
                    # Continue to next retry attempt
                    continue
            
            # If turn failed after all retries, re-raise the last exception
            # This allows the dispatcher to handle the error gracefully and dump an error record
            if not turn_succeeded:
                if turn_idx > 0:
                    break # this allows to save at least the already processed turns which results in valid conversations only with less turns (which is still good data)
                raise last_exception
        
        # Dump eval_funcs as dicts with instruction_ids as keys
        eval_funcs_dict = {}
        for turn_idx, turn_instruction_ids in enumerate(instruction_ids):
            for instr_idx, instruction_id in enumerate(turn_instruction_ids):
                if instruction_id not in eval_funcs_dict and turn_idx < len(eval_funcs) and instr_idx < len(eval_funcs[turn_idx]):
                    eval_funcs_dict[instruction_id] = eval_funcs[turn_idx][instr_idx]

        # Return results
        return {
            'uuid': task_id,
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
