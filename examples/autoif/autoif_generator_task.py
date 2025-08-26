"""Example task – two responses + judge"""
from typing import Any, Dict, Generator, List, Union
import re
import os

from dispatcher.taskmanager.backend.request import Request, Response
from dispatcher.taskmanager.task.base import GeneratorTask, TaskFailed

from src.utils.response_handler import response_verify, extract_score, construct_scoring_messages, format_instructions_with_conjunctions, is_no_followup_case, construct_rephrase_scoring_messages

__all__ = ["GenerateQueryResponsesTask"]


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
        #     'instruction_ids': instruction_ids, # backwards compatible with 'instruction_id'
        #     'instructions': instructions, # backwards compatible with 'instruction'
        #     'queries': queries, # backwards compatible with 'query'
        #     'queries_responses': queries_responses, # backwards compatible with 'query_response'
        #     'query_metadata': query_metadata,
        #     'eval_funcs': [["def evaluate():...",],], # backwards compatible with 'eval_func' ["def evaluate():...",]
        #     'cases': [[{'input', 'output'},],], # backwards compatible with 'cases' [{'input', 'output'},]
        #     'prompts': prompts # backwards compatible with 'prompt'
        # }
        
        # Determine if this is multi-turn (check for plural keys)
        prompts = self.data.get("prompts", [self.data.get("prompt")] if self.data.get("prompt") else [])
        queries = self.data.get("queries", [self.data.get("query")] if self.data.get("query") else [])
        queries_responses = self.data.get("queries_responses", [self.data.get("query_response")] if self.data.get("query_response") else [])
        
        # Handle instructions - can be list of lists (multi-turn) or single list/string (single-turn)
        instructions_data = self.data.get("instructions", self.data.get("instruction", []))
        if not isinstance(instructions_data[0], list) if instructions_data else True:
            # Single-turn format: wrap in list
            instructions_per_turn = [instructions_data] if instructions_data else [[]]
        else:
            # Multi-turn format: already list of lists
            instructions_per_turn = instructions_data
        
        # Handle instruction_ids similarly
        instruction_ids_data = self.data.get("instruction_ids", self.data.get("instruction_id", []))
        if not isinstance(instruction_ids_data[0], list) if instruction_ids_data else True:
            instruction_ids_per_turn = [instruction_ids_data] if instruction_ids_data else [[]]
        else:
            instruction_ids_per_turn = instruction_ids_data
        
        # Handle eval_funcs and cases
        eval_funcs_data = self.data.get("eval_funcs", self.data.get("eval_func", []))
        if not isinstance(eval_funcs_data[0], list) if eval_funcs_data else True:
            eval_funcs_per_turn = [eval_funcs_data] if eval_funcs_data else [[]]
        else:
            eval_funcs_per_turn = eval_funcs_data
            
        cases_data = self.data.get("cases", [])
        if not isinstance(cases_data[0], list) if cases_data else True:
            cases_per_turn = [cases_data] if cases_data else [[]]
        else:
            cases_per_turn = cases_data
        
        num_turns = len(prompts)
        queries_messages = []
        all_responses = []
        all_scores = []
        all_scoring_responses = []
        final_messages = []
        
        # Determine if this is a no_followup case
        is_no_followup = is_no_followup_case(queries)
        
        # Process each turn
        for turn_idx in range(num_turns):
            # Add user message for this turn
            queries_messages.append({
                "role": "user",
                "content": prompts[turn_idx]
            })
            
            # Step 1 – get response for the current turn
            queries_resp: Response = yield Request({"messages": queries_messages, **self.GEN_PARAMS})
            response_text = queries_resp.get_text()
            all_responses.append(response_text)
            
            # Step 2 - verify response 
            # For multi-turn, we need to verify against all instructions from all turns up to current turn
            verification_data = self._prepare_verification_data(turn_idx, instruction_ids_per_turn, 
                                                               instructions_per_turn, eval_funcs_per_turn, 
                                                               cases_per_turn)
            response_verify(response_text, verification_data, turn=turn_idx)
            
            # Step 3 - score the response for this turn
            if is_no_followup and turn_idx > 0:
                # For no_followup case after first turn, use rephrase scoring
                scoring_messages = construct_rephrase_scoring_messages(
                    response_text, all_responses[0], queries[0], 
                    instruction_ids_per_turn, instructions_per_turn, turn_idx
                )
            else:
                # Regular scoring - data already contains accumulated constraints
                scoring_data = self._prepare_scoring_data(turn_idx, instruction_ids_per_turn, 
                                                         instructions_per_turn, eval_funcs_per_turn, 
                                                         cases_per_turn)
                scoring_messages = construct_scoring_messages(response_text, scoring_data)
            
            scored_resp: Response = yield Request({"messages": scoring_messages, **self.GEN_PARAMS})
            scoring_text = scored_resp.get_text()
            score = extract_score(scoring_text, turn=turn_idx)
            
            all_scores.append(score)
            all_scoring_responses.append(scoring_text)
            
            # Check if score meets threshold - if not, fail the task
            if score < self.SCORE_THRESHOLD:
                raise TaskFailed(
                    message=f"Score {score} at turn {turn_idx + 1} is below threshold {self.SCORE_THRESHOLD}. Scoring response: <response>{scoring_text}</response>",
                    error_type=f"turn{turn_idx + 1}_score_below_threshold"
                )
            
            # Add assistant response to conversation for next turn
            queries_messages.append({
                "role": "assistant",
                "content": response_text
            })
            
            # Build final messages format for this turn
            query = queries[turn_idx] if turn_idx < len(queries) else ""
            instructions_text = format_instructions_with_conjunctions(instructions_per_turn[turn_idx])
            
            # Only include query if it's not empty
            if query.strip():
                if not re.search(r'[.!?]$', query):
                    query += "."
                user_content = f"{query} {instructions_text}"
            else:
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
        
        # Return results - use plural format for consistency
        return {
            'instruction_ids': instruction_ids_per_turn,
            'instructions': instructions_per_turn,
            'queries': queries,
            'queries_responses': queries_responses,
            'query_metadata': self.data.get("query_metadata"),
            'responses': all_responses,
            'eval_funcs': eval_funcs_per_turn,
            'cases': cases_per_turn,
            'prompts': prompts,
            'messages': final_messages,
            'scores': all_scores,
            'scoring_responses': all_scoring_responses
        }
    
    def _prepare_verification_data(self, turn_idx: int, instruction_ids_per_turn: List[List], 
                                  instructions_per_turn: List[List], eval_funcs_per_turn: List[List], 
                                  cases_per_turn: List[List]) -> Dict[str, Any]:
        """Prepare verification data for a specific turn, including all previous turn constraints."""
        # For verification, we need to check against all instructions from turn 0 to current turn
        cumulative_instruction_ids = []
        cumulative_instructions = []
        cumulative_eval_funcs = []
        cumulative_cases = []
        
        for i in range(turn_idx + 1):
            if i < len(instruction_ids_per_turn):
                cumulative_instruction_ids.extend(instruction_ids_per_turn[i])
            if i < len(instructions_per_turn):
                cumulative_instructions.extend(instructions_per_turn[i])
            if i < len(eval_funcs_per_turn):
                cumulative_eval_funcs.extend(eval_funcs_per_turn[i])
            if i < len(cases_per_turn):
                cumulative_cases.extend(cases_per_turn[i])
        
        return {
            'instruction_ids': cumulative_instruction_ids,
            'instructions': cumulative_instructions,
            'eval_funcs': cumulative_eval_funcs,
            'cases': cumulative_cases,
            **{k: v for k, v in self.data.items() if k not in ['instruction_ids', 'instructions', 'eval_funcs', 'cases']}
        }
    
    def _prepare_scoring_data(self, turn_idx: int, instruction_ids_per_turn: List[List], 
                             instructions_per_turn: List[List], eval_funcs_per_turn: List[List], 
                             cases_per_turn: List[List]) -> Dict[str, Any]:
        """Prepare scoring data for a specific turn, including accumulated constraints."""
        # For scoring, accumulate all constraints up to current turn
        cumulative_instruction_ids = []
        cumulative_instructions = []
        
        for i in range(turn_idx + 1):
            if i < len(instruction_ids_per_turn):
                cumulative_instruction_ids.extend(instruction_ids_per_turn[i])
            if i < len(instructions_per_turn):
                cumulative_instructions.extend(instructions_per_turn[i])
        
        return {
            'instruction_ids': cumulative_instruction_ids,
            'instructions': cumulative_instructions,
            'eval_funcs': eval_funcs_per_turn[turn_idx] if turn_idx < len(eval_funcs_per_turn) else [],
            'cases': cases_per_turn[turn_idx] if turn_idx < len(cases_per_turn) else [],
            **{k: v for k, v in self.data.items() if k not in ['instruction_ids', 'instructions', 'eval_funcs', 'cases']}
        }
