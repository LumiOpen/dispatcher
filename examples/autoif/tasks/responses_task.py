"""Generate scored multi-turn responses for the AutoIF pipeline.

Reads the new ``queries_constraints_verifiers.jsonl`` format produced by
``concat_queries.py``, resolves placeholders, generates responses,
verifies them with eval_funcs, and scores for relevance.
"""

from typing import Any, Dict, Generator, List, Union
import re
import os
import json
import hashlib
import logging

import numpy as np

from dispatcher.taskmanager.backend.request import Request, Response
from dispatcher.taskmanager.task.base import GeneratorTask, TaskFailed

from src.placeholder_resolver import (
    StaticPlaceholderResolver,
    LLMPlaceholderResolver,
    format_value,
)
from src.utils.function_executor import FunctionExecutor
from src.utils.lang_id import get_env_language_name, detect_language
from src.utils.text_utils import format_constraints_with_conjunctions
from src.utils.error_utils import format_error_type_with_turn

logger = logging.getLogger(__name__)

__all__ = ["GenerateResponsesTask"]

# Score extraction patterns (ordered by specificity)
_SCORE_PATTERNS = [
    r'\*\*Score:\s*(\d+)\*\*\s*$',
    r'`Score:\s*(\d+)`\s*$',
    r'\(Score\s*:?\s*(\d+)\)\s*$',
    r'Score:\s*(\d+)\s*$',
    r'Score\s*:?\s*(\d+)\s*$',
]


class GenerateResponsesTask(GeneratorTask):
    """Generate responses from the new constraints/turns input format."""

    GEN_PARAMS: Dict[str, Any] = {
        "temperature": 0.7,
        "top_p": 0.95,
        "max_tokens": 8192,
    }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_task_id(constraint_ids_per_turn: List[List[str]]) -> str:
        flat_ids = []
        for turn_ids in constraint_ids_per_turn:
            flat_ids.extend(sorted(turn_ids))
        return hashlib.sha256("|".join(flat_ids).encode()).hexdigest()[:16]

    # ------------------------------------------------------------------
    # Verification helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _check_error(response: str, turn: int) -> None:
        """Raise TaskFailed if the response contains an error JSON."""

        def _extract_json(text: str):
            m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
            if m:
                try:
                    return json.loads(m.group(1))
                except json.JSONDecodeError:
                    pass
            for m in re.finditer(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL):
                try:
                    parsed = json.loads(m.group())
                    if isinstance(parsed, dict):
                        return parsed
                except json.JSONDecodeError:
                    continue
            try:
                parsed = json.loads(text.strip())
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass
            return None

        data = _extract_json(response)
        if data and "error" in data:
            err = data["error"]
            if err == "contradicting_constraints":
                raise TaskFailed(
                    message=f"Expected error 'contradicting_constraints' found in response: {response}",
                    error_type=format_error_type_with_turn("contradicting_constraints", turn),
                )
            raise TaskFailed(
                message=f"Error found in response: {err}",
                error_type=format_error_type_with_turn("error_in_response", turn),
            )

    @staticmethod
    def _check_language(response: str, turn: int) -> None:
        """Raise TaskFailed if the response is not in the target language."""
        target = os.environ.get("LANGUAGE")
        if not target:
            return
        try:
            code3, code2 = detect_language(response)
        except Exception as e:
            raise TaskFailed(
                message=f"Language detection error: {e} <response>{response}</response>",
                error_type=format_error_type_with_turn("language_detection", turn),
            )
        valid = {code3, code2} if code2 is not None else {code3}
        if target not in valid:
            raise TaskFailed(
                message=f"Response not in expected language {target}, got {valid} <response>{response}</response>",
                error_type=format_error_type_with_turn("invalid_language", turn),
            )

    @staticmethod
    def _run_eval_funcs(
        response: str,
        constraint_ids: List[str],
        eval_funcs_by_id: Dict[str, List[str]],
        resolved_values: Dict[str, Dict[str, Any]],
        turn: int,
    ) -> None:
        """Run evaluation functions for each constraint; raise on failure."""
        executor = FunctionExecutor()
        failed = []

        for cid in constraint_ids:
            funcs = eval_funcs_by_id.get(cid, [])
            if not funcs:
                raise TaskFailed(
                    message=f"No evaluation functions for constraint {cid}",
                    error_type=format_error_type_with_turn("no_eval_functions_for_constraint", turn),
                )
            kwargs = resolved_values.get(cid, {})
            results = []
            for func in funcs:
                try:
                    r = executor.execute_with_response(func, response, **kwargs)
                    if r is not None:
                        results.append(r)
                except Exception as e:
                    raise TaskFailed(
                        message=f"Error executing eval func for constraint {cid}: {e} <response>{response}</response>",
                        error_type=format_error_type_with_turn("function_execution_failed", turn),
                    )
            acc = float(np.mean(results)) if results else 0.0
            if acc <= 0:
                failed.append((cid, acc))

        if failed:
            if len(failed) == 1:
                cid, acc = failed[0]
                raise TaskFailed(
                    message=f"Verification failed for constraint {cid} (acc={acc}) <response>{response}</response>",
                    error_type=format_error_type_with_turn("constraint_verification_failed", turn),
                )
            ids_str = ", ".join(str(c) for c, _ in failed)
            raise TaskFailed(
                message=f"Verification failed for constraints {ids_str} <response>{response}</response>",
                error_type=format_error_type_with_turn("multiple_constraints_verification_failed", turn),
            )

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_scoring_messages(
        turn_idx: int,
        is_rephrase: bool,
        turn_constraints: List[str],
        query: str,
        response_text: str,
        first_response: str,
    ) -> List[Dict[str, str]]:
        """Build messages for the scoring LLM call."""
        if is_rephrase:
            with open("model_prompts/scoring_rephrase_prompt.txt", "r") as f:
                tmpl = f.read().strip()
            prompt = tmpl.format(
                query=query,
                previous_turn_response=first_response,
                current_response=response_text,
                constraints=format_constraints_with_conjunctions(turn_constraints),
            )
        else:
            with open("model_prompts/scoring_prompt.txt", "r") as f:
                tmpl = f.read().strip()
            prompt = tmpl.format(
                constraints=turn_constraints,
                query=query,
                response=response_text,
            )
        return [{"role": "user", "content": prompt}]

    @staticmethod
    def _extract_and_check_score(scoring_text: str, turn_idx: int) -> int:
        """Extract score from scoring text; raise if below threshold."""
        threshold = int(os.environ.get("SCORE_THRESHOLD", "4"))

        for pattern in _SCORE_PATTERNS:
            match = re.search(pattern, scoring_text, re.IGNORECASE)
            if match:
                try:
                    score = int(match.group(1))
                    if score < threshold:
                        raise TaskFailed(
                            message=f"Score {score} at turn {turn_idx + 1} is below threshold {threshold}. Scoring response: <response>{scoring_text}</response>",
                            error_type=format_error_type_with_turn("score_below_threshold", turn_idx),
                        )
                    return score
                except (ValueError, IndexError):
                    continue

        raise TaskFailed(
            message=f"Score not found in the scoring response: {scoring_text}",
            error_type=format_error_type_with_turn("score_extraction_failed", turn_idx),
        )

    # ------------------------------------------------------------------
    # Main generator
    # ------------------------------------------------------------------

    def task_generator(self) -> Generator[Union[Request, List[Request]], Any, Dict[str, Any]]:
        # ---- Step 1: Parse input ----
        query = self.data["messages"][0]["content"]
        constraints = self.data["constraints"]          # Dict[str, constraint_info]
        turns = self.data["turns"]                      # List[{constraint_ids: [...]}]
        query_metadata = self.data.get("query_metadata", {})

        # Collect constraint_ids per turn for task ID
        constraint_ids_per_turn = [t["constraint_ids"] for t in turns]
        task_id = self._generate_task_id(constraint_ids_per_turn)
        num_turns = len(turns)

        # Build eval_funcs lookup: constraint_id -> list of func strings
        eval_funcs_by_id: Dict[str, List[str]] = {}
        for cid, cinfo in constraints.items():
            eval_funcs_by_id[cid] = cinfo.get("eval_funcs", [])

        # ---- Step 2: Resolve all placeholders (once, before turn loop) ----
        placeholder_lookup_file = os.environ.get("PLACEHOLDER_LOOKUP_FILE", "")
        placeholder_lookup: Dict[str, Any] = {}
        if placeholder_lookup_file:
            try:
                with open(placeholder_lookup_file, "r") as f:
                    placeholder_lookup = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                logger.warning(f"[{task_id}] Could not load placeholder lookup from {placeholder_lookup_file}")

        static_resolver = StaticPlaceholderResolver(placeholder_lookup)
        llm_resolver = LLMPlaceholderResolver()

        # Phase A: resolve static placeholders immediately
        resolved_values: Dict[str, Dict[str, Any]] = {}  # cid -> {name: value}
        for cid, cinfo in constraints.items():
            placeholders = cinfo.get("placeholders", {})
            if not placeholders:
                continue
            vals = static_resolver.resolve(placeholders)
            if vals:
                resolved_values.setdefault(cid, {}).update(vals)

        # Phase B: build one LLM request per constraint for remaining placeholders
        pending_requests: List[Request] = []
        for cid, cinfo in constraints.items():
            placeholders = cinfo.get("placeholders", {})
            if not placeholders:
                continue
            req = llm_resolver.build_request(cid, cinfo["constraint"], placeholders, query, self.GEN_PARAMS)
            if req is not None:
                pending_requests.append(req)

        # Phase C: yield batch and collect responses
        if pending_requests:
            responses: Union[Response, List[Response]] = yield pending_requests
            if isinstance(responses, Response):
                responses = [responses]

            # Phase D: parse responses into resolved_values
            for req, resp in zip(pending_requests, responses):
                cid = req.context["constraint_id"]
                try:
                    vals = llm_resolver.parse_response(resp.get_text())
                except Exception as e:
                    raise TaskFailed(
                        message=f"Failed to parse placeholder response for constraint {cid}: {e}. Response: {resp.get_text()}",
                        error_type="placeholder_resolution_failed",
                    )
                resolved_values.setdefault(cid, {}).update(vals)

        # ---- Step 3: Build resolved constraint texts ----
        resolved_constraints: Dict[str, str] = {}
        for cid, cinfo in constraints.items():
            text = cinfo["constraint"]
            for pname, pval in resolved_values.get(cid, {}).items():
                text = text.replace(f"{{{pname}}}", format_value(pval))
            resolved_constraints[cid] = text

        logger.info(f"[TASK:{task_id}] Resolved constraints: {resolved_constraints}")

        # ---- Step 4: Per-turn generation loop ----
        MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "5"))
        language = get_env_language_name()
        is_no_followup = True  # always true – single query, multiple rephrase turns

        queries_messages: List[Dict[str, str]] = []
        output_turns: List[Dict[str, Any]] = []
        all_responses: List[str] = []
        final_messages: List[Dict[str, str]] = []
        formatted_query: str = ""

        for turn_idx in range(num_turns):
            # Save state for rollback on retry
            saved = {
                "queries_messages": len(queries_messages),
                "all_responses": len(all_responses),
                "output_turns": len(output_turns),
                "final_messages": len(final_messages),
            }

            turn_succeeded = False
            last_exception = None

            for retry in range(MAX_RETRIES):
                try:
                    # Rollback on retry
                    if retry > 0:
                        queries_messages[:] = queries_messages[:saved["queries_messages"]]
                        all_responses[:] = all_responses[:saved["all_responses"]]
                        output_turns[:] = output_turns[:saved["output_turns"]]
                        final_messages[:] = final_messages[:saved["final_messages"]]

                    # a. Get this turn's constraint_ids and resolved texts
                    turn_cids = turns[turn_idx]["constraint_ids"]
                    turn_constraint_texts = [resolved_constraints[cid] for cid in turn_cids]

                    # b. Format constraints as bullet-point list
                    constraints_bullet_list = "\n".join(f"- {t}" for t in turn_constraint_texts)

                    # b2. Format query (turn 0 only)
                    if turn_idx == 0:
                        with open("model_prompts/format_query_prompt.txt", "r") as f:
                            fmt_prompt = f.read().strip()
                        fmt_prompt = fmt_prompt.format(query=query, language=language)
                        logger.info(f"[TASK:{task_id},RETRY:{retry}] Turn 0 format query prompt: {fmt_prompt}")

                        fmt_resp: Response = yield Request({"messages": [{"role": "user", "content": fmt_prompt}], **self.GEN_PARAMS})
                        formatted_query = fmt_resp.get_text().strip()
                        logger.info(f"[TASK:{task_id},RETRY:{retry}] Turn 0 formatted query: {formatted_query}")

                    # c. Build prompt
                    if num_turns == 1:
                        template_file = "model_prompts/generate_response_prompt.txt"
                    elif turn_idx == 0:
                        template_file = "model_prompts/generate_response_turn1_prompt.txt"
                    else:
                        template_file = "model_prompts/rephrase_response_turnN_prompt.txt"

                    with open(template_file, "r") as f:
                        tmpl = f.read().strip()

                    if turn_idx == 0 or num_turns == 1:
                        current_prompt = tmpl.format(
                            query=formatted_query,
                            constraints=constraints_bullet_list,
                            language=language,
                        )
                    else:
                        current_prompt = tmpl.format(
                            constraints=constraints_bullet_list,
                            language=language,
                        )

                    logger.info(f"[TASK:{task_id},RETRY:{retry}] Turn {turn_idx} prompt: {current_prompt}")

                    queries_messages.append({"role": "user", "content": current_prompt})

                    # d. Generate response
                    gen_resp: Response = yield Request({"messages": queries_messages, **self.GEN_PARAMS})
                    response_text = gen_resp.get_text()
                    all_responses.append(response_text)

                    # e. Verify response
                    self._check_error(response_text, turn_idx)
                    self._check_language(response_text, turn_idx)
                    self._run_eval_funcs(response_text, turn_cids, eval_funcs_by_id, resolved_values, turn_idx)

                    # f. Score relevance
                    is_rephrase = is_no_followup and turn_idx > 0
                    scoring_messages = self._build_scoring_messages(
                        turn_idx, is_rephrase, turn_constraint_texts,
                        query, response_text, all_responses[0] if all_responses else "",
                    )
                    scored_resp: Response = yield Request({"messages": scoring_messages, **self.GEN_PARAMS})
                    scoring_text = scored_resp.get_text()
                    score = self._extract_and_check_score(scoring_text, turn_idx)

                    # Add assistant message for conversation continuity
                    queries_messages.append({"role": "assistant", "content": response_text})

                    # g. Store per-turn output
                    output_turns.append({
                        "constraint_ids": turn_cids,
                        "response": response_text,
                        "prompt": current_prompt,
                        "score": score,
                        "scoring_response": scoring_text,
                    })

                    formatted_constraints = format_constraints_with_conjunctions(turn_constraint_texts)
                    if formatted_query and turn_idx == 0:
                        user_content = f"{formatted_query} {formatted_constraints}"
                    else:
                        user_content = formatted_constraints

                    logger.info(f"[TASK:{task_id},RETRY:{retry}] Turn {turn_idx} user_content: {user_content}")

                    final_messages.extend([
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": response_text},
                    ])

                    turn_succeeded = True
                    break

                except TaskFailed as tf:
                    last_exception = tf
                    continue

            if not turn_succeeded:
                if turn_idx > 0:
                    # Return partial results for earlier turns
                    break
                raise last_exception

        # ---- Build output constraints dict (mirrors input structure, with resolved text) ----
        used_cids = set()
        for turn in output_turns:
            used_cids.update(turn["constraint_ids"])

        output_constraints: Dict[str, Dict[str, Any]] = {}
        for cid in used_cids:
            cinfo = constraints[cid]
            output_constraints[cid] = {
                "constraint": resolved_constraints[cid],
                "category": cinfo.get("category", "default"),
                "eval_funcs": cinfo.get("eval_funcs", []),
            }

        return {
            "uuid": task_id,
            "query": query,
            "query_metadata": query_metadata,
            "constraints": output_constraints,
            "turns": output_turns,
            "messages": final_messages,
        }
