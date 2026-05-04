"""Generate scored multi-turn responses for the AutoIF pipeline.

Reads the new ``queries_constraints_verifiers.jsonl`` format produced by
``concat_queries.py``, resolves placeholders, generates responses,
verifies them with eval_funcs, and scores for relevance.
"""

from typing import Any, Dict, Generator, List, Optional, Set, Union
import re
import os
import json
import logging
import threading
import uuid

import numpy as np

from dispatcher.taskmanager.backend.request import Request, Response
from dispatcher.taskmanager.task.base import GeneratorTask, TaskFailed

from src.placeholder_resolver import (
    StaticPlaceholderResolver,
    LLMPlaceholderResolver,
    format_value,
)
from src.utils.function_executor import FunctionExecutor, set_worker_pool
from src.utils.lang_id import get_env_language_name, detect_language
from src.utils.text_utils import format_constraints_with_conjunctions
from src.utils.error_utils import format_error_type_with_turn
from src.utils.model_preloader import (
    detect_nlp_model_dependencies,
    preload_model_dependencies,
)

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

    _runtime_lock = threading.Lock()
    _worker_pool = None
    _pool_modules: List[str] = []
    _loaded_model_dependencies: Dict[str, Set[str]] = {
        "spacy": set(),
        "stanza": set(),
        "trankit": set(),
        "nltk": set(),
    }

    @classmethod
    def setup(cls) -> None:
        """Prepare response-generation runtime.

        Worker-pool startup is deferred until task-specific verifier functions
        have been scanned and any required NLP models have been installed into
        the active runtime environment.
        """
        logger.info("Response task setup complete; verifier runtime will initialize lazily")

    GEN_PARAMS: Dict[str, Any] = {
        "temperature": 0.7,
        "top_p": 0.95,
        "max_tokens": 8192,
    }

    @classmethod
    def _get_preimport_modules(cls) -> List[str]:
        modules_str = os.environ.get("PREIMPORT_MODULES", "")
        return [m.strip() for m in modules_str.split(",") if m.strip()]

    @classmethod
    def _ensure_eval_runtime(cls, all_function_strings: List[str]) -> None:
        """Install task-specific NLP models before spawning verifier workers."""
        dependencies = detect_nlp_model_dependencies(all_function_strings)
        missing_dependencies: Dict[str, List[str]] = {
            name: [item for item in items if item not in cls._loaded_model_dependencies[name]]
            for name, items in dependencies.items()
        }
        modules = cls._get_preimport_modules()

        with cls._runtime_lock:
            # Recompute under the lock in case another task initialized first.
            missing_dependencies = {
                name: [item for item in items if item not in cls._loaded_model_dependencies[name]]
                for name, items in dependencies.items()
            }

            if any(missing_dependencies.values()):
                logger.info(
                    "Response runtime preloading missing NLP models: spacy=%s stanza=%s trankit=%s nltk=%s",
                    missing_dependencies["spacy"],
                    missing_dependencies["stanza"],
                    missing_dependencies["trankit"],
                    missing_dependencies["nltk"],
                )
                loaded = preload_model_dependencies(
                    missing_dependencies,
                    logger,
                    spacy_install_mode="current",
                )
                for name, items in loaded.items():
                    cls._loaded_model_dependencies[name].update(items)

                unresolved = {
                    name: sorted(set(missing_dependencies[name]) - cls._loaded_model_dependencies[name])
                    for name in missing_dependencies
                    if missing_dependencies[name]
                }
                unresolved = {name: items for name, items in unresolved.items() if items}
                if unresolved:
                    logger.warning("Response runtime still missing NLP models: %s", unresolved)

                if cls._worker_pool is not None:
                    cls._worker_pool.shutdown()
                    cls._worker_pool = None
                    cls._pool_modules = []
                    set_worker_pool(None)
                    logger.info("WorkerPool restarted after runtime model changes")

            if not modules:
                return

            if cls._worker_pool is None or cls._pool_modules != modules:
                from src.utils.worker_pool import WorkerPool

                n_workers = int(os.environ.get("EVAL_WORKERS", "4"))
                cls._worker_pool = WorkerPool.create(
                    num_workers=n_workers,
                    preimport_modules=modules,
                )
                cls._pool_modules = modules.copy()
                set_worker_pool(cls._worker_pool)
                logger.info(
                    "WorkerPool active (%d workers, pre-imported: %s)",
                    n_workers,
                    ", ".join(modules),
                )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_sample_uuid() -> str:
        return uuid.uuid4().hex

    @staticmethod
    def _build_log_ref(
        sample_uuid: str,
        stage: str,
        turn_idx: Optional[int] = None,
        retry: Optional[int] = None,
    ) -> Dict[str, Any]:
        log_ref: Dict[str, Any] = {
            "sample_uuid": sample_uuid,
            "stage": stage,
        }
        if turn_idx is not None:
            log_ref["turn_index"] = turn_idx
        if retry is not None:
            log_ref["retry"] = retry
        return log_ref

    @classmethod
    def _build_error_entry(
        cls,
        sample_uuid: str,
        error_type: str,
        stage: str,
        turn_idx: Optional[int] = None,
        retry: Optional[int] = None,
        constraint_ids: Optional[List[str]] = None,
        function_index: Optional[int] = None,
        extra_fields: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        entry: Dict[str, Any] = {
            "error_type": error_type,
            "stage": stage,
            "log_ref": cls._build_log_ref(sample_uuid, stage, turn_idx, retry),
        }
        if turn_idx is not None:
            entry["turn_index"] = turn_idx
        if retry is not None:
            entry["retry"] = retry
        if constraint_ids:
            entry["constraint_ids"] = constraint_ids
        if function_index is not None:
            entry["function_index"] = function_index
        if extra_fields:
            for key, value in extra_fields.items():
                if value is not None:
                    entry[key] = value
        return entry

    @staticmethod
    def _log_failure(
        sample_uuid: str,
        stage: str,
        error_type: str,
        message: str,
        turn_idx: Optional[int] = None,
        retry: Optional[int] = None,
        constraint_id: Optional[str] = None,
        function_index: Optional[int] = None,
    ) -> None:
        fields = [f"sample_uuid={sample_uuid}", f"stage={stage}", f"error_type={error_type}"]
        if turn_idx is not None:
            fields.append(f"turn={turn_idx + 1}")
        if retry is not None:
            fields.append(f"retry={retry}")
        if constraint_id is not None:
            fields.append(f"constraint_id={constraint_id}")
        if function_index is not None:
            fields.append(f"function_index={function_index}")
        logger.warning("[%s] %s", " ".join(fields), message)

    # ------------------------------------------------------------------
    # Verification helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _check_error(response: str, turn: int) -> None:
        """Raise TaskFailed if the response is entirely an error JSON object.

        Only matches when the whole response is a JSON object (or a single
        markdown code block wrapping one) — avoids false positives on
        responses that mention errors as part of substantive content.
        """
        text = response.strip()
        data = None

        # Try parsing entire response as JSON
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                data = parsed
        except (json.JSONDecodeError, ValueError):
            pass

        # Try a standalone code block (response is *only* the block)
        if data is None:
            m = re.fullmatch(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
            if m:
                try:
                    data = json.loads(m.group(1))
                except (json.JSONDecodeError, ValueError):
                    pass
        if data and "error" in data:
            err = data["error"]
            if err == "contradicting_constraints":
                raise TaskFailed(
                    message=f"Expected error 'contradicting_constraints' found in response",
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
                message=f"Language detection error: {e}",
                error_type=format_error_type_with_turn("language_detection", turn),
            )
        valid = {code3, code2} if code2 is not None else {code3}
        if target not in valid:
            raise TaskFailed(
                message=f"Response not in expected language {target}, got {valid}",
                error_type=format_error_type_with_turn("invalid_language", turn),
            )

    @staticmethod
    def _run_eval_funcs(
        response: str,
        constraint_ids: List[str],
        eval_funcs_by_id: Dict[str, List[str]],
        resolved_values: Dict[str, Dict[str, Any]],
        turn: int,
        executor: FunctionExecutor,
        sample_uuid: str,
        retry: int,
    ) -> List[Dict[str, Any]]:
        """Run evaluation functions for each constraint; raise on failure."""
        failed = []
        execution_errors: List[Dict[str, Any]] = []

        for cid in constraint_ids:
            funcs = eval_funcs_by_id.get(cid, [])
            if not funcs:
                raise TaskFailed(
                    message=f"No evaluation functions for constraint {cid}",
                    error_type=format_error_type_with_turn("no_eval_functions_for_constraint", turn),
                )
            kwargs = resolved_values.get(cid, {})
            results = []
            for func_idx, func in enumerate(funcs):
                try:
                    r = executor.execute_with_response(func, response, **kwargs)
                    if r is not None:
                        results.append(r)
                except Exception as e:
                    error_type = getattr(e, "error_type", "function_execution_error")
                    GenerateResponsesTask._log_failure(
                        sample_uuid=sample_uuid,
                        stage="verification_executor",
                        error_type=error_type,
                        message=str(e),
                        turn_idx=turn,
                        retry=retry,
                        constraint_id=cid,
                        function_index=func_idx,
                    )
                    execution_errors.append(
                        GenerateResponsesTask._build_error_entry(
                            sample_uuid=sample_uuid,
                            error_type=error_type,
                            stage="verification_executor",
                            turn_idx=turn,
                            retry=retry,
                            constraint_ids=[cid],
                            function_index=func_idx,
                        )
                    )
                    results.append(0)
            acc = float(np.mean(results)) if results else 0.0
            if acc <= 0:
                failed.append((cid, acc))

        if failed:
            if len(failed) == 1:
                cid, acc = failed[0]
                raise TaskFailed(
                    message=f"Verification failed for constraint {cid} (acc={acc})",
                    error_type=format_error_type_with_turn("constraint_verification_failed", turn),
                )
            ids_str = ", ".join(str(c) for c, _ in failed)
            raise TaskFailed(
                message=f"Verification failed for constraints {ids_str}",
                error_type=format_error_type_with_turn("multiple_constraints_verification_failed", turn),
            )
        return execution_errors

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
                            message=f"Score {score} at turn {turn_idx + 1} is below threshold {threshold}.",
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
        sample_uuid = self._generate_sample_uuid()
        num_turns = len(turns)
        errors: List[Dict[str, Any]] = []
        attempted_turns: List[Dict[str, Any]] = []
        successful_turns: List[Dict[str, Any]] = []
        successful_messages: List[Dict[str, str]] = []
        first_failed_turn_index: Optional[int] = None

        def register_error(
            error_type: str,
            stage: str,
            turn_idx: Optional[int] = None,
            retry: Optional[int] = None,
            constraint_ids: Optional[List[str]] = None,
            function_index: Optional[int] = None,
            extra_fields: Optional[Dict[str, Any]] = None,
        ) -> Dict[str, Any]:
            entry = self._build_error_entry(
                sample_uuid=sample_uuid,
                error_type=error_type,
                stage=stage,
                turn_idx=turn_idx,
                retry=retry,
                constraint_ids=constraint_ids,
                function_index=function_index,
                extra_fields=extra_fields,
            )
            errors.append(entry)
            return entry

        # Build eval_funcs lookup: constraint_id -> list of func strings
        eval_funcs_by_id: Dict[str, List[str]] = {}
        for cid, cinfo in constraints.items():
            eval_funcs_by_id[cid] = cinfo.get("eval_funcs", [])

        all_eval_funcs = [
            func
            for funcs in eval_funcs_by_id.values()
            for func in funcs
        ]
        self._ensure_eval_runtime(all_eval_funcs)

        # ---- Step 2: Resolve all placeholders (once, before turn loop) ----
        placeholder_lookup_file = os.environ.get("PLACEHOLDER_LOOKUP_FILE", "")
        placeholder_lookup: Dict[str, Any] = {}
        if placeholder_lookup_file:
            try:
                with open(placeholder_lookup_file, "r") as f:
                    placeholder_lookup = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError) as e:
                self._log_failure(
                    sample_uuid=sample_uuid,
                    stage="placeholder_lookup",
                    error_type="placeholder_lookup_failed",
                    message=f"Could not load placeholder lookup from {placeholder_lookup_file}: {e}",
                )
                register_error("placeholder_lookup_failed", "placeholder_lookup")

        static_resolver = StaticPlaceholderResolver(placeholder_lookup)
        llm_resolver = LLMPlaceholderResolver()
        unresolved_constraints: Set[str] = set()
        placeholder_failure_details: Dict[str, Dict[str, Any]] = {}

        def record_placeholder_failure(
            cid: str,
            stage: str,
            message: str,
            placeholder_names: Optional[List[str]] = None,
            failure_reason: Optional[str] = None,
        ) -> Dict[str, Any]:
            extra_fields: Dict[str, Any] = {}
            if placeholder_names:
                extra_fields["placeholder_names"] = placeholder_names
            if failure_reason:
                extra_fields["failure_reason"] = failure_reason

            self._log_failure(
                sample_uuid=sample_uuid,
                stage=stage,
                error_type="placeholder_resolution_failed",
                message=message,
                constraint_id=cid,
            )
            error_entry = register_error(
                "placeholder_resolution_failed",
                stage,
                constraint_ids=[cid],
                extra_fields=extra_fields,
            )
            placeholder_failure_details[cid] = {
                "constraint_id": cid,
                "error_type": error_entry["error_type"],
                "stage": error_entry["stage"],
                "log_ref": error_entry["log_ref"],
            }
            if placeholder_names:
                placeholder_failure_details[cid]["placeholder_names"] = placeholder_names
            if failure_reason:
                placeholder_failure_details[cid]["failure_reason"] = failure_reason
            return error_entry

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
        # (includes static placeholders that couldn't be resolved from the lookup)
        pending_requests: List[Request] = []
        for cid, cinfo in constraints.items():
            placeholders = cinfo.get("placeholders", {})
            if not placeholders:
                continue
            already_resolved = set(resolved_values.get(cid, {}).keys())
            req = llm_resolver.build_request(cid, cinfo["constraint"], placeholders, query, self.GEN_PARAMS, resolved_names=already_resolved)
            if req is not None:
                pending_requests.append(req)

        # Phase C: yield batch and collect responses
        if pending_requests:
            responses: Union[Response, List[Response]] = yield pending_requests
            if isinstance(responses, Response):
                responses = [responses]
            expected_cids = {str(req.context["constraint_id"]) for req in pending_requests}
            seen_cids = set()

            # Phase D: parse responses into resolved_values
            for resp in responses:
                context = getattr(getattr(resp, "request", None), "context", None)
                if not isinstance(context, dict) or "constraint_id" not in context:
                    continue
                cid = str(context["constraint_id"])
                if cid not in expected_cids:
                    continue
                seen_cids.add(cid)
                raw_placeholder_response = resp.get_text()
                if raw_placeholder_response is None:
                    unresolved_constraints.add(cid)
                    record_placeholder_failure(
                        cid,
                        stage="placeholder_resolution",
                        message=(
                            f"Missing placeholder text for constraint {cid}. "
                            "The response did not contain model output text."
                        ),
                        placeholder_names=sorted(constraints[cid].get("placeholders", {}).keys()),
                        failure_reason="missing_response",
                    )
                    continue

                try:
                    vals = llm_resolver.parse_response(raw_placeholder_response)
                except Exception as e:
                    unresolved_constraints.add(cid)
                    record_placeholder_failure(
                        cid,
                        stage="placeholder_resolution",
                        message=(
                            f"Failed to parse placeholder response for constraint {cid}: "
                            f"{e}. Response: {raw_placeholder_response}"
                        ),
                        placeholder_names=sorted(constraints[cid].get("placeholders", {}).keys()),
                        failure_reason="parse_response",
                    )
                    continue

                resolved_values.setdefault(cid, {}).update(vals)

            for req in pending_requests:
                cid = str(req.context["constraint_id"])
                if cid in seen_cids:
                    continue
                unresolved_constraints.add(cid)
                record_placeholder_failure(
                    cid,
                    stage="placeholder_resolution",
                    message=(
                        f"Missing placeholder response for constraint {cid}. "
                        "The batch response did not include a matching request context."
                    ),
                    placeholder_names=sorted(constraints[cid].get("placeholders", {}).keys()),
                    failure_reason="missing_response",
                )

        # ---- Step 3: Build resolved constraint texts ----
        resolved_constraints: Dict[str, str] = {}
        for cid, cinfo in constraints.items():
            text = cinfo["constraint"]
            for pname, pval in resolved_values.get(cid, {}).items():
                text = text.replace(f"{{{pname}}}", format_value(pval))
            if re.search(r"\{[^{}]+\}", text):
                unresolved_constraints.add(cid)
                unresolved_names = sorted(set(re.findall(r"\{([^{}]+)\}", text)))
                if cid not in placeholder_failure_details:
                    record_placeholder_failure(
                        cid,
                        stage="placeholder_resolution",
                        message=(
                            f"Constraint {cid} still has unresolved placeholders after resolution: "
                            f"{', '.join(unresolved_names)}"
                        ),
                        placeholder_names=unresolved_names,
                        failure_reason="unresolved_after_resolution",
                    )
            resolved_constraints[cid] = text

        # ---- Step 4: Per-turn generation loop ----
        MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "5"))
        language = get_env_language_name()
        is_no_followup = True  # always true – single query, multiple rephrase turns

        # Format query once (before turn loop)
        with open("model_prompts/format_query_prompt.txt", "r") as f:
            fmt_prompt = f.read().strip()
        fmt_prompt = fmt_prompt.format(query=query, language=language)
        fmt_resp: Response = yield Request({"messages": [{"role": "user", "content": fmt_prompt}], **self.GEN_PARAMS})
        formatted_query = fmt_resp.get_text().strip()

        executor = FunctionExecutor()
        conversation_messages: List[Dict[str, str]] = []
        conversation_responses: List[str] = []

        for turn_idx in range(num_turns):
            turn_cids = turns[turn_idx]["constraint_ids"]
            turn_constraint_texts = [resolved_constraints[cid] for cid in turn_cids]
            turn_record: Dict[str, Any] = {
                "turn_index": turn_idx,
                "constraint_ids": turn_cids,
                "status": "pending",
                "usable_for_final_dataset": False,
                "attempts": [],
            }
            attempted_turns.append(turn_record)

            if any(cid in unresolved_constraints for cid in turn_cids):
                failed_cids = [cid for cid in turn_cids if cid in unresolved_constraints]
                related_errors = [
                    placeholder_failure_details[cid]
                    for cid in failed_cids
                    if cid in placeholder_failure_details
                ]
                turn_record["status"] = "failed"
                error_entry = register_error(
                    "placeholder_resolution_failed",
                    "turn_setup",
                    turn_idx=turn_idx,
                    constraint_ids=turn_cids,
                    extra_fields={
                        "failed_constraint_ids": failed_cids,
                        "related_errors": related_errors or None,
                    },
                )
                turn_record["error_types"] = [error_entry["error_type"]]
                turn_record["last_error"] = error_entry
                self._log_failure(
                    sample_uuid=sample_uuid,
                    stage="turn_setup",
                    error_type="placeholder_resolution_failed",
                    message=(
                        f"Skipping turn {turn_idx + 1} due to unresolved placeholders in "
                        f"constraints {', '.join(failed_cids)}"
                    ),
                    turn_idx=turn_idx,
                )
                if first_failed_turn_index is None:
                    first_failed_turn_index = turn_idx
                continue

            saved = {
                "conversation_messages": len(conversation_messages),
                "conversation_responses": len(conversation_responses),
            }

            turn_succeeded = False
            last_error: Optional[Dict[str, Any]] = None

            for retry in range(MAX_RETRIES):
                response_text: Optional[str] = None
                scoring_text: Optional[str] = None
                current_prompt: Optional[str] = None
                score: Optional[int] = None
                current_stage = "turn_setup"
                verification_errors: List[Dict[str, Any]] = []
                attempt_record: Dict[str, Any] = {
                    "retry_index": retry,
                    "status": "pending",
                }
                try:
                    if retry > 0:
                        conversation_messages[:] = conversation_messages[:saved["conversation_messages"]]
                        conversation_responses[:] = conversation_responses[:saved["conversation_responses"]]

                    current_stage = "prompt_build"
                    constraints_bullet_list = "\n".join(f"- {t}" for t in turn_constraint_texts)

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

                    attempt_record["prompt"] = current_prompt

                    conversation_messages.append({"role": "user", "content": current_prompt})

                    current_stage = "generation"
                    gen_resp: Response = yield Request({"messages": conversation_messages, **self.GEN_PARAMS})
                    response_text = gen_resp.get_text()
                    attempt_record["response"] = response_text

                    current_stage = "response_validation"
                    self._check_error(response_text, turn_idx)
                    current_stage = "language_validation"
                    self._check_language(response_text, turn_idx)
                    current_stage = "verification"
                    verification_errors = self._run_eval_funcs(
                        response_text,
                        turn_cids,
                        eval_funcs_by_id,
                        resolved_values,
                        turn_idx,
                        executor,
                        sample_uuid,
                        retry,
                    )
                    if verification_errors:
                        errors.extend(verification_errors)
                        attempt_record["verification_errors"] = verification_errors

                    current_stage = "scoring_request"
                    is_rephrase = is_no_followup and turn_idx > 0
                    scoring_messages = self._build_scoring_messages(
                        turn_idx, is_rephrase, turn_constraint_texts,
                        query, response_text, conversation_responses[0] if conversation_responses else "",
                    )
                    scored_resp: Response = yield Request({"messages": scoring_messages, **self.GEN_PARAMS})
                    scoring_text = scored_resp.get_text()
                    attempt_record["scoring_response"] = scoring_text
                    current_stage = "score_validation"
                    score = self._extract_and_check_score(scoring_text, turn_idx)
                    attempt_record["score"] = score

                    conversation_messages.append({"role": "assistant", "content": response_text})
                    conversation_responses.append(response_text)
                    attempt_record["status"] = "success"
                    turn_record["attempts"].append(attempt_record)

                    successful_turn = {
                        "constraint_ids": turn_cids,
                        "response": response_text,
                        "prompt": current_prompt,
                        "score": score,
                        "scoring_response": scoring_text,
                    }
                    turn_record.update(successful_turn)

                    formatted_constraints = format_constraints_with_conjunctions(turn_constraint_texts)
                    if formatted_query and turn_idx == 0:
                        user_content = f"{formatted_query} {formatted_constraints}"
                    else:
                        user_content = formatted_constraints


                    if first_failed_turn_index is None:
                        successful_turns.append(successful_turn)
                        successful_messages.extend([
                            {"role": "user", "content": user_content},
                            {"role": "assistant", "content": response_text},
                        ])
                        turn_record["usable_for_final_dataset"] = True
                    turn_record["status"] = "success"
                    turn_succeeded = True
                    break

                except TaskFailed as tf:
                    self._log_failure(
                        sample_uuid=sample_uuid,
                        stage=current_stage,
                        error_type=tf.error_type,
                        message=tf.message,
                        turn_idx=turn_idx,
                        retry=retry,
                    )
                    error_entry = register_error(
                        tf.error_type,
                        current_stage,
                        turn_idx=turn_idx,
                        retry=retry,
                        constraint_ids=turn_cids,
                    )
                    attempt_record["status"] = "failed"
                    attempt_record["error"] = error_entry
                    if verification_errors:
                        attempt_record["verification_errors"] = verification_errors
                    if current_prompt is not None:
                        attempt_record["prompt"] = current_prompt
                    if response_text is not None:
                        attempt_record["response"] = response_text
                    if scoring_text is not None:
                        attempt_record["scoring_response"] = scoring_text
                    if score is not None:
                        attempt_record["score"] = score
                    turn_record["attempts"].append(attempt_record)
                    last_error = error_entry

                    if retry == MAX_RETRIES - 1:
                        if response_text is not None:
                            conversation_messages.append({"role": "assistant", "content": response_text})
                            conversation_responses.append(response_text)
                        else:
                            conversation_messages[:] = conversation_messages[:saved["conversation_messages"]]
                            conversation_responses[:] = conversation_responses[:saved["conversation_responses"]]
                    continue

                except Exception as e:
                    error_type = format_error_type_with_turn("unexpected_error", turn_idx)
                    self._log_failure(
                        sample_uuid=sample_uuid,
                        stage=current_stage,
                        error_type=error_type,
                        message=str(e),
                        turn_idx=turn_idx,
                        retry=retry,
                    )
                    error_entry = register_error(
                        error_type,
                        current_stage,
                        turn_idx=turn_idx,
                        retry=retry,
                        constraint_ids=turn_cids,
                    )
                    attempt_record["status"] = "failed"
                    attempt_record["error"] = error_entry
                    if verification_errors:
                        attempt_record["verification_errors"] = verification_errors
                    if current_prompt is not None:
                        attempt_record["prompt"] = current_prompt
                    if response_text is not None:
                        attempt_record["response"] = response_text
                    if scoring_text is not None:
                        attempt_record["scoring_response"] = scoring_text
                    if score is not None:
                        attempt_record["score"] = score
                    turn_record["attempts"].append(attempt_record)
                    last_error = error_entry

                    if retry == MAX_RETRIES - 1:
                        if response_text is not None:
                            conversation_messages.append({"role": "assistant", "content": response_text})
                            conversation_responses.append(response_text)
                        else:
                            conversation_messages[:] = conversation_messages[:saved["conversation_messages"]]
                            conversation_responses[:] = conversation_responses[:saved["conversation_responses"]]
                    continue

            if not turn_succeeded:
                turn_record["status"] = "failed"
                turn_record["error_types"] = sorted({
                    attempt["error"]["error_type"]
                    for attempt in turn_record["attempts"]
                    if "error" in attempt
                })
                if last_error is not None:
                    turn_record["last_error"] = last_error
                if first_failed_turn_index is None:
                    first_failed_turn_index = turn_idx

        # ---- Build output constraints dict (mirrors input structure, with resolved text) ----
        used_cids = set()
        for turn in attempted_turns:
            used_cids.update(turn["constraint_ids"])

        output_constraints: Dict[str, Dict[str, Any]] = {}
        for cid in used_cids:
            cinfo = constraints[cid]
            output_constraints[cid] = {
                "constraint": resolved_constraints[cid],
                "category": cinfo.get("category", "default"),
                "eval_funcs": cinfo.get("eval_funcs", []),
            }

        if len(successful_turns) == num_turns:
            status = "success"
        elif successful_turns:
            status = "partial_success"
        else:
            status = "failed"

        return {
            "uuid": sample_uuid,
            "query": query,
            "query_metadata": query_metadata,
            "constraints": output_constraints,
            "turns": successful_turns,
            "messages": successful_messages,
            "attempted_turns": attempted_turns,
            "errors": errors,
            "successful_turn_count": len(successful_turns),
            "first_failed_turn_index": first_failed_turn_index,
            "has_errors": bool(errors),
            "status": status,
        }
