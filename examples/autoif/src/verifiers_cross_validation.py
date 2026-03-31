#!/usr/bin/env python3
"""Standalone cross-validation for verifier functions.

Thin wrapper around utils.cross_validator.run_cross_validation().
Reads a JSONL file produced by the generation task (with SKIP_INLINE_CROSSVAL=true),
runs cross-validation on each item, and writes a single output JSONL where each
line carries the original fields plus the full CrossValidationResult.  Downstream
steps can filter on the ``success`` field.

Features:
  - Per-instruction log files in ``cv_logs/`` next to the output file.
  - Checkpoint support: results are written incrementally so that a restarted
    job resumes from where it left off.
  - Final output is sorted by ``instruction_id``.
"""

import argparse
import json
import logging
import os
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.cross_validator import run_cross_validation
from utils.function_executor import _append_to_log
from utils.model_preloader import scan_and_preload_models

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def _load_checkpoint(output_file: str) -> set:
    """Build a set of already-processed instruction_ids.

    Reads both the output JSONL (source of truth) and the companion
    ``.checkpoint`` file.  Unparseable lines are silently skipped so that
    partial writes from a killed process don't block resumption.
    """
    done_ids: set = set()
    checkpoint_file = output_file + ".checkpoint"

    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    iid = obj.get("instruction_id")
                    if iid is not None:
                        done_ids.add(str(iid))
                except json.JSONDecodeError:
                    pass

    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    done_ids.add(line)

    return done_ids


def _sort_output_file(output_file: str) -> int:
    """Re-read the output JSONL, sort lines by instruction_id, and rewrite.

    Returns the number of lines written.
    """
    if not os.path.exists(output_file):
        return 0

    lines = []
    with open(output_file, "r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                lines.append(json.loads(raw))
            except json.JSONDecodeError:
                pass

    def _sort_key(obj):
        iid = obj.get("instruction_id", "0")
        try:
            return int(iid)
        except (ValueError, TypeError):
            return iid

    lines.sort(key=_sort_key)

    with open(output_file, "w", encoding="utf-8") as f:
        for obj in lines:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    return len(lines)


def _process_item(item, min_functions, min_test_cases, function_pass_rate,
                  min_case_passes, log_dir):
    """Run cross-validation for a single item (thread-safe)."""
    iid = item.get("instruction_id", "unknown")
    functions = item.get("eval_funcs") or item.get("eval_func") or []
    cases = item.get("cases", [])

    log_file = os.path.join(log_dir, f"{iid}.log") if log_dir else None

    if log_file:
        _append_to_log(log_file,
            f"=== Instruction {iid} | {len(functions)} functions x {len(cases)} cases ===\n")

    cv = run_cross_validation(
        functions, cases,
        min_functions=min_functions,
        min_test_cases=min_test_cases,
        function_pass_rate=function_pass_rate,
        min_case_passes=min_case_passes,
        log_file=log_file,
    )

    status = "PASS" if cv.success else "FAIL"
    detail_suffix = ""
    if not cv.success:
        parts = []
        if cv.failure_reason:
            parts.append(f"reason={cv.failure_reason}")
        if cv.error_counts:
            top_errors = sorted(cv.error_counts.items(), key=lambda x: -x[1])[:3]
            parts.append("errors={%s}" % ", ".join(f"{k}:{v}" for k, v in top_errors))
        if cv.first_error_details:
            parts.append(f"first_error=\"{cv.first_error_details[:200]}\"")
        if parts:
            detail_suffix = " " + " ".join(parts)

    logger.info("IID:%s funcs=%d/%d cases=%d/%d acc=%.0f%% %s%s",
                iid,
                len(cv.passing_functions), cv.total_functions,
                len(cv.passing_cases), cv.total_cases,
                cv.best_accuracy * 100,
                status, detail_suffix)

    if log_file:
        _append_to_log(log_file,
            f"\n=== Summary: {status} | "
            f"funcs={len(cv.passing_functions)}/{cv.total_functions} "
            f"cases={len(cv.passing_cases)}/{cv.total_cases} "
            f"acc={cv.best_accuracy:.0%}"
            f"{(' | ' + cv.failure_reason) if cv.failure_reason else ''}"
            f" ===\n")

    return {**item, **cv.to_dict()}, cv.success


def main():
    p = argparse.ArgumentParser(description="Cross-validate verifier functions")
    p.add_argument("--input-file", required=True,
                   help="Input JSONL (items with eval_funcs + cases)")
    p.add_argument("--output-file", required=True,
                   help="Output JSONL (each item merged with CrossValidationResult)")
    p.add_argument("--workers", type=int, default=None,
                   help="Number of threads (default: CROSSVAL_WORKERS env or CPU count)")
    args = p.parse_args()

    min_functions = int(os.getenv("MIN_FUNCTIONS", 1))
    min_test_cases = int(os.getenv("MIN_TEST_CASES", 1))
    function_pass_rate = float(os.getenv("FUNCTION_PASS_RATE", 0.8))
    min_case_passes = int(os.getenv("MIN_CASE_PASSES", 1))
    workers = args.workers or int(os.getenv("CROSSVAL_WORKERS", 32))

    from utils.function_executor import FUNCTION_TIMEOUT, INIT_TIMEOUT, CASE_TIMEOUT
    logger.info("Parameters: min_functions=%d min_test_cases=%d "
                "function_pass_rate=%.2f min_case_passes=%d workers=%d "
                "function_timeout=%d init_timeout=%d case_timeout=%d",
                min_functions, min_test_cases, function_pass_rate, min_case_passes, workers,
                FUNCTION_TIMEOUT, INIT_TIMEOUT, CASE_TIMEOUT)

    # ---- Log directory ----
    output_dir = os.path.dirname(args.output_file) or "."
    log_dir = os.path.join(output_dir, "cv_logs")
    os.makedirs(log_dir, exist_ok=True)
    logger.info("Per-instruction logs: %s", log_dir)

    # ---- Load input ----
    with open(args.input_file) as f:
        items = [json.loads(line) for line in f if line.strip()]
    logger.info("Loaded %d items from %s", len(items), args.input_file)

    valid_items = [(i, item) for i, item in enumerate(items) if "__ERROR__" not in item]
    skipped_errors = len(items) - len(valid_items)

    # ---- Checkpoint: skip already-processed items ----
    done_ids = _load_checkpoint(args.output_file)
    if done_ids:
        before = len(valid_items)
        valid_items = [
            (i, item) for i, item in valid_items
            if str(item.get("instruction_id", "")) not in done_ids
        ]
        resumed = before - len(valid_items)
        logger.info("Checkpoint: %d items already processed, %d remaining",
                     resumed, len(valid_items))

    total = len(valid_items)
    if total == 0:
        logger.info("Nothing to process — all items already checkpointed")
        if not os.path.exists(args.output_file):
            open(args.output_file, "a").close()
        _sort_output_file(args.output_file)
        return

    # ---- NLP model pre-download ----
    all_functions = []
    for _, item in valid_items:
        funcs = item.get("eval_funcs") or item.get("eval_func") or []
        if isinstance(funcs, str):
            funcs = [funcs]
        all_functions.extend(funcs)
    loaded_models = scan_and_preload_models(all_functions, logger)

    # ---- Worker pool (pre-fork after model preloading) ----
    worker_pool = None
    preimport = []
    if loaded_models.get("trankit"):
        preimport.extend(["torch", "trankit", "transformers"])
    if loaded_models.get("spacy"):
        preimport.append("spacy")
    if loaded_models.get("nltk"):
        preimport.append("nltk")

    if preimport:
        from utils.worker_pool import WorkerPool
        from utils.function_executor import set_worker_pool
        from utils.model_preloader import _TRANKIT_CACHE_DIR

        # Build pipeline preload config for trankit languages
        pipeline_preload = None
        trankit_langs = loaded_models.get("trankit", [])
        if trankit_langs and _TRANKIT_CACHE_DIR:
            pipeline_preload = {
                lang: {"gpu": False, "cache_dir": _TRANKIT_CACHE_DIR}
                for lang in trankit_langs
            }
            logger.info("Trankit pipelines will be pre-loaded in workers: %s",
                         trankit_langs)

        worker_pool = WorkerPool.create(
            num_workers=workers,
            preimport_modules=preimport,
            preload_pipelines=pipeline_preload,
        )
        set_worker_pool(worker_pool)
        logger.info("Worker pool active (%d workers, pre-imported: %s)",
                     worker_pool.num_workers, ", ".join(preimport))

    # ---- Process items with incremental checkpointing ----
    passed = 0
    processed = 0
    aggregate_errors: Counter = Counter()
    aggregate_failure_reasons: Counter = Counter()

    write_lock = threading.Lock()
    checkpoint_file = args.output_file + ".checkpoint"

    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_item = {
            pool.submit(_process_item, item,
                        min_functions, min_test_cases,
                        function_pass_rate, min_case_passes,
                        log_dir): item
            for _orig_idx, item in valid_items
        }
        for future in as_completed(future_to_item):
            out, success = future.result()
            iid = str(out.get("instruction_id", "unknown"))

            with write_lock:
                with open(args.output_file, "a", encoding="utf-8") as fout:
                    fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                    fout.flush()
                with open(checkpoint_file, "a", encoding="utf-8") as fcp:
                    fcp.write(iid + "\n")
                    fcp.flush()

            processed += 1
            if success:
                passed += 1
            else:
                if out.get("failure_reason"):
                    aggregate_failure_reasons[out["failure_reason"]] += 1
                if out.get("error_counts"):
                    for err_type, cnt in out["error_counts"].items():
                        aggregate_errors[err_type] += cnt

    # ---- Sort final output by instruction_id ----
    n_sorted = _sort_output_file(args.output_file)
    logger.info("Output sorted by instruction_id (%d lines)", n_sorted)

    total_with_resumed = n_sorted
    logger.info("Done: %d processed this run, %d passed, %d failed, "
                "%d error records skipped, %d total in output",
                processed, passed, processed - passed, skipped_errors,
                total_with_resumed)

    if aggregate_failure_reasons:
        logger.info("Failure reasons across all failed items:")
        for reason, cnt in aggregate_failure_reasons.most_common():
            logger.info("  %-55s %d", reason, cnt)

    if aggregate_errors:
        logger.info("Execution error breakdown across all failed items:")
        for err_type, cnt in aggregate_errors.most_common():
            logger.info("  %-55s %d", err_type, cnt)

    if worker_pool is not None:
        worker_pool.shutdown()


if __name__ == "__main__":
    main()
