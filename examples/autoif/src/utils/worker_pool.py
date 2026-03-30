"""Pre-forked worker pool for function evaluation.

Workers inherit heavy imports (torch, trankit, transformers, spacy, nltk)
from the parent process via fork copy-on-write.  Subsequent ``import``
statements inside exec'd function code are instant (sys.modules cache hit),
eliminating the ~156 s cold-import overhead that dominates CPU execution.

The pool is safe for concurrent callers (e.g. ThreadPoolExecutor threads):
each ``run_function`` call gets a unique request ID and waits on a
per-request ``threading.Event`` so that results are routed back to the
correct caller regardless of completion order.

Usage::

    pool = WorkerPool.create(num_workers=8, preimport_modules=["trankit", ...])
    result = pool.run_function(func_str, test_inputs, init_timeout=300, case_timeout=30)
    pool.shutdown()
"""

import logging
import multiprocessing
import os
import signal
import sys
import threading
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_SHUTDOWN = None


_PRELOADED_PIPELINES: Dict[str, Any] = {}

_MAX_CONSECUTIVE_TIMEOUTS = int(os.environ.get("MAX_CONSECUTIVE_TIMEOUTS", 3))


_LANG_ALIASES: Dict[str, str] = {
    "fi": "finnish", "en": "english", "de": "german", "fr": "french",
    "sv": "swedish", "et": "estonian", "no": "norwegian", "da": "danish",
}


def _monkey_patch_trankit() -> None:
    """Replace ``trankit.Pipeline`` with a factory that returns cached instances.

    When a pre-loaded Pipeline exists for the requested language, the cached
    object is returned immediately instead of re-loading from disk.  This is
    robust against any call style the LLM might generate (positional args,
    keyword args, extra kwargs like ``gpu=False``, etc.).

    Language aliases (e.g. 'fi' → 'finnish') are resolved so that
    ``trankit.Pipeline('fi')`` returns the cached 'finnish' pipeline.
    """
    try:
        import trankit
    except ImportError:
        return

    _OrigPipeline = trankit.Pipeline

    def _cached_pipeline(lang="english", **kwargs):
        key = _LANG_ALIASES.get(lang.lower(), lang.lower())
        if key in _PRELOADED_PIPELINES:
            return _PRELOADED_PIPELINES[key]
        return _OrigPipeline(lang, **kwargs)

    trankit.Pipeline = _cached_pipeline


def _worker_loop(task_queue: multiprocessing.Queue,
                 result_queue: multiprocessing.Queue,
                 worker_id: int,
                 preload_pipelines: Optional[Dict[str, dict]] = None) -> None:
    """Main loop executed by each forked worker process."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    # Limit per-worker CPU threads to avoid oversubscription when all workers
    # run CPU-bound inference (PyTorch, trankit) simultaneously.
    _threads_per_worker = os.environ.get("THREADS_PER_WORKER", "2")
    os.environ.setdefault("OMP_NUM_THREADS", _threads_per_worker)
    os.environ.setdefault("MKL_NUM_THREADS", _threads_per_worker)
    os.environ.setdefault("OPENBLAS_NUM_THREADS", _threads_per_worker)
    try:
        import torch
        torch.set_num_threads(int(_threads_per_worker))
    except Exception:
        pass

    def _timeout_handler(signum, frame):
        raise TimeoutError("timeout")

    signal.signal(signal.SIGALRM, _timeout_handler)

    # Pre-load trankit Pipelines once per worker (shared across all tasks).
    if preload_pipelines:
        try:
            import trankit
            for lang, kwargs in preload_pipelines.items():
                logger.info("Worker %d: loading trankit Pipeline('%s')", worker_id, lang)
                _PRELOADED_PIPELINES[lang] = trankit.Pipeline(lang=lang, **kwargs)
                logger.info("Worker %d: Pipeline('%s') ready", worker_id, lang)
        except Exception as e:
            logger.warning("Worker %d: pipeline preload failed: %s", worker_id, e)

    if preload_pipelines:
        _monkey_patch_trankit()

    while True:
        try:
            task = task_queue.get()
        except Exception:
            break
        if task is _SHUTDOWN:
            break

        request_id, func_str, mode, init_timeout, case_timeout, extras = task
        result: Dict[str, Any] = {}

        old_stdout = sys.stdout
        sys.stdout = sys.stderr

        try:
            signal.alarm(init_timeout)
            namespace = {"_PRELOADED_PIPELINES": _PRELOADED_PIPELINES}
            exec(func_str, namespace)
            evaluate_func = namespace.get("evaluate")

            if evaluate_func is None:
                result = {"error_type": "no_evaluate_function",
                          "error": "No evaluate function found"}
            elif mode == "test_batch":
                test_inputs = extras.get("test_inputs", [])
                case_results = []
                consecutive_timeouts = 0
                for ti in test_inputs:
                    if consecutive_timeouts >= _MAX_CONSECUTIVE_TIMEOUTS:
                        case_results.append({
                            "error_type": "timeout",
                            "error": f"Skipped after {consecutive_timeouts} consecutive timeouts"})
                        continue
                    signal.alarm(case_timeout)
                    try:
                        if isinstance(ti, dict):
                            res = evaluate_func(**ti)
                        else:
                            res = evaluate_func(ti)
                        case_results.append({"result": res})
                        consecutive_timeouts = 0
                    except TimeoutError:
                        case_results.append({"error_type": "timeout",
                                             "error": f"Function timed out after {case_timeout}s"})
                        consecutive_timeouts += 1
                    except Exception as e:
                        case_results.append({"error_type": "execution_error",
                                             "error": str(e)})
                        consecutive_timeouts = 0
                result = {"results": case_results}
            elif mode == "test":
                signal.alarm(case_timeout)
                ti = extras.get("test_input")
                if isinstance(ti, dict):
                    res = evaluate_func(**ti)
                else:
                    res = evaluate_func(ti)
                result = {"result": res}
            elif mode == "execute":
                signal.alarm(case_timeout)
                response = extras["response"]
                kwargs = extras.get("kwargs", {})
                res = evaluate_func(response, **kwargs)
                if res is None:
                    result = {"error_type": "returned_none",
                              "error": "Evaluation function returned None"}
                else:
                    result = {"result": int(res)}
        except TimeoutError:
            result = {"error_type": "timeout",
                      "error": f"Function timed out after {init_timeout}s (init phase)"}
        except Exception as e:
            result = {"error_type": "execution_error", "error": str(e)}
        finally:
            signal.alarm(0)
            sys.stdout = old_stdout

        try:
            result_queue.put((request_id, result))
        except Exception:
            pass


class WorkerPool:
    """Pool of pre-forked processes with heavy modules already imported.

    Thread-safe: multiple callers can invoke ``run_function`` concurrently.
    Results are routed back to the correct caller via per-request events.
    """

    def __init__(self, num_workers: int,
                 preload_pipelines: Optional[Dict[str, dict]] = None):
        self._num_workers = num_workers
        ctx = multiprocessing.get_context("fork")
        self._task_queue: multiprocessing.Queue = ctx.Queue()
        self._result_queue: multiprocessing.Queue = ctx.Queue()
        self._workers: List[multiprocessing.Process] = []
        self._alive = True

        self._pending_lock = threading.Lock()
        self._pending: Dict[int, tuple] = {}
        self._next_id = 0

        for i in range(num_workers):
            p = ctx.Process(
                target=_worker_loop,
                args=(self._task_queue, self._result_queue, i,
                      preload_pipelines),
                daemon=True,
            )
            p.start()
            self._workers.append(p)

        # Background thread that reads from _result_queue and wakes callers.
        self._dispatcher = threading.Thread(target=self._dispatch_results,
                                            daemon=True)
        self._dispatcher.start()

        logger.info("WorkerPool started: %d workers (pid %s)",
                     num_workers,
                     ", ".join(str(w.pid) for w in self._workers))

    def _dispatch_results(self) -> None:
        """Background thread: read (request_id, result) and wake the caller."""
        while self._alive:
            try:
                item = self._result_queue.get(timeout=1.0)
            except Exception:
                continue
            if item is None:
                break
            request_id, result = item
            with self._pending_lock:
                entry = self._pending.get(request_id)
            if entry is not None:
                event, holder = entry
                holder["result"] = result
                event.set()

    @staticmethod
    def create(num_workers: int,
               preimport_modules: Optional[List[str]] = None,
               preload_pipelines: Optional[Dict[str, dict]] = None,
               ) -> "WorkerPool":
        """Pre-import heavy modules in this process, then fork workers.

        Because ``import`` caches modules in ``sys.modules``, the forked
        workers inherit them via COW pages.  Any later ``import trankit``
        inside exec'd function code is a no-op (~0 s instead of ~156 s).

        If *preload_pipelines* is given (e.g. ``{"finnish": {"gpu": False,
        "cache_dir": "..."}}``), each worker instantiates those trankit
        Pipelines once at startup.  Function code is rewritten to use the
        pre-loaded instances instead of calling ``trankit.Pipeline()``.
        """
        for mod_name in (preimport_modules or []):
            try:
                __import__(mod_name)
                logger.info("Pre-imported %s for worker pool", mod_name)
            except Exception as e:
                logger.warning("Failed to pre-import %s: %s", mod_name, e)

        return WorkerPool(num_workers, preload_pipelines=preload_pipelines)

    def run_function(self, func_str: str, mode: str,
                     init_timeout: int = 300,
                     case_timeout: int = 30,
                     **extras) -> Dict[str, Any]:
        """Submit a function to a worker and wait for the result.

        Thread-safe.  Multiple callers block independently; results are
        matched to callers by request_id.
        """
        if not self._alive:
            return {"error_type": "execution_error",
                    "error": "WorkerPool is shut down"}

        with self._pending_lock:
            request_id = self._next_id
            self._next_id += 1
            event = threading.Event()
            holder: Dict[str, Any] = {}
            self._pending[request_id] = (event, holder)

        n_cases = len(extras.get("test_inputs", []) or [])
        if isinstance(extras.get("test_input"), dict):
            n_cases = 1
        wall = init_timeout + case_timeout * max(n_cases, 1) + 30

        self._task_queue.put(
            (request_id, func_str, mode, init_timeout, case_timeout, extras))

        event.wait(timeout=wall)

        with self._pending_lock:
            self._pending.pop(request_id, None)

        if "result" not in holder and "results" not in holder.get("result", {}):
            if not holder:
                return {"error_type": "timeout",
                        "error": f"Worker did not respond within {wall}s"}
        return holder.get("result", {"error_type": "timeout",
                                     "error": f"Worker did not respond within {wall}s"})

    @property
    def num_workers(self) -> int:
        return self._num_workers

    def shutdown(self) -> None:
        """Send shutdown sentinels and join all workers."""
        if not self._alive:
            return
        self._alive = False
        for _ in self._workers:
            try:
                self._task_queue.put(_SHUTDOWN)
            except Exception:
                pass
        self._result_queue.put(None)
        self._dispatcher.join(timeout=3)
        for w in self._workers:
            w.join(timeout=5)
            if w.is_alive():
                w.kill()
        logger.info("WorkerPool shut down")
