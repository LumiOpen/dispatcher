"""Worker pool for function evaluation using spawn context.

Workers are started via ``multiprocessing.get_context("spawn")`` so that
each one begins with a clean Python interpreter.  This avoids POSIX
fork-safety issues with libraries (stanza, PyTorch) that hold internal
C-level locks from background threads — those locks would deadlock
permanently in a forked child.

Each worker imports heavy modules (torch, stanza, spacy, nltk, …) once
at startup (~2-10 s one-time cost).  Subsequent ``import`` statements
inside exec'd function code are instant (``sys.modules`` cache hit within
the worker), preserving the same performance benefit that fork/COW provided.

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

_SHUTDOWN = "__SHUTDOWN_SENTINEL__"


_PRELOADED_PIPELINES: Dict[str, Any] = {}
_CACHED_SPACY_MODELS: Dict[str, Any] = {}
_CACHED_STANZA_PIPELINES: Dict[str, Any] = {}

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


def _monkey_patch_spacy() -> None:
    """Cache ``spacy.load()`` results so repeated calls with the same model
    name return the already-loaded object instead of re-reading from disk.

    Without this, every ``exec(func_str)`` that contains
    ``nlp = spacy.load("fi_core_news_sm")`` reloads the model (~1-2 s),
    which under shared-filesystem I/O contention can spike much higher.
    """
    try:
        import spacy
    except ImportError:
        return

    _original_load = spacy.load

    def _cached_load(name, **kwargs):
        if name in _CACHED_SPACY_MODELS:
            return _CACHED_SPACY_MODELS[name]
        model = _original_load(name, **kwargs)
        _CACHED_SPACY_MODELS[name] = model
        return model

    spacy.load = _cached_load


def _monkey_patch_stanza() -> None:
    """Cache ``stanza.Pipeline()`` results so repeated calls with the same
    language and processors return the already-loaded pipeline.

    Without this, every ``exec(func_str)`` that contains
    ``nlp = stanza.Pipeline('fi', processors='tokenize,lemma')`` reloads
    the ~348 MB PyTorch model from disk (~5-30 s on CPU, worse under I/O
    contention with multiple workers).
    """
    try:
        import stanza
    except ImportError:
        return

    _OrigPipeline = stanza.Pipeline

    def _cached_pipeline(lang="en", **kwargs):
        processors = kwargs.get("processors", "default")
        if isinstance(processors, dict):
            key = f"{lang}|" + ",".join(sorted(processors.keys()))
        else:
            key = f"{lang}|{processors}"
        if key in _CACHED_STANZA_PIPELINES:
            return _CACHED_STANZA_PIPELINES[key]
        pipeline = _OrigPipeline(lang, **kwargs)
        _CACHED_STANZA_PIPELINES[key] = pipeline
        return pipeline

    stanza.Pipeline = _cached_pipeline


def _worker_loop(task_queue: multiprocessing.Queue,
                 result_queue: multiprocessing.Queue,
                 worker_id: int,
                 preload_pipelines: Optional[Dict[str, dict]] = None,
                 preimport_modules: Optional[List[str]] = None) -> None:
    """Main loop executed by each spawned worker process."""
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

    # With spawn context, workers must import heavy modules themselves.
    # This is a one-time cost per worker (~2-10s depending on modules).
    for mod_name in (preimport_modules or []):
        try:
            __import__(mod_name)
            logger.info("Worker %d: imported %s", worker_id, mod_name)
        except Exception as e:
            logger.warning("Worker %d: failed to import %s: %s", worker_id, mod_name, e)

    # Disable stanza's download check inside workers (models already cached).
    if preimport_modules and "stanza" in preimport_modules:
        try:
            import stanza.pipeline.core as _core
            _original_init = _core.Pipeline.__init__

            def _patched_init(self, *args, **kwargs):
                kwargs.setdefault("download_method", None)
                kwargs.setdefault("verbose", False)
                return _original_init(self, *args, **kwargs)

            _core.Pipeline.__init__ = _patched_init
            logger.info("Worker %d: patched stanza download_method=None", worker_id)
        except Exception as e:
            logger.warning("Worker %d: failed to patch stanza: %s", worker_id, e)

    # Cache NLP model objects across function calls within this worker.
    # Without caching, every exec(func_str) re-loads models from disk
    # (stanza ~5-30s, spacy ~1-2s), which under I/O contention can exceed
    # init_timeout.  With caching, only the first function pays the cost.
    if preimport_modules:
        if "spacy" in preimport_modules:
            _monkey_patch_spacy()
            logger.info("Worker %d: patched spacy.load() for caching", worker_id)
        if "stanza" in preimport_modules:
            _monkey_patch_stanza()
            logger.info("Worker %d: patched stanza.Pipeline() for caching", worker_id)

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
        if task == _SHUTDOWN:
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
    """Pool of spawned processes that import heavy modules at startup.

    Thread-safe: multiple callers can invoke ``run_function`` concurrently.
    Results are routed back to the correct caller via per-request events.
    """

    def __init__(self, num_workers: int,
                 preload_pipelines: Optional[Dict[str, dict]] = None,
                 preimport_modules: Optional[List[str]] = None):
        self._num_workers = num_workers
        ctx = multiprocessing.get_context("spawn")
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
                      preload_pipelines, preimport_modules),
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
            if item == _SHUTDOWN:
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
        """Validate modules in this process, then spawn workers.

        Modules listed in *preimport_modules* are imported here first as a
        validation step (confirms they are installed).  Each spawned worker
        then imports them independently at startup so that later ``import``
        statements inside exec'd function code hit ``sys.modules`` and are
        instant (~0 s instead of ~156 s).

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

        return WorkerPool(num_workers, preload_pipelines=preload_pipelines,
                          preimport_modules=preimport_modules)

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
        self._result_queue.put(_SHUTDOWN)
        self._dispatcher.join(timeout=3)
        for w in self._workers:
            w.join(timeout=5)
            if w.is_alive():
                w.kill()
        logger.info("WorkerPool shut down")
