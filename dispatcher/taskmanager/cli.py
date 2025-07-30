"""dispatcher-run command‑line entry‑point and reusable `run()` helper.

Supports **two task sources** out of the box:

* FileTaskSource – supply ``--input`` + ``--output``
* DispatcherTaskSource – supply ``--dispatcher host:port``

Example – local file test
------------------------
>>> dispatcher-task-run \
        --task task.MyTask \
        --input input.jsonl --output out.jsonl \
        --model meta-llama/Llama-3.1-8B-Instruct

Example – distributed run with Dispatcher server
------------------------------------------------
>>> dispatcher-task-run \
        --task mypkg.tasks.ChatTask \
        --dispatcher $(hostname):9999 \
        --model meta-llama/Llama-3.1-8B-Instruct

Users who prefer Python can also import ``run`` directly::

    from dispatcher.cli import run
    from task import MyTask

    run(task_cls=MyTask, input_path="input.jsonl", output_path="out.jsonl", model="…")
"""
from __future__ import annotations

import argparse
import importlib
import json
import logging
import signal
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Literal, Optional, Type

from dispatcher.taskmanager.backend import VLLMBackendManager
from dispatcher.taskmanager.taskmanager import TaskManager
from dispatcher.taskmanager.tasksource import FileTaskSource, DispatcherTaskSource
from dispatcher.taskmanager.task.base import Task

logger = logging.getLogger(__name__)


###############################################################################
# Internal helpers
###############################################################################

def _import_dotted(path: str) -> Type[Task]:
    """Import ``pkg.sub.module:Class`` or ``pkg.mod.Class`` dotted path."""
    if ":" in path:
        module_path, _, cls_name = path.partition(":")
    else:
        module_path, _, cls_name = path.rpartition(".")
        if not module_path:
            # simple module at top level, e.g. "task.MyTask"
            module_path, cls_name = path, None
    module: ModuleType = importlib.import_module(module_path)
    if cls_name is None:
        raise ValueError("--task must include a class name (e.g. task.MyTask)")
    return getattr(module, cls_name)


def _install_signal_handlers(backend: VLLMBackendManager):
    def _handler(signum, _):
        logger.info("Signal %s received – shutting down…", signum)
        backend.close()
        sys.exit(0)

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _handler)
        except ValueError:
            # In threads / unsuitable contexts just ignore.
            pass

###############################################################################
# Public runner API (can be imported)
###############################################################################

def run(
    *,
    task_cls: Type[Task],
    model: str,
    # task source selection
    input_path: Optional[str | Path] = None,
    output_path: Optional[str | Path] = None,
    dispatcher_url: Optional[str] = None,
    host: str = "127.0.0.1",
    port: int = 8000,
    launch_vllm: bool = True,
    tensor_parallel: int = 1,
    max_model_len: int = 16_384,
    startup_timeout: int = 1500,
    request_timeout: int = 600,
    silence_vllm_logs: bool = False,
    enforce_eager: bool = False, 
    # task manager params
    workers: int = 16,
    batch_size: int = 4,
) -> None:
    """Wire up TaskSource → TaskManager → VLLM backend and process to completion."""

    def show_handlers():
        for name in logging.root.manager.loggerDict:
            logger = logging.getLogger(name)
            if logger.level <= logging.INFO or any(h.level <= logging.INFO for h in logger.handlers):
                print(f"{name}: level={logger.level}, handlers={logger.handlers}")
    #show_handlers()
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


    if dispatcher_url:
        source = DispatcherTaskSource(dispatcher_url, task_cls, batch_size=batch_size)
        logger.info("Using DispatcherTaskSource at %s (batch=%d)", dispatcher_url, batch_size)
    else:
        if not (input_path and output_path):
            raise ValueError("--input and --output are required when --dispatcher is not provided")
        source = FileTaskSource(input_path, output_path, task_cls, batch_size=batch_size)
        logger.info("Using FileTaskSource %s → %s (batch=%d)", input_path, output_path, batch_size)

    backend = VLLMBackendManager(
        model_name=model,
        host=host,
        port=port,
        launch_server=launch_vllm,
        tensor_parallel_size=tensor_parallel,
        max_model_len=max_model_len,
        startup_timeout=startup_timeout,
        request_timeout=request_timeout,
        disable_output=silence_vllm_logs,
        enforce_eager=enforce_eager,
    )

    _install_signal_handlers(backend)

    manager = TaskManager(num_workers=workers)
    manager.process_tasks(source, backend)

    backend.close()
    logger.info("All tasks completed.")

###############################################################################
# CLI entry‑point
###############################################################################

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="dispatcher-run",
        description="CLI for Dispatcher TaskManager jobs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # task module
    p.add_argument("--task", required=True, help="Dotted path to Task subclass (e.g. task.MyTask)")

    # mutually exclusive sources
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--dispatcher", metavar="HOST:PORT", help="Use DispatcherTaskSource at host:port")
    src.add_argument("--input", help="Input JSONL (file mode)")

    # file mode needs output path too
    p.add_argument("--output", help="Output JSONL (file mode)")

    # model / vLLM
    p.add_argument("--model", required=True, help="HF model ID or path for vLLM")
    p.add_argument("--host", default="127.0.0.1", help="Bind host for launched vLLM server")
    p.add_argument("--port", type=int, default=8000, help="Bind port")
    p.add_argument("--no-launch", action="store_true", help="Do not start a vLLM server (connect only)")
    p.add_argument("--tensor-parallel", type=int, default=1, help="Tensor parallel degree")
    p.add_argument("--max-model-len", type=int, default=16_384, help="Max context length override")
    p.add_argument("--startup-timeout", type=int, default=1500, help="Maximum time to wait for vllm server to start")
    p.add_argument("--request-timeout", type=int, default=600, help="Maximum time to wait for a request")
    p.add_argument("--silence-vllm-logs", action="store_true", help="Suppress all logging from vllm")

    # vllm tuning
    p.add_argument("--enforce-eager", action="store_true", help="Force vLLM to run in eager mode")

    # manager & batches
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=4)

    return p


def main(argv: Optional[list[str]] = None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _build_parser().parse_args(argv)

    task_cls = _import_dotted(args.task)

    run(
        task_cls=task_cls,
        model=args.model,
        input_path=args.input,
        output_path=args.output,
        dispatcher_url=args.dispatcher,
        host=args.host,
        port=args.port,
        launch_vllm=not args.no_launch,
        tensor_parallel=args.tensor_parallel,
        max_model_len=args.max_model_len,
        startup_timeout=args.startup_timeout,
        request_timeout=args.request_timeout,
        silence_vllm_logs=args.silence_vllm_logs,
        enforce_eager=args.enforce_eager,
        workers=args.workers,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
