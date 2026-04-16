# CLAUDE.md - Project Notes for AI Agents

## Stanza + fork() Deadlock (2026-04-16)

### Problem

The cross-validation worker pool (`src/utils/worker_pool.py`) uses `multiprocessing.get_context("fork")` to create workers. Stanza (and PyTorch underneath it) uses internal threads/locks that become permanently deadlocked after `fork()`. This is a well-known POSIX issue: any thread holding a lock at fork time leaves that lock permanently held in the child, causing the child to hang on the next attempt to acquire it.

All stanza-based verifier functions hang indefinitely in forked workers. SIGALRM cannot interrupt the hang because it occurs in C-level code (PyTorch model loading).

### Evidence

Tested with `examples/autoif/test_stanza.py` (run both in Singularity container via SLURM job 11769 and directly on a CPU node):

| Test | Method | Result |
|------|--------|--------|
| 1 | stanza in main process | PASS (2.0s) |
| 2 | exec() in main process | PASS (0.9s) |
| 3 | exec() in fork'd child, no pre-import | FAIL (hung 90s) |
| 4 | exec() in fork'd child, stanza pre-imported | FAIL (hung 90s) |
| 5 | exec() in fork'd child, Pipeline pre-loaded + monkey-patch | FAIL (hung 90s) |
| 6 | exec() in spawn'd child (`multiprocessing.get_context("spawn")`) | PASS (6.4s) |
| 7 | exec() via `subprocess.Popen` | PASS (6.5s) |

Fork hangs regardless of whether stanza is pre-imported, pre-loaded, or monkey-patched. Both `spawn` and `subprocess` work because they start a fresh Python interpreter.

### Affected files

- `examples/autoif/src/utils/worker_pool.py` - Line 202: `ctx = multiprocessing.get_context("fork")`
- `examples/autoif/src/utils/function_executor.py` - Has a `_run_in_subprocess` fallback that already works correctly (used when no worker pool is set)

### Solution direction

Switch the worker pool from `fork` to either `spawn` or `subprocess`. This must be a universal change for ALL cross-validation functions, not a per-library special case. Do not add complexity like library detection to choose between fork/spawn.

Key trade-off: with `fork`, workers inherit `sys.modules` from the parent via COW pages, making imports free. With `spawn`, each worker must import modules itself at startup. This is a one-time cost per worker (~2-10s depending on the library) and is acceptable.

The existing `_run_in_subprocess` / `_run_in_subprocess_batch` codepath in `function_executor.py` already works correctly with stanza and could serve as the sole execution path (eliminating the worker pool entirely). Alternatively, the worker pool can be kept but switched to `spawn` context.

### Other stanza notes

- stanza downloads `resources.json` on every `Pipeline()` call by default. Must set `download_method=None` or use `DownloadMethod.REUSE_RESOURCES` to prevent this. A monkey-patch for this already exists in `model_preloader.py` (`_set_stanza_offline`).
- stanza models for Finnish are ~348MB, stored in `~/.cache/stanza/`.
- spaCy does NOT have this fork problem - it works fine in forked workers.
- trankit also works in forked workers (it was the original NLP library).

### Test script

`examples/autoif/test_stanza.py` - standalone script to reproduce the issue. Can run directly (with stanza+torch installed) or inside the Singularity container. SLURM job wrapper: `examples/autoif/test_stanza_job.sh`.
