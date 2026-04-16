"""Pre-download NLP models detected in generated function code.

Scans all function strings for known NLP library patterns (spaCy model
loads, stanza pipeline initialisations, trankit pipeline initialisations,
NLTK resource imports) and downloads the required artefacts once before
cross-validation starts.  This avoids per-subprocess download overhead
and ensures models are available in the same environment workers use.

Checks and downloads run **in-process** (spacy, stanza, nltk) so they
see exactly the same sys.path and PYTHONUSERBASE as the spawned workers.
Only trankit warm-up and external downloads use subprocesses.
"""

import glob as _glob
import json
import logging
import os
import re
import subprocess
import sys
import time
import zipfile
from typing import Dict, List, Optional, Set

DOWNLOAD_TIMEOUT = int(os.getenv("MODEL_DOWNLOAD_TIMEOUT", 300))

_TRANKIT_LANG_ALIASES: Dict[str, str] = {
    "fi": "finnish",
    "en": "english",
    "de": "german",
    "fr": "french",
    "sv": "swedish",
    "et": "estonian",
    "no": "norwegian",
    "da": "danish",
}

_NLTK_IMPORT_TO_RESOURCE: Dict[str, str] = {
    "wordnet": "wordnet",
    "stopwords": "stopwords",
    "averaged_perceptron_tagger": "averaged_perceptron_tagger",
    "punkt": "punkt",
    "punkt_tab": "punkt_tab",
}

_TRANKIT_HF_URL = (
    "https://huggingface.co/uonlp/trankit/resolve/main/"
    "models/{version}/{embedding}/{lang}.zip"
)
_TRANKIT_CACHE_DIR = os.path.join("cache", "trankit")
_TRANKIT_EMBEDDING = "xlm-roberta-base"
_TRANKIT_VERSION = "v1.0.0"


# ---------------------------------------------------------------------------
# Scanners — detect NLP library usage in function code
# ---------------------------------------------------------------------------

def _scan_spacy_models(functions: List[str]) -> Set[str]:
    """Detect ``spacy.load("model_name")`` calls."""
    models: Set[str] = set()
    for fn in functions:
        for m in re.findall(r'spacy\.load\(\s*["\']([^"\']+)["\']\s*\)', fn):
            models.add(m)
    return models


def _scan_trankit_languages(functions: List[str]) -> Set[str]:
    """Detect ``trankit.Pipeline('lang')`` or ``trankit.Pipeline(lang='lang')``."""
    langs: Set[str] = set()
    for fn in functions:
        for m in re.findall(r'trankit\.Pipeline\(\s*["\'](\w+)["\']\s*\)', fn):
            langs.add(_TRANKIT_LANG_ALIASES.get(m.lower(), m.lower()))
        for m in re.findall(r'trankit\.Pipeline\(\s*lang\s*=\s*["\'](\w+)["\']\s*\)', fn):
            langs.add(_TRANKIT_LANG_ALIASES.get(m.lower(), m.lower()))
    return langs


def _scan_stanza_languages(functions: List[str]) -> Set[str]:
    """Detect ``stanza.Pipeline('lang')`` or ``stanza.Pipeline(lang='lang')``."""
    langs: Set[str] = set()
    for fn in functions:
        for m in re.findall(r'stanza\.Pipeline\(\s*["\'](\w+)["\']\s*', fn):
            langs.add(m.lower())
        for m in re.findall(r'stanza\.Pipeline\(\s*lang\s*=\s*["\'](\w+)["\']\s*', fn):
            langs.add(m.lower())
    return langs


def _scan_nltk_resources(functions: List[str]) -> Set[str]:
    """Detect NLTK corpus/tokenizer imports that need downloadable data."""
    resources: Set[str] = set()
    for fn in functions:
        for corpus in re.findall(r'from\s+nltk\.corpus\s+import\s+(\w+)', fn):
            mapped = _NLTK_IMPORT_TO_RESOURCE.get(corpus)
            if mapped:
                resources.add(mapped)
        if re.search(r'word_tokenize|sent_tokenize|nltk\.tokenize', fn):
            resources.add("punkt")
            resources.add("punkt_tab")
    return resources


# ---------------------------------------------------------------------------
# In-process downloaders — spacy, stanza, nltk
#
# These run in the same process as the caller so they see the exact same
# sys.path, PYTHONUSERBASE, and installed packages that spawned workers
# will see.  No subprocess environment mismatches.
# ---------------------------------------------------------------------------

def _ensure_spacy_model(model: str, logger: logging.Logger) -> bool:
    """Ensure a spaCy model is loadable, downloading if needed.

    Check is in-process (same sys.path as workers).  Download requires
    a subprocess because ``spacy download`` is a pip install wrapper.
    """
    try:
        import spacy
        spacy.load(model)
        logger.info("  spaCy model %s already installed", model)
        return True
    except OSError:
        pass  # model not found — download below
    except ImportError:
        logger.warning("  spaCy is not installed — cannot load model %s", model)
        return False

    t0 = time.monotonic()
    logger.info("  Downloading spaCy model: %s ...", model)
    try:
        # spaCy models are hosted on GitHub releases, not PyPI, so we
        # must use `spacy download` for URL resolution.  Extra args
        # after the model name are forwarded to `pip install`; pass
        # --user so the model installs into PYTHONUSERBASE (required
        # in read-only containers like Singularity).
        proc = subprocess.run(
            [sys.executable, "-m", "spacy", "download", model, "--user"],
            capture_output=True, text=True, timeout=DOWNLOAD_TIMEOUT,
        )
        elapsed = time.monotonic() - t0
        if proc.returncode == 0:
            logger.info("  spaCy model %s OK (%.0fs)", model, elapsed)
            return True
        logger.warning("  spaCy model %s FAILED (%.0fs): %s",
                        model, elapsed, proc.stderr[-300:] if proc.stderr else "")
    except subprocess.TimeoutExpired:
        logger.warning("  spaCy model %s timed out after %ds", model, DOWNLOAD_TIMEOUT)
    except Exception as e:
        logger.warning("  spaCy model %s error: %s", model, e)
    return False


def _ensure_stanza_model(lang: str, logger: logging.Logger) -> bool:
    """Ensure a stanza model is downloaded, downloading if needed.

    Both the check and the download run in-process.
    """
    try:
        import stanza
    except ImportError:
        logger.warning("  stanza is not installed — cannot load model %s", lang)
        return False

    # Check if model is already usable.
    try:
        stanza.Pipeline(lang, processors='tokenize', verbose=False,
                        download_method=stanza.DownloadMethod.REUSE_RESOURCES)
        logger.info("  stanza model %s already downloaded", lang)
        return True
    except Exception:
        pass  # model not found — download below

    t0 = time.monotonic()
    logger.info("  Downloading stanza model: %s ...", lang)
    try:
        stanza.download(lang, verbose=False)
        elapsed = time.monotonic() - t0
        logger.info("  stanza model %s OK (%.0fs)", lang, elapsed)
        return True
    except Exception as e:
        elapsed = time.monotonic() - t0
        logger.warning("  stanza model %s FAILED (%.0fs): %s", lang, elapsed, e)
    return False


def _set_stanza_offline(logger: logging.Logger) -> None:
    """Disable stanza's resource-update check.

    By default ``stanza.Pipeline(...)`` downloads ``resources.json``
    on every invocation to check for model updates.

    Stanza has no global env-var to control this, so we monkey-patch
    the default ``download_method`` to ``None`` (skip all downloads).
    Since models are already downloaded in the preloading phase, this
    is safe.  This patch applies to the parent process (for model
    verification steps); spawned workers apply their own patch at
    startup in ``_worker_loop``.
    """
    try:
        import stanza.pipeline.core as _core
        _original_init = _core.Pipeline.__init__

        def _patched_init(self, *args, **kwargs):
            kwargs.setdefault("download_method", None)
            kwargs.setdefault("verbose", False)
            return _original_init(self, *args, **kwargs)

        _core.Pipeline.__init__ = _patched_init
        logger.info("Patched stanza.Pipeline default download_method=None for workers")
    except Exception as e:
        logger.warning("Failed to patch stanza download_method: %s", e)


def _ensure_nltk_resource(resource: str, logger: logging.Logger) -> bool:
    """Ensure an NLTK resource is downloaded, downloading if needed.

    Both the check and the download run in-process.
    """
    try:
        import nltk
    except ImportError:
        logger.warning("  nltk is not installed — cannot download resource %s", resource)
        return False

    # Check if already available.
    try:
        nltk.data.find(f"tokenizers/{resource}" if resource.startswith("punkt")
                       else f"corpora/{resource}")
        logger.info("  NLTK resource %s already downloaded", resource)
        return True
    except LookupError:
        pass  # not found — download below

    t0 = time.monotonic()
    logger.info("  Downloading NLTK resource: %s ...", resource)
    try:
        if nltk.download(resource, quiet=True):
            elapsed = time.monotonic() - t0
            logger.info("  NLTK resource %s OK (%.0fs)", resource, elapsed)
            return True
        logger.warning("  NLTK resource %s download returned False", resource)
    except Exception as e:
        elapsed = time.monotonic() - t0
        logger.warning("  NLTK resource %s FAILED (%.0fs): %s", resource, elapsed, e)
    return False


# ---------------------------------------------------------------------------
# HF cache helpers
# ---------------------------------------------------------------------------

def _hf_cache_dir(logger: logging.Logger) -> Optional[str]:
    """Return the HuggingFace hub cache directory, or None if not configured."""
    hf_home = os.environ.get("HF_HOME")
    if not hf_home:
        logger.warning("HF_HOME not set — cannot determine HF cache path")
        return None
    return os.path.abspath(os.path.join(hf_home, "hub"))


def _find_model_safetensors(cache_dir: str, model_id: str) -> Optional[str]:
    """Find model.safetensors (or pytorch_model.bin) in the HF cache."""
    model_cache = os.path.join(cache_dir, f"models--{model_id.replace('/', '--')}")
    if not os.path.isdir(model_cache):
        return None
    for name in ("model.safetensors", "pytorch_model.bin"):
        candidates = _glob.glob(os.path.join(model_cache, "snapshots", "*", name))
        if candidates:
            return os.path.realpath(candidates[0])
    return None


# ---------------------------------------------------------------------------
# XLM-RoBERTa base model verification
# ---------------------------------------------------------------------------

def _ensure_xlmr_cached(logger: logging.Logger) -> bool:
    """Ensure xlm-roberta-base is present in the HF cache, downloading if needed."""
    cache_dir = _hf_cache_dir(logger)
    if not cache_dir:
        return False

    logger.info("Checking %s in HF cache: %s", _TRANKIT_EMBEDDING, cache_dir)

    model_file = _find_model_safetensors(cache_dir, _TRANKIT_EMBEDDING)
    if model_file:
        fsize = os.path.getsize(model_file)
        logger.info("  Found: %s (size=%d bytes)", model_file, fsize)
        return True

    logger.info("  %s not found in HF cache — downloading ...", _TRANKIT_EMBEDDING)
    t0 = time.monotonic()
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(_TRANKIT_EMBEDDING, cache_dir=cache_dir)
        elapsed = time.monotonic() - t0

        model_file = _find_model_safetensors(cache_dir, _TRANKIT_EMBEDDING)
        if model_file:
            logger.info("  %s downloaded successfully (%.0fs)", _TRANKIT_EMBEDDING, elapsed)
            return True
        logger.warning("  %s download completed but model file not found", _TRANKIT_EMBEDDING)
    except ImportError:
        logger.warning("  huggingface_hub not installed — cannot download %s", _TRANKIT_EMBEDDING)
    except Exception as e:
        elapsed = time.monotonic() - t0
        logger.warning("  %s download error (%.0fs): %s", _TRANKIT_EMBEDDING, elapsed, e)
    return False


# ---------------------------------------------------------------------------
# Trankit language model download
# ---------------------------------------------------------------------------

def _download_trankit_model(lang: str, logger: logging.Logger) -> bool:
    """Download a trankit language model from the HuggingFace mirror."""
    t0 = time.monotonic()
    lang_dir = os.path.join(_TRANKIT_CACHE_DIR, _TRANKIT_EMBEDDING, lang)
    marker = os.path.join(lang_dir, f"{lang}.downloaded")

    if os.path.exists(marker):
        logger.info("  trankit model %s already cached at %s", lang, lang_dir)
        return True

    url = _TRANKIT_HF_URL.format(
        version=_TRANKIT_VERSION,
        embedding=_TRANKIT_EMBEDDING,
        lang=lang,
    )
    logger.info("Pre-downloading trankit model: %s from %s", lang, url)

    os.makedirs(lang_dir, exist_ok=True)
    zip_path = os.path.join(lang_dir, f"{lang}.zip")

    try:
        proc = subprocess.run(
            ["curl", "-fSL", "--max-time", str(DOWNLOAD_TIMEOUT),
             "-o", zip_path, url],
            capture_output=True, text=True, timeout=DOWNLOAD_TIMEOUT + 30,
        )
        elapsed = time.monotonic() - t0
        if proc.returncode != 0:
            logger.warning("  trankit model %s download FAILED (%.0fs): %s",
                           lang, elapsed, proc.stderr[-300:] if proc.stderr else "")
            return False

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(lang_dir)
        os.remove(zip_path)

        if not os.path.exists(marker):
            open(marker, "w").close()

        logger.info("  trankit model %s OK (%.0fs)", lang, elapsed)
        return True

    except subprocess.TimeoutExpired:
        logger.warning("  trankit model %s timed out after %ds", lang, DOWNLOAD_TIMEOUT)
    except zipfile.BadZipFile:
        logger.warning("  trankit model %s: corrupt zip file", lang)
        if os.path.exists(zip_path):
            os.remove(zip_path)
    except Exception as e:
        logger.warning("  trankit model %s error: %s", lang, e)
    return False


# ---------------------------------------------------------------------------
# Trankit warm-up: full end-to-end Pipeline test (subprocess for memory
# isolation — loading a trankit Pipeline consumes significant memory that
# would bloat the parent before forking workers).
# ---------------------------------------------------------------------------
_WARMUP_SCRIPT = r"""
import sys, os, json, traceback

_real_stdout = sys.stdout
sys.stdout = sys.stderr

lang = sys.argv[1] if len(sys.argv) > 1 else "finnish"
result = {}

try:
    print(f"[warmup] importing trankit ...", flush=True)
    import trankit
    print(f"[warmup] trankit imported, version={getattr(trankit, '__version__', '?')}", flush=True)

    print(f"[warmup] creating Pipeline(lang='{lang}', gpu=False) ...", flush=True)
    p = trankit.Pipeline(lang=lang, gpu=False)
    print(f"[warmup] Pipeline created, running test sentence ...", flush=True)

    out = p("Tama on testi.")
    n_tokens = sum(len(s.get("tokens", [])) for s in out.get("sentences", []))
    print(f"[warmup] test sentence OK, {n_tokens} tokens", flush=True)
    result = {"ok": True, "tokens": n_tokens}

except Exception as e:
    tb = traceback.format_exc()
    print(f"[warmup] EXCEPTION:\n{tb}", flush=True)
    result = {"ok": False, "error": str(e)[:1000], "traceback": tb[-2000:]}

sys.stdout = _real_stdout
json.dump(result, sys.stdout)
"""


def _warmup_trankit(lang: str, logger: logging.Logger) -> bool:
    """Fully initialize trankit.Pipeline in a subprocess and verify it works.

    Uses a subprocess for memory isolation — loading a trankit Pipeline
    consumes significant memory that we don't want in the parent before fork.
    """
    t0 = time.monotonic()
    logger.info("Warming up trankit Pipeline('%s') ...", lang)
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _WARMUP_SCRIPT, lang],
            capture_output=True, text=True,
            timeout=DOWNLOAD_TIMEOUT + 120,
        )
        elapsed = time.monotonic() - t0

        if proc.stderr:
            for line in proc.stderr.strip().splitlines()[-30:]:
                logger.info("  [warmup stderr] %s", line)

        if proc.returncode == 0 and proc.stdout.strip():
            try:
                result = json.loads(proc.stdout)
            except json.JSONDecodeError:
                logger.warning("  trankit warm-up: bad JSON from subprocess "
                              "(stdout likely polluted)\n  stdout[:500]: %s",
                              proc.stdout[:500])
                return False

            if result.get("ok"):
                logger.info("  trankit warm-up OK (%.0fs, %d tokens)",
                            elapsed, result.get("tokens", 0))
                return True
            logger.warning("  trankit warm-up FAILED (%.0fs):\n  error: %s\n  traceback:\n%s",
                          elapsed,
                          result.get("error", "unknown"),
                          result.get("traceback", "(none)"))
        else:
            logger.warning("  trankit warm-up FAILED (%.0fs, exit=%d)\n"
                          "  stdout[:500]: %s\n  stderr[-1000:]:\n%s",
                          elapsed, proc.returncode,
                          proc.stdout[:500] if proc.stdout else "(empty)",
                          proc.stderr[-1000:] if proc.stderr else "(empty)")
    except subprocess.TimeoutExpired:
        logger.warning("  trankit warm-up timed out after %ds", DOWNLOAD_TIMEOUT + 120)
    except Exception as e:
        logger.warning("  trankit warm-up exception: %s", e)
    return False


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def scan_and_preload_models(
    all_function_strings: List[str],
    logger: logging.Logger,
    timeout: int = DOWNLOAD_TIMEOUT,
) -> Dict[str, List[str]]:
    """Scan function code for NLP model usage and pre-download detected models.

    Runs once at startup before cross-validation begins.  Downloads are
    sequential (they share the network) and failures are logged as warnings
    without aborting the run.

    Returns:
        Dict mapping library name to list of successfully pre-loaded models.
    """
    global DOWNLOAD_TIMEOUT
    DOWNLOAD_TIMEOUT = timeout

    # Log environment for debugging
    logger.info("Environment: HF_HOME=%s  PYTHONUSERBASE=%s  cwd=%s  python=%s",
                os.environ.get("HF_HOME", "(unset)"),
                os.environ.get("PYTHONUSERBASE", "(unset)"),
                os.getcwd(),
                sys.executable)

    spacy_models = _scan_spacy_models(all_function_strings)
    stanza_langs = _scan_stanza_languages(all_function_strings)
    trankit_langs = _scan_trankit_languages(all_function_strings)
    nltk_resources = _scan_nltk_resources(all_function_strings)

    if not spacy_models and not stanza_langs and not trankit_langs and not nltk_resources:
        logger.info("No NLP model dependencies detected in functions")
        return {"spacy": [], "stanza": [], "trankit": [], "nltk": []}

    logger.info("Detected NLP models: spacy=%s, stanza=%s, trankit=%s, nltk=%s",
                sorted(spacy_models), sorted(stanza_langs),
                sorted(trankit_langs), sorted(nltk_resources))

    loaded: Dict[str, List[str]] = {"spacy": [], "stanza": [], "trankit": [], "nltk": []}

    for model in sorted(spacy_models):
        if _ensure_spacy_model(model, logger):
            loaded["spacy"].append(model)

    for lang in sorted(stanza_langs):
        if _ensure_stanza_model(lang, logger):
            loaded["stanza"].append(lang)

    # After downloading stanza models, patch Pipeline to skip update checks.
    # Must happen before fork() so workers inherit the patched default.
    if loaded["stanza"]:
        _set_stanza_offline(logger)

    # Ensure xlm-roberta-base is in the HF cache (download if missing)
    xlmr_ok = False
    if trankit_langs:
        xlmr_ok = _ensure_xlmr_cached(logger)

    for lang in sorted(trankit_langs):
        if _download_trankit_model(lang, logger):
            loaded["trankit"].append(lang)

    for resource in sorted(nltk_resources):
        if _ensure_nltk_resource(resource, logger):
            loaded["nltk"].append(resource)

    # Full end-to-end warm-up: load trankit.Pipeline and process a test
    # sentence.  Runs in a subprocess for memory isolation.
    warmup_ok = False
    for lang in sorted(trankit_langs):
        if _warmup_trankit(lang, logger):
            warmup_ok = True
        else:
            logger.warning("trankit warm-up failed for '%s'", lang)
            if xlmr_ok:
                logger.warning("  xlm-roberta-base is cached — "
                              "problem may be in trankit adapter loading")
            else:
                logger.warning("  xlm-roberta-base also missing — "
                              "download failed")

    # If the base model is unavailable AND warmup failed, trankit is unusable.
    if not xlmr_ok and not warmup_ok and loaded.get("trankit"):
        logger.warning("Clearing loaded trankit languages — xlm-roberta-base "
                       "unavailable and warmup failed; trankit is unusable")
        loaded["trankit"] = []

    # Set offline mode after preloading ONLY when all models are cached.
    # On Lustre this avoids cache corruption from concurrent worker downloads.
    # On other filesystems (e.g. TensorWave) we leave online mode so that
    # trankit can auto-download models on first import as a fallback.
    if xlmr_ok and warmup_ok:
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
        logger.info("Set TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 for worker subprocesses")
    else:
        logger.info("Skipping offline mode — preloading incomplete, "
                     "workers may download models on first use")

    # Tell the function executor where the trankit cache lives so workers
    # can find it via the environment.
    if trankit_langs:
        abs_cache = os.path.abspath(_TRANKIT_CACHE_DIR)
        logger.info("Trankit cache_dir: %s", abs_cache)

    total = sum(len(v) for v in loaded.values())
    expected = len(spacy_models) + len(stanza_langs) + len(trankit_langs) + len(nltk_resources)
    logger.info("Model pre-loading complete: %d/%d downloads succeeded, "
                "xlmr_cached=%s, trankit_warmup=%s",
                total, expected, xlmr_ok, warmup_ok)

    return loaded
