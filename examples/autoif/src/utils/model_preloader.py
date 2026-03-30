"""Pre-download NLP models detected in generated function code.

Scans all function strings for known NLP library patterns (spaCy model
loads, trankit pipeline initialisations, NLTK resource imports) and
downloads the required artefacts once before cross-validation starts.
This avoids per-subprocess download overhead and ensures models that
are distributed as pip packages (spaCy) are actually installed.
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
# Warmup script — executed in a subprocess.
# Redirects sys.stdout to stderr BEFORE importing anything so that library
# print() calls (trankit's "Loading pretrained XLM-Roberta…") don't corrupt
# the JSON result written to the real stdout fd at the end.
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

    print(f"[warmup] creating Pipeline(lang='{lang}') ...", flush=True)
    p = trankit.Pipeline(lang=lang)
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


def _download_spacy_model(model: str, logger: logging.Logger) -> bool:
    # Check if already installed — avoids a slow download on compute nodes.
    try:
        proc = subprocess.run(
            [sys.executable, "-c", f"import spacy; spacy.load('{model}')"],
            capture_output=True, text=True, timeout=30,
        )
        if proc.returncode == 0:
            logger.info("  spaCy model %s already installed — skipping download", model)
            return True
    except subprocess.TimeoutExpired:
        pass
    t0 = time.monotonic()
    logger.info("Pre-downloading spaCy model: %s ...", model)
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "spacy", "download", model],
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


def _download_nltk_resource(resource: str, logger: logging.Logger) -> bool:
    t0 = time.monotonic()
    logger.info("Pre-downloading NLTK resource: %s ...", resource)
    script = f"import nltk; nltk.download('{resource}', quiet=True)"
    try:
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, timeout=DOWNLOAD_TIMEOUT,
        )
        elapsed = time.monotonic() - t0
        if proc.returncode == 0:
            logger.info("  NLTK resource %s OK (%.0fs)", resource, elapsed)
            return True
        logger.warning("  NLTK resource %s FAILED (%.0fs): %s",
                        resource, elapsed, proc.stderr[-300:] if proc.stderr else "")
    except subprocess.TimeoutExpired:
        logger.warning("  NLTK resource %s timed out after %ds", resource, DOWNLOAD_TIMEOUT)
    except Exception as e:
        logger.warning("  NLTK resource %s error: %s", resource, e)
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

def _verify_xlmr_cached(logger: logging.Logger) -> bool:
    """Verify xlm-roberta-base is present in the HF cache.

    Does NOT download — downloading on compute nodes is slow and unreliable.
    If the model is missing, logs an error with pre-download instructions.
    """
    cache_dir = _hf_cache_dir(logger)
    if not cache_dir:
        return False

    logger.info("Checking %s in HF cache: %s", _TRANKIT_EMBEDDING, cache_dir)

    model_file = _find_model_safetensors(cache_dir, _TRANKIT_EMBEDDING)
    if model_file:
        fsize = os.path.getsize(model_file)
        logger.info("  Found: %s (size=%d bytes)", model_file, fsize)
        return True

    logger.error(
        "  %s NOT FOUND in HF cache at %s\n"
        "  Pre-download from the login node:\n"
        "    module load cray-python\n"
        "    HF_HOME=%s python3 -c \"\n"
        "      from huggingface_hub import snapshot_download\n"
        "      snapshot_download('%s', cache_dir='%s')\n"
        "    \"",
        _TRANKIT_EMBEDDING,
        cache_dir,
        os.environ.get("HF_HOME", "/scratch/project_462000963/cache"),
        _TRANKIT_EMBEDDING,
        cache_dir,
    )
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
# Trankit warm-up: full end-to-end Pipeline test
# ---------------------------------------------------------------------------

def _warmup_trankit(lang: str, logger: logging.Logger) -> bool:
    """Fully initialize trankit.Pipeline in a subprocess and verify it works.

    Catches model-loading errors (state_dict mismatches, missing files)
    BEFORE workers start, and reports them loudly.
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

        # Always log stderr — it has trankit's own progress messages and any errors
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
    logger.info("Environment: HF_HOME=%s  TRANSFORMERS_CACHE=%s  cwd=%s  python=%s",
                os.environ.get("HF_HOME", "(unset)"),
                os.environ.get("TRANSFORMERS_CACHE", "(unset)"),
                os.getcwd(),
                sys.executable)

    spacy_models = _scan_spacy_models(all_function_strings)
    trankit_langs = _scan_trankit_languages(all_function_strings)
    nltk_resources = _scan_nltk_resources(all_function_strings)

    if not spacy_models and not trankit_langs and not nltk_resources:
        logger.info("No NLP model dependencies detected in functions")
        return {"spacy": [], "trankit": [], "nltk": []}

    logger.info("Detected NLP models: spacy=%s, trankit=%s, nltk=%s",
                sorted(spacy_models), sorted(trankit_langs), sorted(nltk_resources))

    loaded: Dict[str, List[str]] = {"spacy": [], "trankit": [], "nltk": []}

    for model in sorted(spacy_models):
        if _download_spacy_model(model, logger):
            loaded["spacy"].append(model)

    # Verify xlm-roberta-base is cached (downloading on compute nodes is unreliable)
    xlmr_ok = False
    if trankit_langs:
        xlmr_ok = _verify_xlmr_cached(logger)

    for lang in sorted(trankit_langs):
        if _download_trankit_model(lang, logger):
            loaded["trankit"].append(lang)

    for resource in sorted(nltk_resources):
        if _download_nltk_resource(resource, logger):
            loaded["nltk"].append(resource)

    # Full end-to-end warm-up: load trankit.Pipeline and process a test sentence
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
                              "pre-download it from the login node")

    # If the base model is unavailable AND warmup failed, trankit is unusable.
    if not xlmr_ok and not warmup_ok and loaded.get("trankit"):
        logger.warning("Clearing loaded trankit languages — xlm-roberta-base "
                       "unavailable and warmup failed; trankit is unusable")
        loaded["trankit"] = []

    # Set offline mode after preloading.  Workers should never download
    # models themselves — it causes cache corruption on Lustre.
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    logger.info("Set TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 for worker subprocesses")

    # Tell the function executor where the trankit cache lives so workers
    # can find it via the environment.
    if trankit_langs:
        abs_cache = os.path.abspath(_TRANKIT_CACHE_DIR)
        logger.info("Trankit cache_dir: %s", abs_cache)

    total = sum(len(v) for v in loaded.values())
    expected = len(spacy_models) + len(trankit_langs) + len(nltk_resources)
    logger.info("Model pre-loading complete: %d/%d downloads succeeded, "
                "xlmr_cached=%s, trankit_warmup=%s",
                total, expected, xlmr_ok, warmup_ok)

    return loaded
