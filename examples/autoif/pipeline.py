#!/usr/bin/env python3
"""
AutoIF Pipeline - Simple config-driven orchestrator

Reads config from OUT_DIR/config.yaml and submits SLURM jobs with dependencies.
All configuration passed as environment variables to job scripts (auto-discovery).
"""

import argparse
import json
import subprocess
import sys
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# =============================================================================
# CONFIG LOADING
# =============================================================================

def load_config(out_dir: Path) -> dict:
    """Load config.yaml from OUT_DIR, copy default if missing"""
    import yaml

    config_path = out_dir / 'config.yaml'

    if not config_path.exists():
        default_config = Path(__file__).parent / 'config.default.yaml'
        if not default_config.exists():
            print(f"ERROR: No config at {config_path} and no default config found")
            sys.exit(1)

        print(f"No config found, copying default to {config_path}")
        out_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(default_config, config_path)
        print(f"Please edit {config_path} and rerun")
        sys.exit(0)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Validate required fields
    if not config.get('pipeline'):
        print("ERROR: pipeline not defined in config")
        sys.exit(1)

    return config


# =============================================================================
# PARAMETER BUILDING (GENERIC CASCADE RESOLUTION)
# =============================================================================

# Reserved keys that should not be passed as env vars
RESERVED_KEYS = {
    'experiment', 'pipeline', 'jobs'
}


def resolve_config_value(key: str, global_cfg: dict, job_cfg: dict):
    """
    Generic cascade resolution: job > global.

    Returns None if key not found at any level.
    """
    if job_cfg and key in job_cfg:
        return job_cfg[key]
    if global_cfg and key in global_cfg:
        return global_cfg[key]
    return None


def build_env_vars(config: dict, job: str, out_dir: Path) -> dict:
    """
    Build environment variables for job using generic cascade resolution.

    Any config at global level can be overridden at job level using the same keyword.
    Cascade: job config > global config
    """
    env = {}

    # Extract config levels
    global_cfg = config
    job_cfg = config.get('jobs', {}).get(job, {})

    # Handle 'pass' keyword in YAML (becomes None in Python)
    if job_cfg is None:
        job_cfg = {}

    # 1. Collect ALL keys from all levels (excluding reserved keys)
    all_keys = set()
    all_keys.update(k for k in global_cfg.keys() if k not in RESERVED_KEYS)
    all_keys.update(k for k in job_cfg.keys())

    # 2. Generic cascade resolution for all keys
    for key in all_keys:
        value = resolve_config_value(key, global_cfg, job_cfg)
        if value is not None:
            env[key] = str(value)

    # 3. Output directory
    env['out_dir'] = str(out_dir)

    # 4. Resolve file paths to absolute paths (relative to out_dir)
    for key, value in list(env.items()):
        # If value looks like a relative file path, resolve to out_dir
        if isinstance(value, str) and value and not value.startswith('/'):
            # Check if it's a file path (ends with common extensions or is a directory name)
            if any(value.endswith(ext) for ext in ['.jsonl', '.txt', '.json', '.csv']) or \
               'dataset' in key.lower() or 'dir' in key.lower():
                # Keep HF dataset paths and absolute paths as-is
                if not ('/' in value and len(value) > 50):  # Heuristic: long paths with / are likely HF paths
                    env[key] = str(out_dir / value)

    return env


# =============================================================================
# JOB SUBMISSION
# =============================================================================

def submit_job(job_script: str, env_vars: dict, dependency: Optional[str] = None) -> Optional[str]:
    """Submit SLURM job with environment variables, return job ID"""
    cmd = ['sbatch']

    if dependency:
        cmd += ['--dependency', dependency]

    export_vars = {}
    for k, v in env_vars.items():
        export_vars[k] = v

    export_str = ','.join([f'{k}={v}' for k, v in export_vars.items()])
    cmd += ['--export', f'ALL,{export_str}']

    cmd.append(job_script)
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)

    if result.returncode == 0:
        # Parse "Submitted batch job 12345"
        parts = result.stdout.strip().split()
        if len(parts) >= 4:
            return parts[-1]
    else:
        print(f"  Error: {result.stderr.strip()}")

    return None


def get_pipeline_sequence(config: dict) -> List[Tuple[str, bool]]:
    """
    Parse pipeline config into list of (job_name, enabled) tuples.

    Pipeline format: list of dicts with job_name: enabled
    Example: [{augmentation_generation: true}, {verifiers_generation: false}]
    """
    pipeline = config.get('pipeline', [])
    sequence = []

    for item in pipeline:
        if isinstance(item, dict):
            for job_name, enabled in item.items():
                sequence.append((job_name, enabled))
        elif isinstance(item, str):
            # If just a string, assume enabled
            sequence.append((item, True))

    return sequence


def submit_pipeline(config: dict, out_dir: Path) -> Dict[str, str]:
    """Submit all enabled jobs with dependencies"""
    print("\nSubmitting pipeline...\n")

    job_ids = {}
    prev_job_id = None

    pipeline_sequence = get_pipeline_sequence(config)

    for job_name, enabled in pipeline_sequence:
        if not enabled:
            continue

        # Job script path convention: jobs/{job_name}.sh
        job_script = f"jobs/{job_name}.sh"
        env_vars = build_env_vars(config, job_name, out_dir)

        dependency = f"afterany:{prev_job_id}" if prev_job_id else None
        job_id = submit_job(job_script, env_vars, dependency)

        if job_id:
            job_ids[job_name] = job_id
            prev_job_id = job_id
            dep_str = f" (depends on {prev_job_id})" if dependency else ""
            print(f" {job_name:45} → {job_id}{dep_str}")
        else:
            print(f" Failed to submit {job_name}")
            # Cancel already submitted jobs
            cancel_jobs(list(job_ids.values()))
            sys.exit(1)

    return job_ids


# =============================================================================
# STATUS CHECKING
# =============================================================================

def get_job_status(job_id: str) -> str:
    """Get SLURM job status using sacct"""
    result = subprocess.run(
        ['sacct', '-j', job_id, '--format=State', '--noheader', '--parsable2'],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True
    )

    if result.returncode == 0 and result.stdout.strip():
        # Take first status (main job)
        return result.stdout.strip().split('\n')[0]

    return 'UNKNOWN'


def check_pipeline_status(job_ids: Dict[str, str]) -> Tuple[bool, List[str], List[str]]:
    """
    Check pipeline status.

    Returns:
        (all_done, failed_jobs, running_jobs)
    """
    failed = []
    running = []
    all_done = True

    for job, job_id in job_ids.items():
        status = get_job_status(job_id)

        if status in ['FAILED', 'CANCELLED', 'TIMEOUT', 'NODE_FAIL', 'OUT_OF_MEMORY'] or 'CANCELLED' in status:
            failed.append(job)
            all_done = False
        elif status in ['PENDING', 'RUNNING']:
            running.append(job)
            all_done = False

    return all_done, failed, running


def print_status(job_ids: Dict[str, str]):
    """Print pipeline status"""
    print("\nPipeline Status:\n" + "=" * 70)

    for job, job_id in job_ids.items():
        status = get_job_status(job_id)
        print(f"{job:40} {job_id:10} {status}")

    print("=" * 70 + "\n")


# =============================================================================
# JOB HISTORY (MINIMAL)
# =============================================================================

def save_job_ids(out_dir: Path, job_ids: Dict[str, str]):
    """Save job IDs for recovery"""
    history_file = out_dir / 'job_ids.json'
    with open(history_file, 'w') as f:
        json.dump(job_ids, f, indent=2)


def load_job_ids(out_dir: Path) -> Dict[str, str]:
    """Load last submitted job IDs"""
    history_file = out_dir / 'job_ids.json'
    if history_file.exists():
        with open(history_file, 'r') as f:
            return json.load(f)
    return {}


# =============================================================================
# FAILURE RECOVERY
# =============================================================================

def cancel_jobs(job_ids: List[str]):
    """Cancel SLURM jobs"""
    for job_id in job_ids:
        subprocess.run(['scancel', job_id], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)


def get_all_jobs_ordered(config: dict) -> List[str]:
    """Get all enabled jobs in pipeline order as job names"""
    result = []
    pipeline_sequence = get_pipeline_sequence(config)

    for job_name, enabled in pipeline_sequence:
        if enabled:
            result.append(job_name)

    return result


def handle_force(config: dict, out_dir: Path):
    """Cancel all existing jobs and resubmit entire pipeline"""
    print("\n  Force mode: Cancelling existing jobs and resubmitting...\n")

    old_jobs = load_job_ids(out_dir)
    if old_jobs:
        cancel_jobs(list(old_jobs.values()))
        
    print(f"Re-running all jobs. Do you need to delete dispatcher checkpoints for generator jobs?")
    job_ids = submit_pipeline(config, out_dir)
    save_job_ids(out_dir, job_ids)


def handle_resubmit_failed(config: dict, out_dir: Path):
    """Resubmit from first failed job onwards"""
    old_jobs = load_job_ids(out_dir)

    if not old_jobs:
        print("No previous jobs found")
        sys.exit(1)

    all_done, failed, running = check_pipeline_status(old_jobs)

    if not failed:
        print("\nNo failed jobs to resubmit")
        if running:
            print(f"{len(running)} job(s) still running")
        elif all_done:
            print("All jobs completed successfully")
        sys.exit(0)

    print(f"\n Resubmitting {len(failed)} failed job(s) and downstream dependencies...\n")

    # Find first failed job in pipeline order
    all_jobs = get_all_jobs_ordered(config)

    first_failed_idx = min(all_jobs.index(f) for f in failed)

    # Cancel downstream jobs
    to_cancel = [old_jobs[name] for name in all_jobs[first_failed_idx:] if name in running]
    if to_cancel:
        cancel_jobs(to_cancel)
        print(f"Cancelled {len(to_cancel)} downstream job(s)")

    # Get dependency from previous job
    prev_job_id = None
    if first_failed_idx > 0:
        prev_name = all_jobs[first_failed_idx - 1]
        prev_job_id = old_jobs.get(prev_name)

    # Resubmit from failed job onwards
    print("\nResubmitting:\n")
    new_jobs = {}

    for job_name in all_jobs[first_failed_idx:]:
        job_script = f"jobs/{job_name}.sh"
        env_vars = build_env_vars(config, job_name, out_dir)

        dependency = f"afterany:{prev_job_id}" if prev_job_id else None
        job_id = submit_job(job_script, env_vars, dependency)

        if job_id:
            new_jobs[job_name] = job_id
            old_jobs[job_name] = job_id  # Update history
            prev_job_id = job_id
            print(f" {job_name:45} → {job_id}")
        else:
            print(f" Failed to submit {job_name}")
            cancel_jobs(list(new_jobs.values()))
            sys.exit(1)

    save_job_ids(out_dir, old_jobs)


def handle_continue(config: dict, out_dir: Path):
    """Skip failed jobs, submit downstream without dependency"""
    old_jobs = load_job_ids(out_dir)

    if not old_jobs:
        print("No previous jobs found")
        sys.exit(1)

    all_done, failed, running = check_pipeline_status(old_jobs)

    if not failed:
        print("\nNo failed jobs to skip")
        sys.exit(0)

    print(f"\n  Skipping {len(failed)} failed job(s), submitting downstream...\n")

    # Find first failed job
    all_jobs = get_all_jobs_ordered(config)

    first_failed_idx = min(all_jobs.index(f) for f in failed)

    # Cancel downstream jobs
    to_cancel = [old_jobs[name] for name in all_jobs[first_failed_idx + 1:] if name in running]
    if to_cancel:
        cancel_jobs(to_cancel)
        print(f"Cancelled {len(to_cancel)} downstream job(s)")

    # Submit downstream WITHOUT dependency (assumes data exists)
    print("\nSubmitting downstream jobs:\n")
    new_jobs = {}

    for job_name in all_jobs[first_failed_idx + 1:]:
        job_script = f"jobs/{job_name}.sh"
        env_vars = build_env_vars(config, job_name, out_dir)

        job_id = submit_job(job_script, env_vars, dependency=None)

        if job_id:
            new_jobs[job_name] = job_id
            old_jobs[job_name] = job_id
            print(f" {job_name:45} → {job_id}")
        else:
            print(f" Failed to submit {job_name}")
            cancel_jobs(list(new_jobs.values()))
            sys.exit(1)

    save_job_ids(out_dir, old_jobs)
    print(f"\n    Skipped failed: {', '.join(failed)}")
    print("   Ensure output files exist for downstream jobs\n")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='AutoIF Pipeline - Config-driven SLURM orchestrator',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--out-dir', type=Path, required=True,
                       help='Output directory (must contain config.yaml)')
    parser.add_argument('--force', action='store_true',
                       help='Cancel all jobs and resubmit entire pipeline')
    parser.add_argument('--resubmit-failed', action='store_true',
                       help='Resubmit failed jobs and all downstream dependencies')
    parser.add_argument('--continue', dest='continue_mode', action='store_true',
                       help='Skip failed jobs, submit downstream (assumes data exists)')

    args = parser.parse_args()

    # Load config
    config = load_config(args.out_dir)

    print(f"\nAutoIF Pipeline")
    print(f"Experiment: {config.get('experiment', {}).get('name', 'N/A')}")
    print(f"Output: {args.out_dir}")

    # Handle different modes
    if args.force:
        handle_force(config, args.out_dir)
    elif args.resubmit_failed:
        handle_resubmit_failed(config, args.out_dir)
    elif args.continue_mode:
        handle_continue(config, args.out_dir)
    else:
        # Check for existing jobs
        old_jobs = load_job_ids(args.out_dir)
        if old_jobs:
            print_status(old_jobs)
            all_done, failed, running = check_pipeline_status(old_jobs)

            if failed:
                print(f" {len(failed)} job(s) failed. Use --resubmit-failed or --continue")
                sys.exit(1)
            elif running:
                print(f" {len(running)} job(s) still running")
                sys.exit(0)
            elif all_done:
                print(" All jobs completed. Use --force to rerun")
                sys.exit(0)

        # Fresh submission
        job_ids = submit_pipeline(config, args.out_dir)
        save_job_ids(args.out_dir, job_ids)


if __name__ == "__main__":
    main()
