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

from src.orchestrator.step_config import get_substep_config


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
    if not config.get('pipeline_sequence'):
        print("ERROR: pipeline_sequence not defined in config")
        sys.exit(1)

    if not config.get('data', {}).get('queries_dataset'):
        print("ERROR: data.queries_dataset required in config")
        sys.exit(1)

    return config


# =============================================================================
# PARAMETER BUILDING (AUTO-DISCOVERY)
# =============================================================================

def build_env_vars(config: dict, step: str, substep: str, out_dir: Path) -> dict:
    """
    Build environment variables for substep - auto-discovery approach.

    Pass all relevant config as env vars, let job scripts use what they need.
    """
    env = {}

    # 1. Model (with substep override support)
    env['model'] = config['model']['name']
    substep_cfg = config['steps'][step]['substeps'].get(substep, {})
    if isinstance(substep_cfg, dict) and 'model' in substep_cfg:
        env['model'] = substep_cfg['model']

    # 2. Common parameters
    env['language'] = config['data']['language']
    env['out_dir'] = str(out_dir)

    # 3. All step parameters (auto-pass)
    step_params = config['steps'][step].get('params', {})
    for key, value in step_params.items():
        env[key] = str(value)

    # 4. All file paths (auto-resolve to out_dir)
    for file_key, file_name in config.get('files', {}).items():
        env[file_key] = str(out_dir / file_name)

    # 5. Data source paths
    env['seed_file'] = config['data']['seed_file']
    env['queries_dataset'] = config['data']['queries_dataset']

    # 6. SLURM parameters
    env.update(get_slurm_params(config, step, substep))

    # 7. Advanced settings
    advanced = config.get('advanced', {})
    env['VENV_DIR'] = advanced.get('venv_dir', '.venv')
    env['HF_HOME'] = advanced.get('hf_home', '/scratch/project_462000353/hf_cache')

    return env


def get_slurm_params(config: dict, step: str, substep: str) -> dict:
    """Get SLURM parameters from defaults + config overrides"""
    substep_def = get_substep_config(step, substep)

    slurm = {
        'partition': config['slurm']['gpu_partition'] if substep_def.requires_gpu
                     else config['slurm']['cpu_partition'],
        'time': substep_def.default_time,
        'nodes': str(substep_def.default_nodes),
        'ntasks_per_node': str(substep_def.default_ntasks_per_node),
        'account': config['slurm']['account'],
    }

    # Apply config overrides
    step_overrides = config['steps'][step].get('slurm_overrides', {})
    substep_overrides = step_overrides.get(substep, {})
    for key, value in substep_overrides.items():
        slurm[key] = str(value)

    # Add GPU-specific parameters for GPU jobs
    if substep_def.requires_gpu:
        workers = config.get('slurm', {}).get('workers', {})
        if step in workers:
            slurm[f'{step.upper()}_WORKERS'] = str(workers[step])

        timeouts = config.get('slurm', {}).get('timeouts', {})
        slurm['STARTUP_TIMEOUT'] = str(timeouts.get('startup', 1800))
        slurm['REQUEST_TIMEOUT'] = str(timeouts.get('request', 1800))
        slurm['WORK_TIMEOUT'] = str(timeouts.get('work', 3600))

    return slurm


# =============================================================================
# JOB SUBMISSION
# =============================================================================

def submit_job(job_script: str, env_vars: dict, dependency: Optional[str] = None) -> Optional[str]:
    """Submit SLURM job with environment variables, return job ID"""
    cmd = ['sbatch']

    if dependency:
        cmd += ['--dependency', dependency]

    # Export all environment variables
    export_str = ','.join([f'{k}={v}' for k, v in env_vars.items()])
    cmd += ['--export', f'ALL,{export_str}']

    cmd.append(job_script)

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        # Parse "Submitted batch job 12345"
        parts = result.stdout.strip().split()
        if len(parts) >= 4:
            return parts[-1]
    else:
        print(f"  Error: {result.stderr.strip()}")

    return None


def submit_pipeline(config: dict, out_dir: Path) -> Dict[str, str]:
    """Submit all enabled substeps with dependencies"""
    print("\nSubmitting pipeline...\n")

    job_ids = {}
    prev_job_id = None

    for step in config['pipeline_sequence']:
        if not config['steps'][step].get('enabled', True):
            continue

        substeps_cfg = config['steps'][step].get('substeps', {})
        for substep_name, substep_cfg in substeps_cfg.items():
            # Check if substep is enabled
            if isinstance(substep_cfg, dict) and not substep_cfg.get('enabled', True):
                continue

            substep_def = get_substep_config(step, substep_name)
            env_vars = build_env_vars(config, step, substep_name, out_dir)

            dependency = f"afterany:{prev_job_id}" if prev_job_id else None
            job_id = submit_job(substep_def.job_script, env_vars, dependency)

            if job_id:
                full_name = f"{step}/{substep_name}"
                job_ids[full_name] = job_id
                prev_job_id = job_id
                dep_str = f" (depends on {prev_job_id[:-len(job_id)]}...)" if dependency else ""
                print(f"✓ {full_name:45} → {job_id}{dep_str}")
            else:
                print(f"✗ Failed to submit {step}/{substep_name}")
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
        capture_output=True, text=True
    )

    if result.returncode == 0 and result.stdout.strip():
        # Take first status (main job)
        return result.stdout.strip().split('\n')[0]

    return 'UNKNOWN'


def check_pipeline_status(job_ids: Dict[str, str]) -> Tuple[bool, List[str], List[str]]:
    """
    Check pipeline status.

    Returns:
        (all_done, failed_substeps, running_substeps)
    """
    failed = []
    running = []
    all_done = True

    for substep, job_id in job_ids.items():
        status = get_job_status(job_id)

        if status in ['FAILED', 'CANCELLED', 'TIMEOUT', 'NODE_FAIL', 'OUT_OF_MEMORY']:
            failed.append(substep)
            all_done = False
        elif status in ['PENDING', 'RUNNING']:
            running.append(substep)
            all_done = False

    return all_done, failed, running


def print_status(job_ids: Dict[str, str]):
    """Print pipeline status"""
    print("\nPipeline Status:\n" + "=" * 70)

    for substep, job_id in job_ids.items():
        status = get_job_status(job_id)

        symbols = {
            'COMPLETED': '✓',
            'FAILED': '✗',
            'CANCELLED': '⊗',
            'TIMEOUT': '⏱',
            'PENDING': '⏸',
            'RUNNING': '⏳',
        }
        symbol = symbols.get(status, '?')

        print(f"{symbol} {substep:40} {job_id:10} {status}")

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
        subprocess.run(['scancel', job_id], capture_output=True)


def get_all_substeps_ordered(config: dict) -> List[Tuple[str, str]]:
    """Get all enabled substeps in pipeline order as (step, substep) tuples"""
    result = []
    for step in config['pipeline_sequence']:
        if not config['steps'][step].get('enabled', True):
            continue

        substeps_cfg = config['steps'][step].get('substeps', {})
        for substep_name, substep_cfg in substeps_cfg.items():
            if isinstance(substep_cfg, dict) and not substep_cfg.get('enabled', True):
                continue
            result.append((step, substep_name))

    return result


def handle_force(config: dict, out_dir: Path):
    """Cancel all existing jobs and resubmit entire pipeline"""
    print("\n🗑️  Force mode: Cancelling existing jobs and resubmitting...\n")

    old_jobs = load_job_ids(out_dir)
    if old_jobs:
        cancel_jobs(list(old_jobs.values()))
        print(f"Cancelled {len(old_jobs)} existing job(s)")

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

    print(f"\n🔄 Resubmitting {len(failed)} failed job(s) and downstream dependencies...\n")

    # Find first failed substep in pipeline order
    all_substeps = get_all_substeps_ordered(config)
    substep_names = [f"{s}/{ss}" for s, ss in all_substeps]

    first_failed_idx = min(substep_names.index(f) for f in failed)

    # Cancel downstream jobs
    to_cancel = [old_jobs[name] for name in substep_names[first_failed_idx:] if name in old_jobs]
    if to_cancel:
        cancel_jobs(to_cancel)
        print(f"Cancelled {len(to_cancel)} downstream job(s)")

    # Get dependency from previous job
    prev_job_id = None
    if first_failed_idx > 0:
        prev_name = substep_names[first_failed_idx - 1]
        prev_job_id = old_jobs.get(prev_name)

    # Resubmit from failed substep onwards
    print("\nResubmitting:\n")
    new_jobs = {}

    for step, substep in all_substeps[first_failed_idx:]:
        substep_def = get_substep_config(step, substep)
        env_vars = build_env_vars(config, step, substep, out_dir)

        dependency = f"afterany:{prev_job_id}" if prev_job_id else None
        job_id = submit_job(substep_def.job_script, env_vars, dependency)

        if job_id:
            full_name = f"{step}/{substep}"
            new_jobs[full_name] = job_id
            old_jobs[full_name] = job_id  # Update history
            prev_job_id = job_id
            print(f"✓ {full_name:45} → {job_id}")
        else:
            print(f"✗ Failed to submit {step}/{substep}")
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

    print(f"\n⏭️  Skipping {len(failed)} failed job(s), submitting downstream...\n")

    # Find first failed substep
    all_substeps = get_all_substeps_ordered(config)
    substep_names = [f"{s}/{ss}" for s, ss in all_substeps]

    first_failed_idx = min(substep_names.index(f) for f in failed)

    # Cancel downstream jobs
    to_cancel = [old_jobs[name] for name in substep_names[first_failed_idx + 1:] if name in old_jobs]
    if to_cancel:
        cancel_jobs(to_cancel)
        print(f"Cancelled {len(to_cancel)} downstream job(s)")

    # Submit downstream WITHOUT dependency (assumes data exists)
    print("\nSubmitting downstream jobs:\n")
    new_jobs = {}

    for step, substep in all_substeps[first_failed_idx + 1:]:
        substep_def = get_substep_config(step, substep)
        env_vars = build_env_vars(config, step, substep, out_dir)

        job_id = submit_job(substep_def.job_script, env_vars, dependency=None)

        if job_id:
            full_name = f"{step}/{substep}"
            new_jobs[full_name] = job_id
            old_jobs[full_name] = job_id
            print(f"✓ {full_name:45} → {job_id}")
        else:
            print(f"✗ Failed to submit {step}/{substep}")
            cancel_jobs(list(new_jobs.values()))
            sys.exit(1)

    save_job_ids(out_dir, old_jobs)
    print(f"\n⚠️  Skipped failed: {', '.join(failed)}")
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
                print(f"❌ {len(failed)} job(s) failed. Use --resubmit-failed or --continue")
                sys.exit(1)
            elif running:
                print(f"⏳ {len(running)} job(s) still running")
                sys.exit(0)
            elif all_done:
                print("✅ All jobs completed. Use --force to rerun")
                sys.exit(0)

        # Fresh submission
        job_ids = submit_pipeline(config, args.out_dir)
        save_job_ids(args.out_dir, job_ids)

    print("\n✅ Done\n")


if __name__ == "__main__":
    main()
