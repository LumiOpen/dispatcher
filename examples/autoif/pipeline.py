#!/usr/bin/env python3
"""
AutoIF Pipeline - Enhanced config-driven orchestrator with template-based job generation

Reads task config from config.yaml and SLURM config from slurm.yaml.
Automatically detects execution mode (interactive vs SLURM) based on vllm_server presence.
Generates job scripts from templates and tracks execution status.
"""

import argparse
import json
import os
import subprocess
import sys
import shutil
import yaml
import jinja2
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime


# =============================================================================
# CONFIG LOADING
# =============================================================================

def load_yaml_file(path: Path) -> dict:
    """Load YAML configuration file"""
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def detect_execution_mode(config: dict) -> str:
    """
    Detect execution mode based on vllm_server presence.

    Returns:
        'interactive' if vllm_server is configured
        'sbatch' otherwise
    """
    return 'interactive' if config.get('vllm_server') else 'sbatch'


# =============================================================================
# STATUS TRACKING
# =============================================================================

def load_status(out_dir: Path) -> dict:
    """Load unified status tracking"""
    status_file = out_dir / 'status.json'
    if status_file.exists():
        with open(status_file) as f:
            return json.load(f)
    return {
        'sbatch': {},
        'interactive': {}
    }


def save_status(out_dir: Path, status: dict):
    """Save unified status"""
    status['last_updated'] = datetime.now().isoformat()
    status_file = out_dir / 'status.json'
    with open(status_file, 'w') as f:
        json.dump(status, f, indent=2)


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
    Check pipeline status for SLURM jobs.

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


def print_status(status: dict, execution_mode: str):
    """Print pipeline status"""
    print("\nPipeline Status:\n" + "=" * 70)

    mode_key = execution_mode
    if mode_key in status and status[mode_key]:
        for job, job_info in status[mode_key].items():
            if execution_mode == 'sbatch':
                job_id = job_info.get('job_id', 'N/A')
                job_status = get_job_status(job_id) if job_id != 'N/A' else 'UNKNOWN'
                print(f"{job:40} {job_id:10} {job_status}")
            else:  # interactive
                job_status = job_info.get('status', 'unknown')
                print(f"{job:40} {job_status:15}")

    print("=" * 70 + "\n")


# =============================================================================
# JOB SCRIPT GENERATION
# =============================================================================

def resolve_template_variables(value, config: dict, out_dir: Path):
    """
    Resolve template variables and paths.

    - Converts relative file paths to absolute (relative to out_dir)
    - Keeps absolute paths and HF dataset paths as-is
    """
    if not isinstance(value, str):
        return value

    # Don't modify absolute paths or HF paths
    if value.startswith('/') or ('/' in value and len(value) > 50):
        return value

    # Check if it looks like a file path
    if any(value.endswith(ext) for ext in ['.jsonl', '.txt', '.json', '.csv', '.log']) or \
       'dataset' in str(value).lower() or 'dir' in str(value).lower():
        return str(out_dir / value)

    return value


def merge_configs(global_config: dict, job_config: dict, out_dir: Path) -> dict:
    """
    Merge global and job-level configurations.
    Job-level configs override global configs.
    Resolves all file paths to absolute paths.
    """
    merged = {}

    # Start with global config (excluding reserved keys)
    reserved_keys = {'experiment', 'pipeline', 'jobs', 'vllm_server', 'environment_setup'}
    for key, value in global_config.items():
        if key not in reserved_keys:
            merged[key] = resolve_template_variables(value, global_config, out_dir)

    # Override with job config
    for key, value in job_config.items():
        merged[key] = resolve_template_variables(value, job_config, out_dir)

    return merged


def generate_job_script(
    job_name: str,
    job_config: dict,
    global_config: dict,
    slurm_config: dict,
    execution_mode: str,
    out_dir: Path
) -> Path:
    """Generate job script from template using Jinja2"""

    job_type = job_config.get('type', 'cpu_script')

    # Determine template
    custom_template = job_config.get('template')
    if custom_template:
        template_name = custom_template
    else:
        # Default templates
        if job_type == 'dispatcher_task':
            template_name = (
                'dispatcher_local_job.sh.j2' if execution_mode == 'interactive'
                else 'dispatcher_job.sh.j2'
            )
        else:  # cpu_script
            template_name = 'cpu_job.sh.j2'

    # Merge configs
    merged_config = merge_configs(global_config, job_config, out_dir)

    # Resolve SLURM settings (global defaults + job type defaults + job overrides)
    slurm_settings = {}
    if execution_mode == 'sbatch':
        slurm_settings = {**slurm_config.get('slurm', {})}
        slurm_settings.update(slurm_config.get('job_type_defaults', {}).get(job_type, {}))
        slurm_settings.update(slurm_config.get('jobs', {}).get(job_name, {}))

    # Prepare template context
    context = {
        'job_name': job_name,
        'execution_mode': execution_mode,
        'environment_setup': global_config.get('environment_setup'),
        'slurm': slurm_settings,
        'vllm_server': global_config.get('vllm_server', {}),
        **merged_config
    }

    # Render template
    template_loader = jinja2.FileSystemLoader('execution/job_templates')
    env = jinja2.Environment(
        loader=template_loader,
        undefined=jinja2.StrictUndefined
    )

    try:
        template = env.get_template(template_name)
        script_content = template.render(**context)
    except jinja2.exceptions.TemplateError as e:
        print(f"ERROR: Failed to render template for {job_name}: {e}")
        sys.exit(1)

    # Write to generated_jobs/
    generated_dir = out_dir / 'generated_jobs'
    generated_dir.mkdir(exist_ok=True)
    script_path = generated_dir / f"{job_name}.sh"
    script_path.write_text(script_content)
    script_path.chmod(0o755)

    return script_path


# =============================================================================
# JOB EXECUTION
# =============================================================================

def submit_job(job_script: str, dependency: Optional[str] = None) -> Optional[str]:
    """Submit SLURM job, return job ID"""
    cmd = ['sbatch']

    if dependency:
        cmd += ['--dependency', dependency]

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


def run_job_interactive(
    job_script: Path,
    job_name: str,
    out_dir: Path,
    status: dict
) -> bool:
    """Run job in interactive mode with logging"""

    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)

    stdout_log = log_dir / f"local_{job_name}.out"
    stderr_log = log_dir / f"local_{job_name}.err"

    # Update status
    status['interactive'][job_name] = {
        'status': 'running',
        'started': datetime.now().isoformat(),
        'log_stdout': str(stdout_log),
        'log_stderr': str(stderr_log)
    }
    save_status(out_dir, status)

    print(f"  Logs: {stdout_log} / {stderr_log}")

    # Run job
    with open(stdout_log, 'w') as fout, open(stderr_log, 'w') as ferr:
        result = subprocess.run(
            ['bash', str(job_script)],
            stdout=fout,
            stderr=ferr,
            cwd=Path.cwd()
        )

    # Update status
    success = result.returncode == 0
    status['interactive'][job_name].update({
        'status': 'completed' if success else 'failed',
        'finished': datetime.now().isoformat(),
        'exit_code': result.returncode
    })
    save_status(out_dir, status)

    return success


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


def execute_jobs(
    config: dict,
    slurm_config: dict,
    out_dir: Path,
    job_names: List[str],
    execution_mode: str,
    prev_job_id: Optional[str] = None
) -> dict:
    """Execute jobs in appropriate mode"""

    status = load_status(out_dir)
    status['execution_mode'] = execution_mode

    mode_key = execution_mode

    for job_name in job_names:
        job_config = config['jobs'].get(job_name, {})

        # Check for manual override
        custom_script = Path('custom_jobs') / f"{job_name}.sh"
        if custom_script.exists():
            print(f"  Using custom script: {custom_script}")
            job_script = custom_script
        else:
            # Generate from template
            job_script = generate_job_script(
                job_name,
                job_config,
                config,
                slurm_config,
                execution_mode,
                out_dir
            )

        # Execute
        if execution_mode == 'sbatch':
            dependency = f"afterany:{prev_job_id}" if prev_job_id else None
            job_id = submit_job(str(job_script), dependency)

            if job_id:
                status['sbatch'][job_name] = {
                    'job_id': job_id,
                    'status': 'PENDING',
                    'started': datetime.now().isoformat()
                }
                save_status(out_dir, status)

                dep_str = f" (depends on {prev_job_id})" if prev_job_id else ""
                print(f"  {job_name:45} → {job_id}{dep_str}")
                prev_job_id = job_id
            else:
                print(f"  Failed to submit {job_name}")
                # Cancel already submitted jobs
                cancel_jobs([info['job_id'] for info in status['sbatch'].values()])
                sys.exit(1)

        else:  # interactive
            print(f"\n  Running {job_name}...")
            success = run_job_interactive(job_script, job_name, out_dir, status)

            if not success:
                print(f"  ERROR: Job {job_name} failed")
                print(f"  Check logs: logs/local_{job_name}.err")
                sys.exit(1)

            print(f"  ✓ {job_name} completed successfully")

    return status


def submit_pipeline(config: dict, slurm_config: dict, out_dir: Path, execution_mode: str) -> dict:
    """Submit all enabled jobs"""
    pipeline_sequence = get_pipeline_sequence(config)
    job_names = [name for name, enabled in pipeline_sequence if enabled]

    if execution_mode == 'sbatch':
        print("\nSubmitting jobs to SLURM...\n")
    else:
        vllm = config['vllm_server']
        print(f"\nRunning jobs in interactive mode...")
        print(f"vLLM Server: {vllm['host']}:{vllm['port']}\n")

    return execute_jobs(config, slurm_config, out_dir, job_names, execution_mode)


# =============================================================================
# FAILURE RECOVERY
# =============================================================================

def cancel_jobs(job_ids: List[str]):
    """Cancel SLURM jobs"""
    for job_id in job_ids:
        subprocess.run(['scancel', job_id], stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def get_all_jobs_ordered(config: dict) -> List[str]:
    """Get all enabled jobs in pipeline order"""
    pipeline_sequence = get_pipeline_sequence(config)
    return [job_name for job_name, enabled in pipeline_sequence if enabled]


def handle_force(config: dict, slurm_config: dict, out_dir: Path, execution_mode: str):
    """Cancel all existing jobs and rerun entire pipeline"""
    print("\n  Force mode: Rerunning entire pipeline...\n")

    status = load_status(out_dir)

    # Cancel SLURM jobs if in sbatch mode
    if execution_mode == 'sbatch' and 'sbatch' in status:
        old_jobs = [info['job_id'] for info in status['sbatch'].values()]
        if old_jobs:
            cancel_jobs(old_jobs)

    # Clear status
    status = {'sbatch': {}, 'interactive': {}}
    save_status(out_dir, status)

    # Rerun pipeline
    submit_pipeline(config, slurm_config, out_dir, execution_mode)


def handle_rerun_failed(config: dict, slurm_config: dict, out_dir: Path, execution_mode: str):
    """Rerun from first failed job onwards"""
    status = load_status(out_dir)

    if execution_mode == 'interactive':
        print("\nInteractive mode: rerunning entire pipeline")
        submit_pipeline(config, slurm_config, out_dir, execution_mode)
        return

    # SLURM mode
    old_jobs = status.get('sbatch', {})
    if not old_jobs:
        print("No previous jobs found")
        sys.exit(1)

    job_ids = {name: info['job_id'] for name, info in old_jobs.items()}
    all_done, failed, running = check_pipeline_status(job_ids)

    if not failed:
        print("\nNo failed jobs to rerun")
        if running:
            print(f"{len(running)} job(s) still running")
        elif all_done:
            print("All jobs completed successfully")
        sys.exit(0)

    # Find first failed job in pipeline order
    all_jobs = get_all_jobs_ordered(config)
    first_failed_idx = min(all_jobs.index(f) for f in failed)

    # Cancel downstream jobs
    to_cancel = [old_jobs[name]['job_id'] for name in all_jobs[first_failed_idx:] if name in running]
    if to_cancel:
        cancel_jobs(to_cancel)
        print(f"Cancelled {len(to_cancel)} downstream job(s)")

    # Get dependency from previous job
    prev_job_id = None
    if first_failed_idx > 0:
        prev_name = all_jobs[first_failed_idx - 1]
        prev_job_id = old_jobs.get(prev_name, {}).get('job_id')

    # Rerun from failed job onwards
    print("\nRerunning:\n")
    jobs_to_rerun = all_jobs[first_failed_idx:]

    execute_jobs(
        config,
        slurm_config,
        out_dir,
        jobs_to_rerun,
        execution_mode,
        prev_job_id=prev_job_id
    )


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='AutoIF Pipeline - Enhanced config-driven orchestrator',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--config', type=Path, required=True,
                       help='Path to task config.yaml')
    parser.add_argument('--slurm-config', type=Path, default=Path('configs/slurm.default.yaml'),
                       help='Path to slurm.yaml (default: configs/slurm.default.yaml)')
    parser.add_argument('--out-dir', type=Path, default=Path('data'),
                       help='Output directory (default: data)')
    parser.add_argument('--force', action='store_true',
                       help='Cancel all jobs and rerun entire pipeline')
    parser.add_argument('--rerun-failed', action='store_true',
                       help='Re-run failed jobs and all downstream dependencies')
    parser.add_argument('--status', action='store_true',
                       help='Show pipeline status and exit')

    args = parser.parse_args()

    # Load configs
    config = load_yaml_file(args.config)
    if not config:
        print(f"ERROR: Failed to load config from {args.config}")
        sys.exit(1)

    slurm_config = load_yaml_file(args.slurm_config)

    # Create output directory
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Detect execution mode
    execution_mode = detect_execution_mode(config)

    print(f"\nAutoIF Pipeline")
    print(f"Experiment: {config.get('experiment', {}).get('name', 'N/A')}")
    print(f"Execution Mode: {execution_mode.upper()}")
    print(f"Output: {args.out_dir}")

    # Handle different modes
    if args.status:
        status = load_status(args.out_dir)
        print_status(status, execution_mode)
        sys.exit(0)
    elif args.force:
        handle_force(config, slurm_config, args.out_dir, execution_mode)
    elif args.rerun_failed:
        handle_rerun_failed(config, slurm_config, args.out_dir, execution_mode)
    else:
        # Check for existing jobs
        status = load_status(args.out_dir)

        if execution_mode == 'sbatch' and status.get('sbatch'):
            print_status(status, execution_mode)
            job_ids = {name: info['job_id'] for name, info in status['sbatch'].items()}
            all_done, failed, running = check_pipeline_status(job_ids)

            if failed:
                print(f" {len(failed)} job(s) failed. Use --rerun-failed or --force")
                sys.exit(1)
            elif running:
                print(f" {len(running)} job(s) still running. Use --status to check")
                sys.exit(0)
            elif all_done:
                print(" All jobs completed. Use --force to rerun")
                sys.exit(0)

        # Fresh submission
        submit_pipeline(config, slurm_config, args.out_dir, execution_mode)


if __name__ == "__main__":
    main()
