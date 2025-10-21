"""SLURM utility functions for job management"""

import subprocess
import re
from typing import Optional, Dict, List
from dataclasses import dataclass
from enum import Enum


class JobStatus(Enum):
    """SLURM job status states"""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    TIMEOUT = "TIMEOUT"
    NODE_FAIL = "NODE_FAIL"
    OUT_OF_MEMORY = "OUT_OF_MEMORY"
    UNKNOWN = "UNKNOWN"
    NOT_FOUND = "NOT_FOUND"


@dataclass
class SlurmJobInfo:
    """Information about a SLURM job"""
    job_id: str
    status: JobStatus
    exit_code: Optional[int]
    start_time: Optional[str]
    end_time: Optional[str]
    state: str  # Raw SLURM state string


def get_job_status(job_id: str) -> SlurmJobInfo:
    """
    Query SLURM for job status using sacct.

    Args:
        job_id: SLURM job ID

    Returns:
        SlurmJobInfo with current job status
    """
    try:
        # Query sacct with specific format
        # Format: JobID|State|ExitCode|Start|End
        result = subprocess.run(
            [
                "sacct",
                "-j", job_id,
                "--format=JobID,State,ExitCode,Start,End",
                "--parsable2",
                "--noheader"
            ],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode != 0:
            # sacct failed - job might not exist or sacct not available
            return SlurmJobInfo(
                job_id=job_id,
                status=JobStatus.NOT_FOUND,
                exit_code=None,
                start_time=None,
                end_time=None,
                state="NOT_FOUND"
            )

        # Parse sacct output
        lines = result.stdout.strip().split('\n')
        if not lines or not lines[0]:
            return SlurmJobInfo(
                job_id=job_id,
                status=JobStatus.NOT_FOUND,
                exit_code=None,
                start_time=None,
                end_time=None,
                state="NOT_FOUND"
            )

        # Get the main job line (not .batch or .extern)
        main_job_line = None
        for line in lines:
            if line and '|' in line:
                parts = line.split('|')
                # Skip .batch, .extern, etc. - we want the main job ID
                if parts[0] == job_id or parts[0] == f"{job_id}.batch":
                    main_job_line = line
                    if parts[0] == job_id:
                        break  # Prefer exact match

        if not main_job_line:
            main_job_line = lines[0]

        parts = main_job_line.split('|')
        if len(parts) < 5:
            return SlurmJobInfo(
                job_id=job_id,
                status=JobStatus.UNKNOWN,
                exit_code=None,
                start_time=None,
                end_time=None,
                state="UNKNOWN"
            )

        job_id_part, state, exit_code_str, start_time, end_time = parts[:5]

        # Parse exit code (format: "0:0" where first is exit code, second is signal)
        exit_code = None
        if exit_code_str and ':' in exit_code_str:
            try:
                exit_code = int(exit_code_str.split(':')[0])
            except (ValueError, IndexError):
                pass

        # Map SLURM state to our JobStatus enum
        status = parse_slurm_state(state, exit_code)

        return SlurmJobInfo(
            job_id=job_id,
            status=status,
            exit_code=exit_code,
            start_time=start_time if start_time != "Unknown" else None,
            end_time=end_time if end_time != "Unknown" else None,
            state=state
        )

    except subprocess.TimeoutExpired:
        return SlurmJobInfo(
            job_id=job_id,
            status=JobStatus.UNKNOWN,
            exit_code=None,
            start_time=None,
            end_time=None,
            state="TIMEOUT_QUERY"
        )
    except Exception as e:
        return SlurmJobInfo(
            job_id=job_id,
            status=JobStatus.UNKNOWN,
            exit_code=None,
            start_time=None,
            end_time=None,
            state=f"ERROR: {str(e)}"
        )


def parse_slurm_state(state: str, exit_code: Optional[int]) -> JobStatus:
    """
    Parse SLURM state string to JobStatus enum.

    SLURM states: PENDING, RUNNING, COMPLETED, FAILED, CANCELLED, TIMEOUT,
                  NODE_FAIL, OUT_OF_MEMORY, etc.
    """
    state_upper = state.upper()

    if state_upper in ["PENDING", "PD"]:
        return JobStatus.PENDING
    elif state_upper in ["RUNNING", "R"]:
        return JobStatus.RUNNING
    elif state_upper in ["COMPLETED", "CD"]:
        # Check exit code to distinguish success from failure
        if exit_code is not None and exit_code != 0:
            return JobStatus.FAILED
        return JobStatus.COMPLETED
    elif state_upper in ["FAILED", "F"]:
        return JobStatus.FAILED
    elif state_upper in ["CANCELLED", "CA", "CANCELED"]:
        return JobStatus.CANCELLED
    elif state_upper in ["TIMEOUT", "TO"]:
        return JobStatus.TIMEOUT
    elif state_upper in ["NODE_FAIL", "NF"]:
        return JobStatus.NODE_FAIL
    elif state_upper in ["OUT_OF_MEMORY", "OOM"]:
        return JobStatus.OUT_OF_MEMORY
    else:
        return JobStatus.UNKNOWN


def cancel_job(job_id: str) -> bool:
    """
    Cancel a SLURM job.

    Args:
        job_id: SLURM job ID to cancel

    Returns:
        True if cancellation successful, False otherwise
    """
    try:
        result = subprocess.run(
            ["scancel", job_id],
            capture_output=True,
            text=True,
            timeout=10
        )
        return result.returncode == 0
    except Exception:
        return False


def submit_job(
    script_path: str,
    dependency: Optional[str] = None,
    env_vars: Optional[Dict[str, str]] = None
) -> Optional[str]:
    """
    Submit a SLURM job script.

    Args:
        script_path: Path to SLURM job script
        dependency: Optional dependency string (e.g., "afterany:12345")
        env_vars: Optional environment variables to export

    Returns:
        Job ID if successful, None otherwise
    """
    try:
        cmd = ["sbatch", "--parsable"]

        if dependency:
            cmd.extend(["--dependency", dependency])

        # Add environment variables
        if env_vars:
            for key, value in env_vars.items():
                cmd.extend(["--export", f"{key}={value}"])

        cmd.append(script_path)

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode != 0:
            print(f"Error submitting job: {result.stderr}")
            return None

        # Parse job ID from output (sbatch --parsable returns just the job ID)
        job_id = result.stdout.strip()

        # Validate job ID is numeric
        if job_id and job_id.isdigit():
            return job_id

        # Try to extract from standard sbatch output if --parsable failed
        match = re.search(r'Submitted batch job (\d+)', result.stdout)
        if match:
            return match.group(1)

        return None

    except Exception as e:
        print(f"Error submitting job: {e}")
        return None


def get_pending_or_running_jobs(job_ids: List[str]) -> List[str]:
    """
    Check which jobs from the list are still pending or running.

    Args:
        job_ids: List of SLURM job IDs to check

    Returns:
        List of job IDs that are PENDING or RUNNING
    """
    active_jobs = []

    for job_id in job_ids:
        info = get_job_status(job_id)
        if info.status in [JobStatus.PENDING, JobStatus.RUNNING]:
            active_jobs.append(job_id)

    return active_jobs
