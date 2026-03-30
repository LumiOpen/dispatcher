#!/usr/bin/env python3
"""Analyze completed/incomplete task volume within a time window from worker logs.

Tracks preemption events to report how much wall-clock time was spent actively
computing vs. waiting for resources / restarting after preemption.
"""

import argparse
import glob
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta


TIMESTAMP_FMT = "%Y-%m-%d %H:%M:%S,%f"
DATETIME_INPUT_FMTS = [
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M",
    "%Y-%m-%d",
]

PROCESSING_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) INFO \[ReasoningTranslationTask\] ID:(\d+) Processing sample$"
)
FINISHED_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) INFO \[ReasoningTranslationTask\] ID:(\d+) Finished processing sample$"
)
PREEMPTION_RE = re.compile(
    r"^\[(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2}:\d{2})\.\d+\] error: \*\*\* JOB \d+ ON .+ CANCELLED AT .+ DUE TO PREEMPTION \*\*\*$"
)

STANDALONE_ARRAY_ID = "4294967294"


@dataclass
class ActiveSession:
    start: datetime
    end: datetime

    @property
    def duration_seconds(self) -> float:
        return (self.end - self.start).total_seconds()


@dataclass
class WorkerInfo:
    log_file: str
    sessions: list[ActiveSession] = field(default_factory=list)
    preemption_times: list[datetime] = field(default_factory=list)
    first_task_event: datetime | None = None
    last_task_event: datetime | None = None


def parse_timestamp(ts_str: str) -> datetime:
    return datetime.strptime(ts_str, TIMESTAMP_FMT)


def parse_preemption_timestamp(date_str: str, time_str: str) -> datetime:
    return datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")


def parse_user_datetime(s: str) -> datetime:
    for fmt in DATETIME_INPUT_FMTS:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    raise argparse.ArgumentTypeError(
        f"Cannot parse datetime '{s}'. Use formats like: "
        "'2026-03-12 11:00:00', '2026-03-12 11:00', '2026-03-12T11:00', '2026-03-12'"
    )


def format_duration(seconds: float) -> str:
    if seconds < 0:
        return f"-{format_duration(-seconds)}"
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    if minutes < 60:
        return f"{minutes}m {secs:.1f}s"
    hours = int(minutes // 60)
    mins = minutes % 60
    return f"{hours}h {mins}m {secs:.1f}s"


def percentile(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        return 0.0
    k = (len(sorted_values) - 1) * (p / 100.0)
    f = int(k)
    c = f + 1
    if c >= len(sorted_values):
        return sorted_values[f]
    return sorted_values[f] + (k - f) * (sorted_values[c] - sorted_values[f])


def make_histogram(values: list[float], bin_count: int = 20) -> str:
    if not values:
        return "  (no data)\n"
    min_val = min(values)
    max_val = max(values)
    if max_val == min_val:
        return f"  All values are {format_duration(min_val)}\n"

    bin_width = (max_val - min_val) / bin_count
    bins = [0] * bin_count
    for v in values:
        idx = int((v - min_val) / bin_width)
        if idx >= bin_count:
            idx = bin_count - 1
        bins[idx] += 1

    max_count = max(bins)
    bar_max_width = 50
    lines = []
    for i, count in enumerate(bins):
        lo = min_val + i * bin_width
        hi = lo + bin_width
        bar_len = int((count / max_count) * bar_max_width) if max_count > 0 else 0
        bar = "\u2588" * bar_len
        lo_str = format_duration(lo).rjust(12)
        hi_str = format_duration(hi).rjust(12)
        lines.append(f"  {lo_str} - {hi_str} | {bar} {count}")
    return "\n".join(lines) + "\n"


def make_timeline_histogram(timestamps: list[datetime], start: datetime, end: datetime) -> str:
    """Show completions over time in hourly buckets."""
    if not timestamps:
        return "  (no data)\n"

    total_hours = (end - start).total_seconds() / 3600
    if total_hours <= 24:
        bucket_seconds = 3600
        bucket_label = "1h"
    elif total_hours <= 168:
        bucket_seconds = 3600 * 6
        bucket_label = "6h"
    else:
        bucket_seconds = 3600 * 24
        bucket_label = "1d"

    num_buckets = max(1, int((end - start).total_seconds() / bucket_seconds) + 1)
    buckets = [0] * num_buckets
    for ts in timestamps:
        idx = int((ts - start).total_seconds() / bucket_seconds)
        if 0 <= idx < num_buckets:
            buckets[idx] += 1

    max_count = max(buckets) if buckets else 0
    bar_max_width = 50
    lines = []
    for i, count in enumerate(buckets):
        bucket_start = start + timedelta(seconds=i * bucket_seconds)
        bucket_end = start + timedelta(seconds=(i + 1) * bucket_seconds)
        if bucket_end > end:
            bucket_end = end
        bar_len = int((count / max_count) * bar_max_width) if max_count > 0 else 0
        bar = "\u2588" * bar_len
        t_start = bucket_start.strftime("%m-%d %H:%M")
        t_end = bucket_end.strftime("%H:%M")
        lines.append(f"  {t_start}-{t_end} | {bar} {count}")
    return f"  (buckets: {bucket_label})\n" + "\n".join(lines) + "\n"


def discover_log_files(log_dir: str, job_id: str) -> tuple[list[str], bool]:
    """Find .err log files for the given job ID. Returns (file_list, is_array_job)."""
    standalone_path = os.path.join(log_dir, f"{job_id}_{STANDALONE_ARRAY_ID}.err")
    if os.path.isfile(standalone_path):
        return [standalone_path], False

    pattern = os.path.join(log_dir, f"{job_id}_*.err")
    matches = sorted(glob.glob(pattern))
    if not matches:
        standalone_plain = os.path.join(log_dir, f"{job_id}.err")
        if os.path.isfile(standalone_plain):
            return [standalone_plain], False
        return [], False

    return matches, len(matches) > 1


def scan_log_file(
    log_path: str,
) -> tuple[
    dict[str, list[datetime]],
    dict[str, list[datetime]],
    WorkerInfo,
    int,
    int,
]:
    """Parse a single log file.

    Returns (starts, finishes, worker_info, line_count, matched_count).
    worker_info contains active sessions and preemption data.
    """
    starts: dict[str, list[datetime]] = defaultdict(list)
    finishes: dict[str, list[datetime]] = defaultdict(list)
    worker = WorkerInfo(log_file=log_path)
    line_count = 0
    matched_count = 0

    session_active = False
    session_start: datetime | None = None
    last_task_ts: datetime | None = None

    with open(log_path, "r") as f:
        for line in f:
            line_count += 1
            line = line.rstrip("\n")

            m = PROCESSING_RE.match(line)
            if m:
                ts = parse_timestamp(m.group(1))
                task_id = m.group(2)
                starts[task_id].append(ts)
                matched_count += 1
                if worker.first_task_event is None:
                    worker.first_task_event = ts
                last_task_ts = ts
                if not session_active:
                    session_start = ts
                    session_active = True
                continue

            m = FINISHED_RE.match(line)
            if m:
                ts = parse_timestamp(m.group(1))
                task_id = m.group(2)
                finishes[task_id].append(ts)
                matched_count += 1
                if worker.first_task_event is None:
                    worker.first_task_event = ts
                last_task_ts = ts
                if not session_active:
                    session_start = ts
                    session_active = True
                continue

            m = PREEMPTION_RE.match(line)
            if m:
                preempt_ts = parse_preemption_timestamp(m.group(1), m.group(2))
                worker.preemption_times.append(preempt_ts)
                if session_active and session_start is not None:
                    worker.sessions.append(ActiveSession(start=session_start, end=preempt_ts))
                session_active = False
                session_start = None

    worker.last_task_event = last_task_ts
    if session_active and session_start is not None and last_task_ts is not None:
        worker.sessions.append(ActiveSession(start=session_start, end=last_task_ts))

    return starts, finishes, worker, line_count, matched_count


def clip_session(session: ActiveSession, win_start: datetime, win_end: datetime) -> float:
    """Return seconds of the session that overlap with the window."""
    s = max(session.start, win_start)
    e = min(session.end, win_end)
    return max(0.0, (e - s).total_seconds())


def merge_dicts(
    target: dict[str, list[datetime]], source: dict[str, list[datetime]]
) -> None:
    for k, v in source.items():
        target[k].extend(v)


def print_worker_uptime(
    workers: list[WorkerInfo],
    window_start: datetime,
    window_end: datetime,
) -> None:
    """Print per-worker and aggregate computing vs queued statistics."""
    window_seconds = (window_end - window_start).total_seconds()
    window_hours = window_seconds / 3600
    num_workers = len(workers)
    total_budget_seconds = window_seconds * num_workers
    total_budget_hours = total_budget_seconds / 3600

    worker_stats: list[tuple[str, float, int, int]] = []

    for w in workers:
        basename = os.path.basename(w.log_file)
        up_sec = sum(clip_session(s, window_start, window_end) for s in w.sessions)
        preemptions_in_window = sum(
            1 for pt in w.preemption_times if window_start <= pt <= window_end
        )
        worker_stats.append((basename, up_sec, preemptions_in_window, len(w.sessions)))

    total_up = sum(u for _, u, _, _ in worker_stats)
    total_down = total_budget_seconds - total_up
    total_preemptions = sum(p for _, _, p, _ in worker_stats)
    up_pct = (total_up / total_budget_seconds * 100) if total_budget_seconds > 0 else 0
    down_pct = 100 - up_pct

    print(f"\n--- COMPUTING TIME vs QUEUE TIME (preemptible job) ---")
    print(f"  Window:                {format_duration(window_seconds)}  x  {num_workers} workers  "
          f"=  {total_budget_hours:.1f} worker-hours")
    print(f"  Computing (up):        {total_up / 3600:.1f} worker-hours  ({up_pct:.1f}%)")
    print(f"  In queue (down):       {total_down / 3600:.1f} worker-hours  ({down_pct:.1f}%)")
    print(f"  Preemptions:           {total_preemptions}")

    if num_workers > 0:
        print(f"\n  Per-worker breakdown:")
        header = (
            f"    {'Worker':<28s} {'Up':>12s} {'Down':>12s} "
            f"{'Up%':>6s} {'Preempts':>8s} {'Sessions':>8s}"
        )
        print(header)
        print(f"    {'-' * len(header.strip())}")
        for basename, up_sec, n_preempt, n_sessions in sorted(worker_stats):
            down_sec = window_seconds - up_sec
            pct = (up_sec / window_seconds * 100) if window_seconds > 0 else 0
            print(
                f"    {basename:<28s} "
                f"{format_duration(up_sec):>12s} "
                f"{format_duration(down_sec):>12s} "
                f"{pct:>5.1f}% "
                f"{n_preempt:>8d} "
                f"{n_sessions:>8d}"
            )

    # Queue wait details: time from preemption to when the worker resumes
    all_gaps: list[tuple[str, datetime, datetime, float]] = []
    for w in workers:
        basename = os.path.basename(w.log_file)
        for pt in w.preemption_times:
            if not (window_start <= pt <= window_end):
                continue
            resume_ts = None
            for s in w.sessions:
                if s.start > pt:
                    resume_ts = s.start
                    break
            if resume_ts is not None:
                gap_sec = (resume_ts - pt).total_seconds()
                if gap_sec > 0:
                    all_gaps.append((basename, pt, resume_ts, gap_sec))

    if all_gaps:
        gap_durations = [g for _, _, _, g in all_gaps]
        gap_sorted = sorted(gap_durations)
        mean_gap = sum(gap_sorted) / len(gap_sorted)

        print(f"\n  Queue wait per preemption (preempted -> resumed computing):")
        print(f"    Count:  {len(all_gaps)}")
        print(f"    Min:    {format_duration(gap_sorted[0])}")
        print(f"    Max:    {format_duration(gap_sorted[-1])}")
        print(f"    Mean:   {format_duration(mean_gap)}")
        print(f"    Median: {format_duration(percentile(gap_sorted, 50))}")

        if len(all_gaps) <= 30:
            print(f"\n    All gaps:")
            for basename, pt, resume, gap_sec in sorted(all_gaps, key=lambda x: x[1]):
                print(
                    f"      {basename:<24s} "
                    f"{pt.strftime('%m-%d %H:%M:%S')} -> "
                    f"{resume.strftime('%m-%d %H:%M:%S')}  "
                    f"({format_duration(gap_sec)})"
                )


def infer_window(workers: list[WorkerInfo]) -> tuple[datetime, datetime]:
    """Derive the time span from the earliest to latest event across all workers."""
    earliest: datetime | None = None
    latest: datetime | None = None
    for w in workers:
        for s in w.sessions:
            if earliest is None or s.start < earliest:
                earliest = s.start
            if latest is None or s.end > latest:
                latest = s.end
        for pt in w.preemption_times:
            if earliest is None or pt < earliest:
                earliest = pt
            if latest is None or pt > latest:
                latest = pt
    assert earliest is not None and latest is not None
    return earliest, latest


def analyze(
    log_dir: str,
    job_id: str,
    window_start: datetime | None,
    window_end: datetime | None,
) -> None:
    log_files, is_array = discover_log_files(log_dir, job_id)
    if not log_files:
        print(f"ERROR: No log files found for job ID {job_id} in {log_dir}", file=sys.stderr)
        sys.exit(1)

    all_starts: dict[str, list[datetime]] = defaultdict(list)
    all_finishes: dict[str, list[datetime]] = defaultdict(list)
    all_workers: list[WorkerInfo] = []
    total_lines = 0
    total_matched = 0

    for lf in log_files:
        starts, finishes, worker, lc, mc = scan_log_file(lf)
        merge_dicts(all_starts, starts)
        merge_dicts(all_finishes, finishes)
        all_workers.append(worker)
        total_lines += lc
        total_matched += mc

    if window_start is None or window_end is None:
        inferred_start, inferred_end = infer_window(all_workers)
        if window_start is None:
            window_start = inferred_start
        if window_end is None:
            window_end = inferred_end

    all_task_ids = set(all_starts.keys()) | set(all_finishes.keys())

    completed_in_window: list[tuple[str, float, datetime]] = []
    incomplete_in_window: list[tuple[str, datetime]] = []
    completed_outside_window: list[tuple[str, float, datetime]] = []
    anomalies: list[str] = []

    for task_id in sorted(all_task_ids, key=int):
        s_list = all_starts.get(task_id, [])
        f_list = all_finishes.get(task_id, [])

        if not s_list and f_list:
            anomalies.append(
                f"  ID:{task_id} - Finished without any 'Processing sample' start "
                f"(finish at {f_list[0].strftime(TIMESTAMP_FMT)})"
            )
            continue

        if s_list and not f_list:
            last_start = max(s_list)
            if window_start <= last_start <= window_end:
                incomplete_in_window.append((task_id, last_start))
            continue

        if len(f_list) > 1:
            anomalies.append(
                f"  ID:{task_id} - Multiple 'Finished' logs ({len(f_list)} times)"
            )

        last_start = max(s_list)
        finish = max(f_list)
        duration = (finish - last_start).total_seconds()

        if duration < 0:
            anomalies.append(
                f"  ID:{task_id} - Negative duration ({format_duration(duration)})"
            )
            continue

        if window_start <= finish <= window_end:
            completed_in_window.append((task_id, duration, finish))
        else:
            completed_outside_window.append((task_id, duration, finish))

    durations = [d for _, d, _ in completed_in_window]
    durations_sorted = sorted(durations)
    finish_times = [ft for _, _, ft in completed_in_window]

    window_hours = (window_end - window_start).total_seconds() / 3600

    # --- Output ---
    print("=" * 80)
    print("TASK VOLUME ANALYSIS")
    print(f"Job ID:       {job_id} ({'array job' if is_array else 'standalone job'})")
    print(f"Log files:    {len(log_files)}")
    if is_array:
        basenames = [os.path.basename(f) for f in log_files]
        print(f"              {', '.join(basenames)}")
    print(f"Time window:  {window_start.strftime('%Y-%m-%d %H:%M:%S')} -> "
          f"{window_end.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"              ({window_hours:.1f} hours)")
    print(f"Total log lines scanned: {total_lines:,}")
    print(f"Matched task events:     {total_matched:,}")
    print("=" * 80)

    # --- Uptime / preemption section ---
    print_worker_uptime(all_workers, window_start, window_end)

    print(f"\n--- TASK COUNTS (within window) ---")
    print(f"  Completed tasks:   {len(completed_in_window):,}")
    print(f"  Incomplete tasks:  {len(incomplete_in_window):,}  (started but not finished)")
    print(f"  Total tasks seen in window: {len(completed_in_window) + len(incomplete_in_window):,}")
    if window_hours > 0:
        rate = len(completed_in_window) / window_hours
        print(f"  Throughput (wall-clock):      {rate:.1f} tasks/hour")
        total_up_sec = sum(
            clip_session(s, window_start, window_end)
            for w in all_workers
            for s in w.sessions
        )
        total_up_hours = total_up_sec / 3600
        if total_up_hours > 0:
            up_rate = len(completed_in_window) / total_up_hours
            print(f"  Throughput (while computing): {up_rate:.1f} tasks/worker-hour")

    print(f"\n--- TASKS OUTSIDE WINDOW ---")
    print(f"  Completed outside window: {len(completed_outside_window):,}")
    print(f"  Total unique task IDs across all time: {len(all_task_ids):,}")

    if durations_sorted:
        mean_d = sum(durations_sorted) / len(durations_sorted)
        variance = sum((x - mean_d) ** 2 for x in durations_sorted) / len(durations_sorted)
        std_d = variance ** 0.5

        print(f"\n--- DURATION STATISTICS (completed tasks in window) ---")
        print(f"  Count:   {len(durations_sorted):,}")
        print(f"  Min:     {format_duration(durations_sorted[0])}")
        print(f"  Max:     {format_duration(durations_sorted[-1])}")
        print(f"  Mean:    {format_duration(mean_d)}")
        print(f"  Std Dev: {format_duration(std_d)}")
        print(f"  Median:  {format_duration(percentile(durations_sorted, 50))}")
        print(f"  P5:      {format_duration(percentile(durations_sorted, 5))}")
        print(f"  P10:     {format_duration(percentile(durations_sorted, 10))}")
        print(f"  P25:     {format_duration(percentile(durations_sorted, 25))}")
        print(f"  P75:     {format_duration(percentile(durations_sorted, 75))}")
        print(f"  P90:     {format_duration(percentile(durations_sorted, 90))}")
        print(f"  P95:     {format_duration(percentile(durations_sorted, 95))}")
        print(f"  P99:     {format_duration(percentile(durations_sorted, 99))}")

        print(f"\n--- DURATION DISTRIBUTION (histogram) ---")
        print(make_histogram(durations))

    if finish_times:
        print(f"--- COMPLETIONS OVER TIME ---")
        print(make_timeline_histogram(finish_times, window_start, window_end))

    if incomplete_in_window:
        sorted_incomplete = sorted(incomplete_in_window, key=lambda x: x[1])
        print(f"--- INCOMPLETE TASKS ({len(incomplete_in_window):,}) ---")
        if len(incomplete_in_window) <= 20:
            for task_id, start_ts in sorted_incomplete:
                elapsed = (window_end - start_ts).total_seconds()
                print(f"  ID:{task_id:>10s}  started {start_ts.strftime('%Y-%m-%d %H:%M:%S')}  "
                      f"(pending for {format_duration(elapsed)})")
        else:
            earliest = sorted_incomplete[0]
            latest = sorted_incomplete[-1]
            print(f"  Earliest start: ID:{earliest[0]} at {earliest[1].strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"  Latest start:   ID:{latest[0]} at {latest[1].strftime('%Y-%m-%d %H:%M:%S')}")

    if anomalies:
        print(f"\n--- ANOMALIES ({len(anomalies)}) ---")
        for a in anomalies:
            print(a)

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze completed/incomplete task volume within a time window."
    )
    parser.add_argument("job_id", help="SLURM job ID (e.g. 6129)")
    parser.add_argument(
        "--log-dir",
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs"),
        help="Directory containing log files (default: ../logs relative to this script)",
    )
    parser.add_argument(
        "--start",
        type=parse_user_datetime,
        default=None,
        help="Window start time (e.g. '2026-03-12 11:00'). Default: earliest event in logs.",
    )
    parser.add_argument(
        "--end",
        type=parse_user_datetime,
        default=None,
        help="Window end time (e.g. '2026-03-13 11:00'). Default: latest event in logs.",
    )
    args = parser.parse_args()

    if args.start is not None and args.end is not None and args.start >= args.end:
        print("ERROR: --start must be before --end", file=sys.stderr)
        sys.exit(1)

    analyze(args.log_dir, args.job_id, args.start, args.end)


if __name__ == "__main__":
    main()
