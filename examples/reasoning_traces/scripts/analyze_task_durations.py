#!/usr/bin/env python3
"""Analyze translation worker task execution durations from log files."""

import argparse
import re
import sys
from collections import defaultdict
from datetime import datetime


TIMESTAMP_FMT = "%Y-%m-%d %H:%M:%S,%f"

PROCESSING_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) INFO \[ReasoningTranslationTask\] ID:(\d+) Processing sample$"
)
FINISHED_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) INFO \[ReasoningTranslationTask\] ID:(\d+) Finished processing sample$"
)


def parse_timestamp(ts_str: str) -> datetime:
    return datetime.strptime(ts_str, TIMESTAMP_FMT)


def format_duration(seconds: float) -> str:
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


def make_histogram(durations_sec: list[float], bin_count: int = 20) -> str:
    if not durations_sec:
        return "  (no data)\n"
    min_val = min(durations_sec)
    max_val = max(durations_sec)
    if max_val == min_val:
        return f"  All values are {format_duration(min_val)}\n"

    bin_width = (max_val - min_val) / bin_count
    bins = [0] * bin_count
    for v in durations_sec:
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


def analyze(log_path: str) -> None:
    starts: dict[str, list[datetime]] = defaultdict(list)
    finishes: dict[str, list[datetime]] = defaultdict(list)

    line_count = 0
    matched_count = 0

    with open(log_path, "r") as f:
        for line in f:
            line_count += 1
            line = line.rstrip("\n")
            m = PROCESSING_RE.match(line)
            if m:
                ts, task_id = m.group(1), m.group(2)
                starts[task_id].append(parse_timestamp(ts))
                matched_count += 1
                continue
            m = FINISHED_RE.match(line)
            if m:
                ts, task_id = m.group(1), m.group(2)
                finishes[task_id].append(parse_timestamp(ts))
                matched_count += 1

    all_task_ids = set(starts.keys()) | set(finishes.keys())

    # Categorize tasks
    normal_tasks: list[tuple[str, float, int]] = []  # (id, duration_sec, num_starts)
    anomalies: list[str] = []

    for task_id in sorted(all_task_ids, key=int):
        s_list = starts.get(task_id, [])
        f_list = finishes.get(task_id, [])

        if not s_list and f_list:
            anomalies.append(
                f"  ID:{task_id} - Finished without any 'Processing sample' start "
                f"(finish at {f_list[0].strftime(TIMESTAMP_FMT)})"
            )
            continue

        if s_list and not f_list:
            continue

        if len(f_list) > 1:
            anomalies.append(
                f"  ID:{task_id} - Multiple 'Finished' logs ({len(f_list)} times), expected exactly 1"
            )

        last_start = max(s_list)
        finish = max(f_list)
        duration = (finish - last_start).total_seconds()

        if duration < 0:
            anomalies.append(
                f"  ID:{task_id} - Negative duration ({format_duration(duration)}): "
                f"last start {last_start.strftime(TIMESTAMP_FMT)} > "
                f"finish {finish.strftime(TIMESTAMP_FMT)}"
            )
            continue

        normal_tasks.append((task_id, duration, len(s_list)))

    durations = [d for _, d, _ in normal_tasks]
    durations_sorted = sorted(durations)
    multi_start_tasks = [(tid, d, n) for tid, d, n in normal_tasks if n > 1]
    single_start_tasks = [(tid, d, n) for tid, d, n in normal_tasks if n == 1]

    # --- Output ---
    print("=" * 80)
    print("TRANSLATION WORKER TASK DURATION ANALYSIS")
    print(f"Log file: {log_path}")
    print(f"Total log lines scanned: {line_count}")
    print(f"Matched task events: {matched_count}")
    print("=" * 80)

    print(f"\n--- OVERVIEW ---")
    print(f"  Unique task IDs seen:          {len(all_task_ids)}")
    print(f"  Tasks with start + finish:     {len(normal_tasks)}")
    print(f"    Single-start tasks:          {len(single_start_tasks)}")
    print(f"    Multi-start tasks (retries): {len(multi_start_tasks)}")
    started_only = sum(1 for tid in all_task_ids if tid in starts and tid not in finishes)
    finished_only = sum(1 for tid in all_task_ids if tid not in starts and tid in finishes)
    print(f"  Tasks started but not finished:{started_only}")
    print(f"  Tasks finished without start:  {finished_only}")

    if durations_sorted:
        mean_d = sum(durations_sorted) / len(durations_sorted)
        variance = sum((x - mean_d) ** 2 for x in durations_sorted) / len(durations_sorted)
        std_d = variance ** 0.5

        print(f"\n--- DURATION STATISTICS (last start -> finish) ---")
        print(f"  Count:   {len(durations_sorted)}")
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

    if durations_sorted:
        # Top 10 longest tasks
        longest = sorted(normal_tasks, key=lambda x: x[1], reverse=True)[:10]
        print(f"--- TOP 10 LONGEST TASKS ---")
        for rank, (tid, dur, n_starts) in enumerate(longest, 1):
            retry_note = f" (retried {n_starts} times)" if n_starts > 1 else ""
            print(f"  {rank:2d}. ID:{tid:>10s}  {format_duration(dur):>15s}{retry_note}")

        # Top 10 shortest tasks
        shortest = sorted(normal_tasks, key=lambda x: x[1])[:10]
        print(f"\n--- TOP 10 SHORTEST TASKS ---")
        for rank, (tid, dur, n_starts) in enumerate(shortest, 1):
            retry_note = f" (retried {n_starts} times)" if n_starts > 1 else ""
            print(f"  {rank:2d}. ID:{tid:>10s}  {format_duration(dur):>15s}{retry_note}")

    if anomalies:
        print(f"\n--- ANOMALIES ({len(anomalies)}) ---")
        for a in anomalies:
            print(a)
    else:
        print(f"\n--- ANOMALIES ---")
        print("  None detected.")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze task execution durations from translation worker logs."
    )
    parser.add_argument("logfile", help="Path to the worker .err log file")
    args = parser.parse_args()
    analyze(args.logfile)


if __name__ == "__main__":
    main()
