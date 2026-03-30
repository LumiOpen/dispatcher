#!/usr/bin/env python3
"""Analyze dispatcher server logs for retry/reissue statistics, split by server session."""

import argparse
import re
from collections import defaultdict
from dataclasses import dataclass, field


# --- Regex patterns ---

REISSUING_RE = re.compile(
    r"^INFO:root:Reissuing (\d+) after expiration \(self\.expired_reissues=(\d+)\)\.$"
)
RELEASED_RE = re.compile(
    r"^INFO:root:Released work item (\d+) for immediate reissue\.$"
)
TOMBSTONE_RE = re.compile(
    r"^WARNING:root:Work item (\d+) exceeded max_retries \((\d+)\)\. Writing tombstone\.$"
)
DUPLICATE_RE = re.compile(
    r"^WARNING:root:Duplicate completion for row (\d+); discarding\.$"
)
CHECKPOINT_RE = re.compile(
    r"^INFO:root:Checkpoint: last_processed_work_id=(-?\d+), input_offset=(\d+), "
    r"output_offset=(\d+), issued=(\d+), pending=(\d+), heap_size=(\d+), "
    r"expired_reissues=(\d+)$"
)
SERVER_START_RE = re.compile(
    r"^INFO:root:Server starting with .* retry_time=(\d+) work_timeout=(\d+), max_retries=(\d+)$"
)
LOADED_CHECKPOINT_RE = re.compile(
    r"^INFO:root:Loaded checkpoint: last_processed_work_id=(-?\d+), "
    r"input_offset=(\d+), output_offset=(\d+)$"
)
RELEASED_SUMMARY_RE = re.compile(
    r"^INFO:root:Released (\d+)/(\d+) work items$"
)
SESSION_BOUNDARY_RE = re.compile(
    r"DeprecationWarning:\s*$"
)


@dataclass
class SessionStats:
    session_id: int = 0
    start_line: int = 0
    end_line: int = 0
    line_count: int = 0
    server_config: dict | None = None
    loaded_checkpoint: dict | None = None
    last_checkpoint: dict | None = None
    reissues_per_id: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    releases_per_id: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    tombstoned: dict[int, int] = field(default_factory=dict)
    duplicate_completions: list[int] = field(default_factory=list)
    release_summaries: list[tuple[int, int]] = field(default_factory=list)
    total_reissues: int = 0
    total_releases: int = 0


def make_bar(count: int, total: int, max_width: int = 50) -> str:
    if total == 0:
        return ""
    bar_len = int((count / total) * max_width)
    return "\u2588" * bar_len


def parse_line(line: str, stats: SessionStats) -> None:
    m = REISSUING_RE.match(line)
    if m:
        wid = int(m.group(1))
        stats.reissues_per_id[wid] += 1
        stats.total_reissues += 1
        return

    m = RELEASED_RE.match(line)
    if m:
        wid = int(m.group(1))
        stats.releases_per_id[wid] += 1
        stats.total_releases += 1
        return

    m = TOMBSTONE_RE.match(line)
    if m:
        wid, max_r = int(m.group(1)), int(m.group(2))
        stats.tombstoned[wid] = max_r
        return

    m = DUPLICATE_RE.match(line)
    if m:
        stats.duplicate_completions.append(int(m.group(1)))
        return

    m = CHECKPOINT_RE.match(line)
    if m:
        stats.last_checkpoint = {
            "last_processed_work_id": int(m.group(1)),
            "input_offset": int(m.group(2)),
            "output_offset": int(m.group(3)),
            "issued": int(m.group(4)),
            "pending": int(m.group(5)),
            "heap_size": int(m.group(6)),
            "expired_reissues": int(m.group(7)),
        }
        return

    m = SERVER_START_RE.match(line)
    if m:
        stats.server_config = {
            "retry_time": int(m.group(1)),
            "work_timeout": int(m.group(2)),
            "max_retries": int(m.group(3)),
        }
        return

    m = LOADED_CHECKPOINT_RE.match(line)
    if m:
        stats.loaded_checkpoint = {
            "last_processed_work_id": int(m.group(1)),
            "input_offset": int(m.group(2)),
            "output_offset": int(m.group(3)),
        }
        return

    m = RELEASED_SUMMARY_RE.match(line)
    if m:
        stats.release_summaries.append((int(m.group(1)), int(m.group(2))))


def merge_sessions(sessions: list[SessionStats]) -> SessionStats:
    """Merge multiple sessions into a single aggregate SessionStats."""
    merged = SessionStats(
        session_id=-1,
        start_line=sessions[0].start_line,
        end_line=sessions[-1].end_line,
        line_count=sum(s.line_count for s in sessions),
    )
    merged.server_config = sessions[-1].server_config
    merged.last_checkpoint = sessions[-1].last_checkpoint
    for s in sessions:
        for wid, count in s.reissues_per_id.items():
            merged.reissues_per_id[wid] += count
        for wid, count in s.releases_per_id.items():
            merged.releases_per_id[wid] += count
        merged.tombstoned.update(s.tombstoned)
        merged.duplicate_completions.extend(s.duplicate_completions)
        merged.release_summaries.extend(s.release_summaries)
        merged.total_reissues += s.total_reissues
        merged.total_releases += s.total_releases
    return merged


def print_report(stats: SessionStats, title: str) -> None:
    print("=" * 80)
    print(title)
    print(f"Lines: {stats.start_line}-{stats.end_line} ({stats.line_count} lines)")
    print("=" * 80)

    if stats.server_config:
        cfg = stats.server_config
        print(f"\n--- SERVER CONFIGURATION ---")
        print(f"  retry_time (checkpoint interval): {cfg['retry_time']}s")
        print(f"  work_timeout:                     {cfg['work_timeout']}s")
        print(f"  max_retries:                      {cfg['max_retries']}")

    if stats.loaded_checkpoint:
        lc = stats.loaded_checkpoint
        print(f"\n--- LOADED CHECKPOINT (state at session start) ---")
        print(f"  last_processed_work_id: {lc['last_processed_work_id']}")
        print(f"  input_offset:           {lc['input_offset']}")
        print(f"  output_offset:          {lc['output_offset']}")

    if stats.last_checkpoint:
        cp = stats.last_checkpoint
        print(f"\n--- LAST CHECKPOINT (state at session end) ---")
        print(f"  last_processed_work_id: {cp['last_processed_work_id']}")
        print(f"  issued (in-flight):     {cp['issued']}")
        print(f"  pending (to write):     {cp['pending']}")
        print(f"  heap_size:              {cp['heap_size']}")
        print(f"  expired_reissues:       {cp['expired_reissues']}")

    ids_with_reissues = set(stats.reissues_per_id.keys())
    released_only = set(stats.releases_per_id.keys()) - ids_with_reissues

    all_observed_ids = ids_with_reissues | set(stats.releases_per_id.keys()) | set(stats.tombstoned.keys())
    max_observed_id = max(all_observed_ids) if all_observed_ids else -1
    estimated_total_work_ids = None
    if stats.last_checkpoint:
        lp = stats.last_checkpoint["last_processed_work_id"]
        issued = stats.last_checkpoint["issued"]
        estimated_total_work_ids = max(lp + issued, max_observed_id + 1)

    taskretry_reissues = 0
    timeout_reissues = 0
    for wid, reissue_count in stats.reissues_per_id.items():
        release_count = stats.releases_per_id.get(wid, 0)
        taskretry_reissues += min(release_count, reissue_count)
        timeout_reissues += max(0, reissue_count - release_count)

    print(f"\n--- REISSUE OVERVIEW ---")
    print(f"  Total reissue events ('Reissuing X after expiration'):   {stats.total_reissues}")
    print(f"  Total release events ('Released work item X ...'):       {stats.total_releases}")
    print(f"  Unique work IDs reissued:                                {len(ids_with_reissues)}")
    print(f"  Unique work IDs released (TaskRetry):                    {len(stats.releases_per_id)}")
    print(f"  Released but not yet reissued (log truncated?):          {len(released_only)}")
    if estimated_total_work_ids is not None:
        never_reissued = estimated_total_work_ids - len(ids_with_reissues)
        print(f"  Estimated total unique work IDs:                         ~{estimated_total_work_ids}")
        print(f"  Work IDs never reissued (succeeded first try):           ~{never_reissued}")
    print(f"\n  Reissue breakdown (approximate):")
    print(f"    TaskRetry-triggered reissues:  {taskretry_reissues}")
    print(f"    Timeout-triggered reissues:    {timeout_reissues}")

    # --- Distribution of reissue counts ---
    reissue_dist: dict[int, int] = defaultdict(int)
    for wid in ids_with_reissues:
        reissue_dist[stats.reissues_per_id[wid]] += 1

    if estimated_total_work_ids is not None:
        never_reissued = estimated_total_work_ids - len(ids_with_reissues)
        reissue_dist[0] = never_reissued
        dist_total = estimated_total_work_ids
    else:
        dist_total = len(ids_with_reissues)

    reissue_counts = list(stats.reissues_per_id.values())
    if reissue_counts:
        mean_reissues = sum(reissue_counts) / len(reissue_counts)
        max_reissues = max(reissue_counts)
        sorted_counts = sorted(reissue_counts)
        median_reissues = sorted_counts[len(sorted_counts) // 2]
    else:
        mean_reissues = max_reissues = median_reissues = 0

    print(f"\n--- ATTEMPT DISTRIBUTION (attempts = reissues + 1) ---")
    print(f"  Among tasks that were reissued at least once:")
    print(f"    Count:  {len(reissue_counts)}")
    print(f"    Mean reissues:   {mean_reissues:.2f}  (mean attempts: {mean_reissues + 1:.2f})")
    print(f"    Median reissues: {median_reissues}  (median attempts: {median_reissues + 1})")
    print(f"    Max reissues:    {max_reissues}  (max attempts: {max_reissues + 1})")

    print(f"\n  {'Attempts':>10s}  {'Reissues':>10s}  {'Tasks':>8s}  {'%':>7s}  {'Cumul %':>8s}  Bar")
    print(f"  {'':->10s}  {'':->10s}  {'':->8s}  {'':->7s}  {'':->8s}  {'':->30s}")
    cumul = 0.0
    for n_reissues in sorted(reissue_dist.keys()):
        count = reissue_dist[n_reissues]
        pct = 100.0 * count / dist_total if dist_total > 0 else 0
        cumul += pct
        bar = make_bar(count, dist_total)
        n_attempts = n_reissues + 1
        print(f"  {n_attempts:>10d}  {n_reissues:>10d}  {count:>8d}  {pct:>6.1f}%  {cumul:>7.1f}%  {bar}")

    # --- TaskRetry vs timeout per-ID breakdown ---
    print(f"\n--- PER-TASK RETRY TRIGGER BREAKDOWN ---")
    both_count = only_taskretry = only_timeout = 0
    for wid in ids_with_reissues:
        rel = stats.releases_per_id.get(wid, 0)
        reis = stats.reissues_per_id[wid]
        timed_out = reis - min(rel, reis)
        if rel > 0 and timed_out > 0:
            both_count += 1
        elif rel > 0:
            only_taskretry += 1
        else:
            only_timeout += 1
    print(f"  Tasks reissued only via TaskRetry:       {only_taskretry}")
    print(f"  Tasks reissued only via timeout:         {only_timeout}")
    print(f"  Tasks reissued via both:                 {both_count}")

    # --- Tombstoned items ---
    print(f"\n--- TOMBSTONED ITEMS (exceeded max_retries) ---")
    print(f"  Total tombstoned: {len(stats.tombstoned)}")
    if stats.tombstoned:
        max_retries_vals = set(stats.tombstoned.values())
        for mr in sorted(max_retries_vals):
            n = sum(1 for v in stats.tombstoned.values() if v == mr)
            print(f"    max_retries={mr}: {n} items")

        print(f"\n  Tombstoned work IDs and their reissue counts:")
        tomb_items = sorted(stats.tombstoned.keys())
        for wid in tomb_items[:20]:
            reis = stats.reissues_per_id.get(wid, 0)
            rel = stats.releases_per_id.get(wid, 0)
            print(f"    ID:{wid:>8d}  reissues: {reis}  "
                  f"(released: {rel}, max_retries: {stats.tombstoned[wid]})")
        if len(tomb_items) > 20:
            print(f"    ... and {len(tomb_items) - 20} more")

    # --- Duplicate completions ---
    if stats.duplicate_completions:
        print(f"\n--- DUPLICATE COMPLETIONS ---")
        print(f"  Total: {len(stats.duplicate_completions)}")
        for wid in stats.duplicate_completions:
            reis = stats.reissues_per_id.get(wid, 0)
            print(f"    ID:{wid:>8d}  reissues: {reis}")

    # --- Top most-reissued items ---
    top_n = min(20, len(reissue_counts))
    if top_n > 0:
        top_reissued = sorted(stats.reissues_per_id.items(), key=lambda x: x[1], reverse=True)[:top_n]
        print(f"\n--- TOP {top_n} MOST REISSUED WORK ITEMS ---")
        for rank, (wid, reis) in enumerate(top_reissued, 1):
            rel = stats.releases_per_id.get(wid, 0)
            tomb_note = " [TOMBSTONED]" if wid in stats.tombstoned else ""
            timeout_count = max(0, reis - rel)
            print(f"  {rank:3d}. ID:{wid:>8d}  reissues: {reis:>3d}  "
                  f"(TaskRetry: {min(rel, reis)}, timeout: {timeout_count}){tomb_note}")

    # --- Release summary stats ---
    if stats.release_summaries:
        batch_released = [r for r, _ in stats.release_summaries]
        batch_requested = [t for _, t in stats.release_summaries]
        total_batch_released = sum(batch_released)
        total_batch_requested = sum(batch_requested)
        print(f"\n--- RELEASE BATCH SUMMARY ---")
        print(f"  Total /release calls:    {len(stats.release_summaries)}")
        print(f"  Total items released:    {total_batch_released}")
        print(f"  Total items requested:   {total_batch_requested}")
        if total_batch_requested > 0:
            print(f"  Release success rate:    "
                  f"{100.0 * total_batch_released / total_batch_requested:.1f}%")

    print("\n" + "=" * 80)


def analyze(log_path: str) -> None:
    # First pass: read all lines and split into sessions at server restart boundaries.
    sessions: list[SessionStats] = []
    current: SessionStats | None = None
    line_num = 0

    with open(log_path, "r") as f:
        for raw_line in f:
            line_num += 1
            line = raw_line.rstrip("\n")

            if SESSION_BOUNDARY_RE.search(line):
                if current is not None:
                    current.end_line = line_num - 1
                    current.line_count = current.end_line - current.start_line + 1
                current = SessionStats(
                    session_id=len(sessions) + 1,
                    start_line=line_num,
                )
                sessions.append(current)

            if current is not None:
                parse_line(line, current)

    if current is not None:
        current.end_line = line_num
        current.line_count = current.end_line - current.start_line + 1

    if not sessions:
        print("No server sessions found in the log file.")
        return

    # Print header
    print("#" * 80)
    print("DISPATCHER SERVER RETRY / REISSUE ANALYSIS")
    print(f"Log file: {log_path}")
    print(f"Total log lines: {line_num}")
    print(f"Server sessions found: {len(sessions)}")
    print("#" * 80)

    # Per-session reports
    for s in sessions:
        cfg_summary = ""
        if s.server_config:
            cfg_summary = (f" (work_timeout={s.server_config['work_timeout']}s, "
                           f"max_retries={s.server_config['max_retries']})")
        title = f"SESSION {s.session_id} of {len(sessions)}{cfg_summary}"
        print(f"\n")
        print_report(s, title)

    # Aggregate report
    if len(sessions) > 1:
        merged = merge_sessions(sessions)
        print(f"\n")
        print_report(merged, "AGGREGATE (all sessions combined)")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze dispatcher server logs for retry/reissue statistics."
    )
    parser.add_argument("logfile", help="Path to the dispatcher server .err log file")
    args = parser.parse_args()
    analyze(args.logfile)


if __name__ == "__main__":
    main()
