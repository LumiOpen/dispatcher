# Inference example notes

## TensorWave epilog kills sibling jobs (workaround)

If the user reports their dispatcher server or workers were suddenly killed
on TensorWave, the cause is almost always the cluster epilog at
`/itshared/silo16/slurm_config/epilog.sh`:

- Checks for sibling jobs at the cgroup v1 path
  (`/sys/fs/cgroup/cpuset/slurm/uid_$UID/job_*`).
- TensorWave is on cgroup v2 → directory missing → sibling list is empty.
- The "are there other jobs of mine on this node?" loop never runs, so the
  script falls through to `kill -9` on every "leftover" process owned by the
  user — including their dispatcher server and workers.

The bug has been reported to cluster ops but is not yet fixed.

### Recommended workflow (documented in README.md)

1. Run the dispatcher server as a SLURM job
   (`jobs/launch_dispatcher_server.sh`), not on the login node.
2. Pin 8 non-preemptible workers (one GPU each) to the server's node to fill
   it, so no other short job of the user can land there and trigger the
   epilog kill.
3. Run any extra workers as `--qos=backlog --requeue` jobs on other nodes —
   epilog kills on those nodes only affect that node, not the server.

**Trade-off**: one full GPU node is held by the inference workload for the
duration of the run. Only flag this trade-off if the user has not already
acknowledged it.

Do not recommend running the server on the login node (`start_dispatcher_server.sh`
without `--run-server`) while this workaround is in effect — it loses the
node-pinning benefit. Login-node mode is fine for short ad-hoc runs but not
for production-length workloads on TensorWave.

Remove this workaround section once the epilog is fixed (cgroup v2 paths, or
`squeue -h -u $SLURM_JOB_UID -w $(hostname)` for sibling detection).
