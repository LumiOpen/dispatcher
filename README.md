# Dispatcher

Dispatcher exists to simplify running large-scale inference workflows in
a batch compute environment.

Specifically, it simplifies:
1. Scaling up to hundreds of inference workers simultaneously
2. Keeping inference backends busy with work to make efficient use of compute
3. Splitting up work across many workers without needing to pre-partition data
4. Task resumption after timeouts or crashes with minimal lost work
5. Complex multi-step inference workflows.

The system is built around the dispatcher server, which functions as a simple
work queue manager and work dispatcher, but the most advanced features come
from the task manager subsystem which allows complex multi-step inference
workflows to be written in simple declarative fashion and scheduling
efficiently while hiding all of the complexity.

Below is some information about the Dispatcher server itself, but it would be
most useful to head to the `examples/` directory to see examples of simple,
realistic implementations built around it.

## Dispatcher server

Simple library to dispatch work from a large line-oriented file (jsonl) for
distributed workers without pre-apportioning work.

Dispatcher is ideal for batch inference workloads where individual requests
may take varying amounts of time, but you want to keep all workers busy and
avoid the long tails that you might run into by dividing the work up
beforehand.

Dispatcher guarantees that each completed work item will be persisted to disk
only once, but items may be processed more than once, so it is inappropriate
for work that changes state in external systems or is otherwise not idempotent.

Work is checkpointed so that if a job ends unexpectedly, work can begin where
it left off with minimal lost work (specifically, only work which is cached
waiting to be written because it has been completed out of order will be lost.)

In order to work efficiently with large data files, ensure each item is written
only once, and avoid costly scans and reconciliation on restart, the
dispatcher works on a line-per-line basis, each nth line of input will
correspond with the nth line of output. On restarting, we only need to
determine where we left off to begin again.

This means the dispatcher must cache out of order work until it can be written
contiguously to the output file. To prevent a single pathological entry (a
"query of death") from stalling the entire job, work that is reissued more than
a configurable number of times (default: 3) will be considered un-processable.
When this limit is reached, a "tombstone" entry containing an error is written
to the output file in place of a valid result. This preserves the 1-to-1 line
mapping between the input and output files while allowing the rest of the job
to proceed.

## To Develop

```bash
pip install -e .[dev]
```

## To run the server
```bash
python -m dispatcher.server --infile path/to/input.jsonl --outfile path/to/output.jsonl
# or
dispatcher-server --infile path/to/input.jsonl --outfile path/to/output.jsonl
```

## Client example
```python
import time
from dispatcher.client import WorkClient
from dispatcher.models import WorkStatus
import json

client = WorkClient("http://127.0.0.1:8000")

while True:
    resp = client.get_work(batch_size=5)
    
    if resp.status == WorkStatus.ALL_WORK_COMPLETE:
        print("All work complete. Exiting.")
        break
        
    elif resp.status == WorkStatus.RETRY:
        print(f"No work available; retry in {resp.retry_in} seconds.")
        time.sleep(resp.retry_in)
        continue
        
    elif resp.status == WorkStatus.SERVER_UNAVAILABLE:
        # The server is not running.
        # The server exits once all work is complete, so let's assume that's the case here.
        print("Server is unavailable. Exiting.")
        break
        
    elif resp.status == WorkStatus.OK:
        batch = resp.items
        for work in batch:

            print(f"Got work: work_id={work.work_id}, content='{work.content}'")

            # Process the work
            # NOTE: work.content is still plain text here. If it contains JSON,
            # you'll still need to parse it.
            #content = json.loads(work.content)
            # do actual work here
            work.set_result(f"processed_{work.work_id}")

        # TODO error check here??
        submit_resp = client.submit_result(batch)
        print(f"Submitted {submit_resp.count} items, status={submit_resp.status}")
    else:
        print(f"Unexpected status: {resp.status}")
        break
```

## Update work timeout

If you have too much work expiring because inferences are taking a long time,
you can update the work timeout without restarting the job like this:

```bash
HOSTNAME=...
curl -X POST -H "Content-Type: application/json" \
     -d '{"timeout": 1600}' \
     http://${HOSTNAME}:8000/work_timeout
```

