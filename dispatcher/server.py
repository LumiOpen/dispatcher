import argparse
import uvicorn
import logging
import threading
import time
import sys
from fastapi import FastAPI, Query, Body
from pydantic import BaseModel

from dispatcher.models import (
    WorkItem,
    BatchWorkResponse,
    BatchResultSubmission,
    BatchResultResponse,
    WorkStatus,
)
from dispatcher.data_tracker import DataTracker

app = FastAPI()

dt = None            # Global DataTracker
retry_time = 300     # default wait time
shutdown_interval = 5

def background_shutdown():
    while True:
        time.sleep(shutdown_interval)
        global dt
        if dt is not None and dt.all_work_complete():
            logging.info("All work complete. Shutting down server.")
            sys.exit(0)

@app.on_event("startup")
def startup_event():
    global dt
    if dt is None:
        raise RuntimeError("DataTracker is not initialized!")
    # start a background thread to check for completion
    threading.Thread(target=background_shutdown, daemon=True).start()

@app.get("/work", response_model=BatchWorkResponse)
def get_work(batch_size: int = Query(1, ge=1)):
    global dt, retry_time
    if dt.all_work_complete():
        return BatchWorkResponse(status=WorkStatus.ALL_WORK_COMPLETE, items=[])
        
    batch = dt.get_work_batch(batch_size)
    if batch:
        items = [WorkItem(work_id=i[0], content=i[1]) for i in batch]
        return BatchWorkResponse(status=WorkStatus.OK, items=items)
    else:
        # The input is exhausted but pending work not done => "retry"
        return BatchWorkResponse(status=WorkStatus.RETRY, retry_in=retry_time)

@app.post("/results", response_model=BatchResultResponse)
def submit_results(batch: BatchResultSubmission):
    global dt
    # TODO how do i get success count?
    success_count = len(batch.items)
    dt.complete_work_batch([(i.work_id, i.result) for i in batch.items])
    return BatchResultResponse(status=WorkStatus.OK, count=success_count)

@app.get("/status")
def get_status():
    global dt
    return {
        "last_processed_work_id": dt.last_processed_work_id,
        "next_work_id": dt.next_work_id,
        "issued": len(dt.issued),
        "pending": len(dt.pending_write),
        "heap_size": len(dt.issued_heap),
        "expired_reissues": dt.expired_reissues
    }

# Pydantic model for the request body
class WorkTimeoutUpdate(BaseModel):
    timeout: float  # or int, depending on your exact need

@app.post("/work_timeout")
def update_work_timeout(data: WorkTimeoutUpdate = Body(...)):
    """
    Update the dt.work_timeout to a new value. 
    The request body should be a JSON object with a key "timeout".
    """
    global dt
    if dt is None:
        return {"error": "DataTracker is not initialized"}

    # Update the DataTracker's work_timeout
    dt.work_timeout = data.timeout
    return {
        "status": "OK",
        "message": f"work_timeout updated to {dt.work_timeout}"
    }

def main():
    parser = argparse.ArgumentParser(description="Dispatcher Server")
    parser.add_argument("--infile", type=str, required=True, help="Input file path")
    parser.add_argument("--outfile", type=str, required=True, help="Output file path")
    parser.add_argument("--checkpoint", type=str, help="Checkpoint file path")
    parser.add_argument("--retry", type=int, default=300, help="Retry time in seconds when no work is available")
    parser.add_argument("--work-timeout", type=int, default=1200, help="Time in seconds to timeout pending work and reissue it.")
    parser.add_argument("--max-retries", type=int, default=3, help="Number of times to reissue an entry before writing a tombstone. Set to -1 for infinite retries.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host")
    parser.add_argument("--port", type=int, default=8000, help="Port")
    args = parser.parse_args()

    global dt, retry_time
    retry_time = args.retry
    checkpoint_path = args.checkpoint if args.checkpoint else args.outfile + ".checkpoint"
    dt = DataTracker(
        args.infile,
        args.outfile,
        checkpoint_path,
        work_timeout=args.work_timeout,
        max_retries=args.max_retries
    )
    logging.info("Server starting with infile=%s, outfile=%s, checkpoint=%s, retry_time=%d work_timeout=%d, max_retries=%d",
                 args.infile, args.outfile, checkpoint_path, retry_time, args.work_timeout, args.max_retries)

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="warning",
        access_log=False,
    )

if __name__ == "__main__":
    main()
