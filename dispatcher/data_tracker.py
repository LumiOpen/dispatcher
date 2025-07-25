import os
import time
import json
import logging
import heapq
import threading

logging.basicConfig(level=logging.INFO)

class DataTracker:
    def __init__(self, infile_path, outfile_path, checkpoint_path,
                 work_timeout=900, checkpoint_interval=60, max_retries=3):
        """
        Parameters:
          - infile_path: Path to the input JSONL file.
          - outfile_path: Path to the output JSONL file.
          - checkpoint_path: Path to the checkpoint file.
          - work_timeout: Seconds after which issued work is considered expired.
          - checkpoint_interval: Seconds between checkpoint writes.
          - max_retries: Number of reissues before an item is discarded. -1 means infinite.
        """
        self.infile_path = infile_path
        self.outfile_path = outfile_path
        self.checkpoint_path = checkpoint_path
        self.work_timeout = work_timeout
        self.checkpoint_interval = checkpoint_interval
        self.max_retries=max_retries

        self.last_processed_work_id = -1   # Last contiguous work id written.
        self.next_work_id = 0              # Next work id to assign.
        self.input_offset = 0              # Start of next line after last recorded work

        self.last_checkpoint_time = time.time()
        self.expired_reissues = 0

        self.issued = {}            # work_id -> (content, input_offset)
        self.issued_heap = []       # min-heap of (timestamp, work_id); uses lazy deletion
        self.pending_write = {}     # work_id -> result

        self._state_lock = threading.Lock()

        # NOTE: we open these files in binary mode because os.seek/os.tell do
        # not actually represent byte offsets in text files, but an opque
        # internal figure, and we want to be able to compare offset to file
        # size.
        self.infile = open(self.infile_path, "rb")
        self.outfile = open(self.outfile_path, "ab+")
        self._load_checkpoint()

    def _load_checkpoint(self):
        # If a checkpoint file exists and is non-empty, load its state.
        if os.path.exists(self.checkpoint_path) and os.path.getsize(self.checkpoint_path) > 0:
            try:
                with open(self.checkpoint_path, "r") as f:
                    cp = json.load(f)
            except json.JSONDecodeError:
                cp = {}
            self.last_processed_work_id = cp.get("last_processed_work_id", -1)
            self.input_offset = cp.get("input_offset", 0)
            self.infile.seek(self.input_offset)
            self.outfile.seek(cp.get("output_offset", 0))

            # Any lines after the output_offset in in output_file have been
            # completed after the checkpoint is written, so we need to move
            # past them in both the outfile and the infile
            extra_lines = self.outfile.readlines()
            extra_count = len(extra_lines)

            # For each extra line in the output, discard one line from the input.
            for _ in range(extra_count):
                self.infile.readline()

            self.last_processed_work_id += extra_count
            self.next_work_id = self.last_processed_work_id + 1

            logging.info(f"Loaded checkpoint: last_processed_work_id={self.last_processed_work_id}, "
                         f"input_offset={self.input_offset}, output_offset={self.outfile.tell()}")
        else:
            self.last_processed_work_id = -1
            self.next_work_id = 0
            logging.info("No checkpoint found; starting fresh.")

    def all_work_complete(self) -> bool:
        """
        Returns True if the input file is exhausted and no pending work remains.
        """
        remaining = os.stat(self.infile_path).st_size - self.infile.tell()
        return remaining == 0 and len(self.pending_write) == 0


    def get_work_batch(self, batch_size=1):
        batch = []
        with self._state_lock:
            now = time.time()
            # check first for expired work needing to be reissued
            while self.issued_heap and len(batch) < batch_size:
                heap_ts, work_id = self.issued_heap[0]
                # lazy deleteion
                if work_id not in self.issued:
                    heapq.heappop(self.issued_heap)
                    continue
                # see if the oldest iten in minheap needs to be reissued
                if now - heap_ts > self.work_timeout:
                    heapq.heappop(self.issued_heap)
                    content, input_offset = self.issued[work_id]
                    batch.append(self._track_issued_work(now, content, input_offset, work_id))
                    continue
                break

            while len(batch) < batch_size:
                # get new work
                line = self.infile.readline()
                if not line:
                    break
                line = line.decode("utf-8")
                content = line.rstrip("\n")
                input_offset = self.infile.tell()
                batch.append(self._track_issued_work(now, content, input_offset))
        if batch:
            return batch
        return None


    def _track_issued_work(self, when, content, input_offset, work_id=None):
        if work_id is None:
            work_id = self.next_work_id
            self.next_work_id += 1
            self.issued[work_id] = (content, input_offset)
        else:
            # this is reissued work
            self.expired_reissues += 1
            logging.info(f"Reissuing {work_id} after expiration ({self.expired_reissues=}).")
            assert(work_id in self.issued)
        heapq.heappush(self.issued_heap, (when, work_id))
        return work_id, content


    def complete_work_batch(self, batch):
        with self._state_lock:
            for work_id, result in batch:
                if work_id <= self.last_processed_work_id or work_id in self.pending_write:
                    logging.warning(f"Duplicate completion for row {work_id}; discarding.")
                elif work_id not in self.issued:
                    logging.warning(f"Completion for row {work_id} not issued; discarding.")
                else:
                    self.pending_write[work_id] = result
            self._flush_pending_writes()
                
            now = time.time()
            if now - self.last_checkpoint_time >= self.checkpoint_interval:
                self._write_checkpoint()
                self.last_checkpoint_time = now
                logging.info(f"Checkpoint: last_processed_work_id={self.last_processed_work_id}, "
                             f"input_offset={self.input_offset}, output_offset={self.outfile.tell()}, "
                             f"issued={len(self.issued)}, pending={len(self.pending_write)}, "
                             f"heap_size={len(self.issued_heap)}, expired_reissues={self.expired_reissues}")


    def _flush_pending_writes(self):
        writes = []
        next_id = self.last_processed_work_id + 1
        while next_id in self.pending_write:
            result = self.pending_write.pop(next_id)
            self.last_processed_work_id = next_id
            _, self.input_offset = self.issued[next_id]
            del self.issued[next_id]

            output = result + "\n"
            output = output.encode("utf-8")
            writes.append(output)

            next_id += 1

        if writes:
            self.outfile.write(b''.join(writes))
            self.outfile.flush()


    def _write_checkpoint(self):
        cp = {
            "last_processed_work_id": self.last_processed_work_id,
            "input_offset": self.input_offset,
            "output_offset": self.outfile.tell()
        }
        temp_path = self.checkpoint_path + ".tmp"
        with open(temp_path, "w") as f:
            json.dump(cp, f)
            f.flush()
            os.fsync(f.fileno())
        os.rename(temp_path, self.checkpoint_path)

    def close(self):
        with self._state_lock:
            # Write a final checkpoint and log status before shutting down.
            self._write_checkpoint()
            logging.info(f"Final checkpoint written: last_processed_work_id={self.last_processed_work_id}, "
                         f"input_offset={self.input_offset}, output_offset={self.outfile.tell()}, "
                         f"issued={len(self.issued)}, pending={len(self.pending_write)}, "
                         f"heap_size={len(self.issued_heap)}, expired_reissues={self.expired_reissues}")
            self.infile.close()
            self.outfile.close()
