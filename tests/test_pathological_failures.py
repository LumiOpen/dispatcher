import os
import time
import tempfile
import unittest
import logging
from dispatcher.data_tracker import DataTracker

# Use short timeouts for testing.
WORK_TIMEOUT = 1  # seconds
CHECKPOINT_INTERVAL = 60 # seconds

# Suppress INFO logging during tests.
logging.basicConfig(level=logging.WARNING)

class TestPathologicalBehavior(unittest.TestCase):
    def setUp(self):
        """Set up temporary files for each test."""
        self.infile = tempfile.NamedTemporaryFile(mode="w+", delete=False)
        self.outfile = tempfile.NamedTemporaryFile(mode="a+", delete=False)
        self.checkpoint = tempfile.mktemp()
        self.infile.close()
        self.outfile.close()

        # Write 10 sample work items to the input file.
        with open(self.infile.name, "w") as f:
            for i in range(10):
                f.write(f"work_content_{i}\n")

        if os.path.exists(self.checkpoint):
            os.remove(self.checkpoint)
            
        self.dt = None

    def tearDown(self):
        """Clean up files after each test."""
        if self.dt:
            self.dt.close()
        os.remove(self.infile.name)
        os.remove(self.outfile.name)
        if os.path.exists(self.checkpoint):
            os.remove(self.checkpoint)
        if os.path.exists(self.checkpoint + ".tmp"):
            os.remove(self.checkpoint + ".tmp")

    def test_completed_work_in_pending_buffer_is_not_reissued(self):
        """
        Verifies that a completed item held in the pending buffer is not reissued
        after a timeout. This prevents a single stalled work item from causing a 
        "reissue storm" of other successfully completed items.
        """
        self.dt = DataTracker(self.infile.name, self.outfile.name, self.checkpoint,
                              work_timeout=WORK_TIMEOUT)

        # Establish a blockage: complete id=0, but id=1 is pathological.
        self.dt.complete_work_batch(self.dt.get_work_batch(batch_size=1)) # id=0
        pathological_work = self.dt.get_work_batch(batch_size=1)           # id=1
        good_work = self.dt.get_work_batch(batch_size=1)                  # id=2

        # Complete the good work; it now sits in the pending buffer.
        self.dt.complete_work_batch([(good_work[0][0], "result_2")])
        self.assertIn(good_work[0][0], self.dt.pending_write)

        time.sleep(WORK_TIMEOUT + 0.5)

        # Get the next batch of work.
        new_batch = self.dt.get_work_batch(batch_size=2)
        new_ids = {item[0] for item in new_batch}

        # The batch should contain the pathological item (1) and NEW work (3),
        # but must NOT contain the already-completed item (2).
        self.assertIn(pathological_work[0][0], new_ids, "Pathological work_id=1 must be reissued.")
        self.assertNotIn(good_work[0][0], new_ids, "Completed work_id=2 must not be reissued.")

    def test_pathological_entry_is_discarded_after_max_retries(self):
        """
        Verifies that a work item is discarded after exceeding `max_retries`, 
        allowing the job to unblock and proceed. This implements a "dead-letter
        queue" mechanism for pathological entries.
        """
        max_retries = 3
        self.dt = DataTracker(self.infile.name, self.outfile.name, self.checkpoint,
                              work_timeout=WORK_TIMEOUT, max_retries=max_retries)

        # Issue work_id=0, the pathological entry.
        pathological_work_id = self.dt.get_work_batch(batch_size=1)[0][0]
        self.assertEqual(pathological_work_id, 0)

        # Simulate it being reissued `max_retries` times.
        for i in range(max_retries):
            time.sleep(WORK_TIMEOUT + 0.5)
            reissued_id = self.dt.get_work_batch(batch_size=1)[0][0]
            self.assertEqual(reissued_id, pathological_work_id, f"Should reissue id=0 on attempt {i+1}")
        
        self.assertEqual(self.dt.expired_reissues, max_retries)
        
        # On the next attempt, the item should be discarded and the system
        # should move on to the next work item (id=1).
        time.sleep(WORK_TIMEOUT + 0.5)
        next_work = self.dt.get_work_batch(batch_size=1)
        self.assertIsNotNone(next_work, "Expected to receive a new work item.")
        next_work_id = next_work[0][0]

        # Assert that the system has advanced past the pathological item.
        self.assertEqual(next_work_id, 1, "System should have moved on to work_id=1 after max_retries.")
        
        # Also, check that the pathological item was "completed" with an error.
        self.assertEqual(self.dt.last_processed_work_id, 0, "Pathological item should be marked processed.")
