import os
import time
import json
import tempfile
import unittest
from dispatcher.data_tracker import DataTracker

# Use short timeouts for testing.
WORK_TIMEOUT = 2      # seconds
CHECKPOINT_INTERVAL = 1  # seconds

class TestDataTracker(unittest.TestCase):
    def setUp(self):
        # Create temporary files for input and output.
        self.infile = tempfile.NamedTemporaryFile(mode="w+", delete=False)
        self.outfile = tempfile.NamedTemporaryFile(mode="a+", delete=False)
        # Use mktemp for the checkpoint file.
        self.checkpoint = tempfile.mktemp()
        self.infile.close()
        self.outfile.close()

        # Write sample rows to the input file (7 rows).
        with open(self.infile.name, "w") as f:
            for i in range(7):
                f.write(f"row_content_{i}\n")

        # Ensure checkpoint file does not exist (simulate cold start).
        if os.path.exists(self.checkpoint):
            os.remove(self.checkpoint)

    def tearDown(self):
        os.remove(self.infile.name)
        os.remove(self.outfile.name)
        if os.path.exists(self.checkpoint):
            os.remove(self.checkpoint)

    def test_cold_start(self):
        dt = DataTracker(self.infile.name, self.outfile.name, self.checkpoint,
                         work_timeout=WORK_TIMEOUT, checkpoint_interval=CHECKPOINT_INTERVAL)
        self.assertEqual(dt.last_processed_work_id, -1)
        self.assertEqual(dt.input_offset, 0)
        self.assertEqual(dt.next_work_id, 0)
        dt.close()

    def test_get_work_batch_and_reissue(self):
        dt = DataTracker(self.infile.name, self.outfile.name, self.checkpoint,
                         work_timeout=WORK_TIMEOUT, checkpoint_interval=CHECKPOINT_INTERVAL)
        row, = dt.get_work_batch()
        self.assertIsNotNone(row)
        row_id, content = row
        self.assertEqual(content, "row_content_0")
        time.sleep(WORK_TIMEOUT + 0.5)
        reissued, = dt.get_work_batch()
        self.assertEqual(reissued[0], row_id)
        self.assertEqual(reissued[1], content)
        row2, = dt.get_work_batch()
        self.assertEqual(row2[0], row_id + 1)
        self.assertEqual(row2[1], "row_content_1")
        dt.close()

    def test_complete_in_order_and_out_of_order(self):
        dt = DataTracker(self.infile.name, self.outfile.name, self.checkpoint,
                         work_timeout=WORK_TIMEOUT, checkpoint_interval=CHECKPOINT_INTERVAL)
        r0, = dt.get_work_batch()  # row 0
        r1, = dt.get_work_batch()  # row 1
        r2, = dt.get_work_batch()  # row 2

        dt.complete_work_batch([(r0[0], "result_0")])
        with open(self.outfile.name, "r") as f:
            lines = f.readlines()
        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0].strip(), "result_0")

        dt.complete_work_batch([(r2[0], "result_2")])
        self.assertEqual(dt.last_processed_work_id, 0)
        self.assertIn(r2[0], dt.pending_write)

        dt.complete_work_batch([(r1[0], "result_1")])
        self.assertEqual(dt.last_processed_work_id, 2)
        with open(self.outfile.name, "r") as f:
            lines = f.readlines()
        self.assertEqual([line.strip() for line in lines],
                         ["result_0", "result_1", "result_2"])
        dt.close()

    def test_duplicate_completion(self):
        dt = DataTracker(self.infile.name, self.outfile.name, self.checkpoint,
                         work_timeout=WORK_TIMEOUT, checkpoint_interval=CHECKPOINT_INTERVAL)
        r0, = dt.get_work_batch()  # row 0
        dt.complete_work_batch([(r0[0], "result_0")])
        dt.complete_work_batch([(r0[0], "result_duplicate")])
        with open(self.outfile.name, "r") as f:
            lines = f.readlines()
        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0].strip(), "result_0")
        dt.close()

    def test_checkpoint_written(self):
        dt = DataTracker(self.infile.name, self.outfile.name, self.checkpoint,
                         work_timeout=WORK_TIMEOUT, checkpoint_interval=CHECKPOINT_INTERVAL)
        r0, = dt.get_work_batch()
        dt.complete_work_batch([(r0[0], "result_0")])
        r1, = dt.get_work_batch()
        dt.complete_work_batch([(r1[0], "result_1")])
        time.sleep(CHECKPOINT_INTERVAL + 0.5)
        r2, = dt.get_work_batch()
        dt.complete_work_batch([(r2[0], "result_2")])
        with open(self.checkpoint, "r") as f:
            cp = json.load(f)
        self.assertEqual(cp.get("last_processed_work_id"), 2)
        dt.close()

    def test_load_from_checkpoint(self):
        # Process 2 rows and write a checkpoint.
        dt1 = DataTracker(self.infile.name, self.outfile.name, self.checkpoint,
                          work_timeout=WORK_TIMEOUT, checkpoint_interval=CHECKPOINT_INTERVAL)
        r0, = dt1.get_work_batch()  # row 0
        dt1.complete_work_batch([(r0[0], "result_0")])
        r1, = dt1.get_work_batch()  # row 1
        dt1.complete_work_batch([(r1[0], "result_1")])
        dt1._write_checkpoint()
        dt1.close()
        with open(self.checkpoint, "r") as f:
            cp = json.load(f)
        self.assertEqual(cp.get("last_processed_work_id"), 1)

        # Create a new DataTracker from the same files.
        dt2 = DataTracker(self.infile.name, self.outfile.name, self.checkpoint,
                          work_timeout=WORK_TIMEOUT, checkpoint_interval=CHECKPOINT_INTERVAL)
        self.assertEqual(dt2.last_processed_work_id, 1)
        self.assertEqual(dt2.next_work_id, 2)
        r2, = dt2.get_work_batch()
        self.assertEqual(r2[0], 2)
        self.assertEqual(r2[1], "row_content_2")
        dt2.complete_work_batch([(r2[0], "result_2")])
        with open(self.outfile.name, "r") as f:
            lines = f.readlines()
        self.assertEqual([line.strip() for line in lines],
                         ["result_0", "result_1", "result_2"])
        dt2.close()

    def test_load_from_checkpoint_with_extra_rows(self):
        """
        Process some rows, then process additional rows after the checkpoint was written.
        Upon loading, the DataTracker should:
          - Seek to the saved output offset,
          - For each extra output line, read and discard one input line,
          - Update last_processed_work_id and next_work_id accordingly.
        """
        dt1 = DataTracker(self.infile.name, self.outfile.name, self.checkpoint,
                          work_timeout=WORK_TIMEOUT, checkpoint_interval=CHECKPOINT_INTERVAL)
        r0, = dt1.get_work_batch()  # row 0
        dt1.complete_work_batch([(r0[0], "result_0")])
        r1, = dt1.get_work_batch()  # row 1
        dt1.complete_work_batch([(r1[0], "result_1")])
        r2, = dt1.get_work_batch()  # row 2
        dt1.complete_work_batch([(r2[0], "result_2")])
        # Write a checkpoint now; it records last_processed_work_id==2.
        dt1._write_checkpoint()
        # Now process additional rows.
        r3, = dt1.get_work_batch()  # row 3
        dt1.complete_work_batch([(r3[0], "result_3")])
        r4, = dt1.get_work_batch()  # row 4
        dt1.complete_work_batch([(r4[0], "result_4")])

        # checkpoint should reflect the earlier state
        with open(self.checkpoint, "r") as f:
            cp = json.load(f)
        self.assertEqual(cp.get("last_processed_work_id"), 2)
        
        # Now load a new tracker and ensure it reconciles correctly.
        dt2 = DataTracker(self.infile.name, self.outfile.name, self.checkpoint,
                          work_timeout=WORK_TIMEOUT, checkpoint_interval=CHECKPOINT_INTERVAL)
        self.assertEqual(dt2.last_processed_work_id, 4)
        self.assertEqual(dt2.next_work_id, 5)
        r5, = dt2.get_work_batch()
        self.assertEqual(r5[0], 5)
        self.assertEqual(r5[1], "row_content_5")
        dt2.close()

    def test_load_from_checkpoint_with_extra_rows_unwritten(self):
        """
        Process some rows, then process additional rows after the checkpoint was written.
        Upon loading, the DataTracker should:
          - Seek to the saved output offset,
          - For each extra output line, read and discard one input line,
          - Update last_processed_work_id and next_work_id accordingly.
        """
        dt1 = DataTracker(self.infile.name, self.outfile.name, self.checkpoint,
                          work_timeout=WORK_TIMEOUT, checkpoint_interval=CHECKPOINT_INTERVAL)
        r0, = dt1.get_work_batch()  # row 0
        dt1.complete_work_batch([(r0[0], "result_0")])
        r1, = dt1.get_work_batch()  # row 1
        dt1.complete_work_batch([(r1[0], "result_1")])
        r2, = dt1.get_work_batch()  # row 2
        dt1.complete_work_batch([(r2[0], "result_2")])
        # Write a checkpoint now; it records last_processed_work_id==2.
        dt1._write_checkpoint()
        # Now process additional rows.
        r3, = dt1.get_work_batch()  # row 3
        r4, = dt1.get_work_batch()  # row 4

        # checkpoint should reflect the earlier state
        with open(self.checkpoint, "r") as f:
            cp = json.load(f)
        self.assertEqual(cp.get("last_processed_work_id"), 2)

        # Now load a new tracker and it continues from the last written record
        dt2 = DataTracker(self.infile.name, self.outfile.name, self.checkpoint,
                          work_timeout=WORK_TIMEOUT, checkpoint_interval=CHECKPOINT_INTERVAL)
        self.assertEqual(dt2.last_processed_work_id, 2)
        self.assertEqual(dt2.next_work_id, 3)
        r5, = dt2.get_work_batch()
        self.assertEqual(r5[0], 3)
        self.assertEqual(r5[1], "row_content_3")
        dt2.close()

    def test_load_from_checkpoint_with_unsubmitted_work(self):
        """
        Test the behavior of the DataTracker when there is pending work that has been issued
        but not submitted before a checkpoint is written. The test ensures that upon loading
        from the checkpoint:
          - The DataTracker correctly resumes from the last processed work ID.
          - Issued but unsubmitted work is not incorrectly marked as processed.
          - The input offset and next work ID are consistent with the checkpoint state.
        """
        dt1 = DataTracker(self.infile.name, self.outfile.name, self.checkpoint,
                          work_timeout=WORK_TIMEOUT, checkpoint_interval=CHECKPOINT_INTERVAL)
        r0, = dt1.get_work_batch()  # row 0
        dt1.complete_work_batch([(r0[0], "result_0")])
        r1, = dt1.get_work_batch()  # row 1
        dt1.complete_work_batch([(r1[0], "result_1")])
        r2, = dt1.get_work_batch()  # row 2
        dt1.complete_work_batch([(r2[0], "result_2")])

        # Now request additional work but don't submit it.
        r3, = dt1.get_work_batch()  # row 3
        r4, = dt1.get_work_batch()  # row 4

        # Write a checkpoint now; it records last_processed_work_id==2.
        dt1._write_checkpoint()

        # checkpoint should reflect the earlier state
        with open(self.checkpoint, "r") as f:
            cp = json.load(f)
        self.assertEqual(cp.get("last_processed_work_id"), 2)

        # Now load a new tracker and it continues from the last written record
        dt2 = DataTracker(self.infile.name, self.outfile.name, self.checkpoint,
                          work_timeout=WORK_TIMEOUT, checkpoint_interval=CHECKPOINT_INTERVAL)
        self.assertEqual(dt2.last_processed_work_id, 2)
        self.assertEqual(dt2.next_work_id, 3)
        r5, = dt2.get_work_batch()
        self.assertEqual(r5[0], 3)
        self.assertEqual(r5[1], "row_content_3")
        dt2.close()

    def test_release_work_reissues_immediately(self):
        """Released work should be returned by the next get_work_batch call."""
        dt = DataTracker(self.infile.name, self.outfile.name, self.checkpoint,
                         work_timeout=WORK_TIMEOUT, checkpoint_interval=CHECKPOINT_INTERVAL)
        # Issue two items
        r0, = dt.get_work_batch()
        r1, = dt.get_work_batch()

        # Release the first one
        released = dt.release_work([r0[0]])
        self.assertEqual(released, 1)

        # Next get_work_batch should return the released item (timestamp 0 = expired)
        reissued, = dt.get_work_batch()
        self.assertEqual(reissued[0], r0[0])
        self.assertEqual(reissued[1], r0[1])
        dt.close()

    def test_release_work_increments_retry(self):
        """Released items go through the normal reissue path and increment retry_count."""
        dt = DataTracker(self.infile.name, self.outfile.name, self.checkpoint,
                         work_timeout=WORK_TIMEOUT, checkpoint_interval=CHECKPOINT_INTERVAL)
        r0, = dt.get_work_batch()
        work_id = r0[0]

        # Check initial retry_count is 0
        _, _, retry_count, _ = dt.issued[work_id]
        self.assertEqual(retry_count, 0)

        # Release and reissue
        dt.release_work([work_id])
        dt.get_work_batch()

        # retry_count should be 1
        _, _, retry_count, _ = dt.issued[work_id]
        self.assertEqual(retry_count, 1)
        dt.close()

    def test_release_completed_or_unknown_is_noop(self):
        """Releasing already-completed or unknown work_ids should be a no-op."""
        dt = DataTracker(self.infile.name, self.outfile.name, self.checkpoint,
                         work_timeout=WORK_TIMEOUT, checkpoint_interval=CHECKPOINT_INTERVAL)
        r0, = dt.get_work_batch()
        dt.complete_work_batch([(r0[0], "result_0")])

        # Release a completed item and a non-existent one
        released = dt.release_work([r0[0], 9999])
        self.assertEqual(released, 0)
        dt.close()

    def test_release_pending_write_is_noop(self):
        """Releasing an item that is in pending_write (completed but not flushed) should be a no-op."""
        dt = DataTracker(self.infile.name, self.outfile.name, self.checkpoint,
                         work_timeout=WORK_TIMEOUT, checkpoint_interval=CHECKPOINT_INTERVAL)
        # Issue items 0 and 1, complete item 1 out of order so it stays in pending_write
        r0, = dt.get_work_batch()
        r1, = dt.get_work_batch()
        dt.complete_work_batch([(r1[0], "result_1")])

        # Item 1 is in pending_write (blocked by item 0)
        self.assertIn(r1[0], dt.pending_write)

        released = dt.release_work([r1[0]])
        self.assertEqual(released, 0)
        dt.close()

    def test_genuine_timeout_still_increments_retry(self):
        """Genuine timeouts (not releases) should still increment retry_count."""
        dt = DataTracker(self.infile.name, self.outfile.name, self.checkpoint,
                         work_timeout=WORK_TIMEOUT, checkpoint_interval=CHECKPOINT_INTERVAL)
        r0, = dt.get_work_batch()
        work_id = r0[0]

        # Wait for timeout
        time.sleep(WORK_TIMEOUT + 0.5)

        # Reissue via timeout
        dt.get_work_batch()

        # retry_count should be 1
        _, _, retry_count, _ = dt.issued[work_id]
        self.assertEqual(retry_count, 1)
        dt.close()

    # ---------------------------------------------------------------
    # Scenario tests for release + timeout interactions
    # ---------------------------------------------------------------

    def test_release_stale_heap_entry_skipped_in_same_batch(self):
        """When batch_size is large enough, the released entry and the stale
        original entry could both be processed in the same get_work_batch call.
        The stale entry must be skipped."""
        dt = DataTracker(self.infile.name, self.outfile.name, self.checkpoint,
                         work_timeout=WORK_TIMEOUT, checkpoint_interval=CHECKPOINT_INTERVAL)
        r0, = dt.get_work_batch()
        work_id = r0[0]

        # Release and wait for original entry to also expire
        dt.release_work([work_id])
        time.sleep(WORK_TIMEOUT + 0.5)

        # Request a large batch — both (0, id) and (T1, id) are expired
        batch = dt.get_work_batch(batch_size=5)
        self.assertIsNotNone(batch)
        ids_in_batch = [item[0] for item in batch]
        # work_id should appear at most once
        self.assertEqual(ids_in_batch.count(work_id), 1,
                         f"work_id {work_id} appeared {ids_in_batch.count(work_id)} times in batch")
        dt.close()

    def test_release_then_complete_race(self):
        """If release happens first and then a result arrives for the same
        work_id, the result should still be accepted (item is in self.issued).
        The released heap entry should be lazily deleted."""
        dt = DataTracker(self.infile.name, self.outfile.name, self.checkpoint,
                         work_timeout=WORK_TIMEOUT, checkpoint_interval=CHECKPOINT_INTERVAL)
        # Issue two items so that completing item 1 leaves it in pending_write
        # (blocked by item 0 which is not yet complete).
        r0, = dt.get_work_batch()
        r1, = dt.get_work_batch()

        # Release item 1
        dt.release_work([r1[0]])

        # Result for item 1 arrives (worker finished just before preemption)
        dt.complete_work_batch([(r1[0], "result_1")])
        # Item 1 is in pending_write (blocked by item 0)
        self.assertIn(r1[0], dt.pending_write)

        # The released (0, id) entry should be lazily deleted since
        # work_id is now in pending_write — should not be reissued.
        batch = dt.get_work_batch()
        if batch:
            self.assertNotEqual(batch[0][0], r1[0])
        dt.close()

    def test_complete_then_release_race(self):
        """If a result is submitted and flushed before the release arrives,
        the release should be a no-op."""
        dt = DataTracker(self.infile.name, self.outfile.name, self.checkpoint,
                         work_timeout=WORK_TIMEOUT, checkpoint_interval=CHECKPOINT_INTERVAL)
        r0, = dt.get_work_batch()
        work_id = r0[0]

        # Complete and flush (it's work_id 0, so contiguous → flushed)
        dt.complete_work_batch([(work_id, "result_0")])
        self.assertNotIn(work_id, dt.issued)

        # Late release arrives — should be no-op
        released = dt.release_work([work_id])
        self.assertEqual(released, 0)
        dt.close()

    def test_multiple_releases_same_work_id(self):
        """Releasing the same work_id multiple times should only create one
        reissue, not multiple."""
        dt = DataTracker(self.infile.name, self.outfile.name, self.checkpoint,
                         work_timeout=WORK_TIMEOUT, checkpoint_interval=CHECKPOINT_INTERVAL)
        r0, = dt.get_work_batch()
        work_id = r0[0]

        # Release twice
        dt.release_work([work_id])
        dt.release_work([work_id])

        # Reissue — should appear once
        reissued, = dt.get_work_batch()
        self.assertEqual(reissued[0], work_id)

        # The second (0, id) entry should be detected as stale (issued_at
        # is now T2 from the reissue, not 0) and skipped.
        # Wait for any stale entries to expire and verify no double reissue.
        time.sleep(WORK_TIMEOUT + 0.5)
        batch = dt.get_work_batch(batch_size=5)
        if batch:
            ids = [item[0] for item in batch]
            # work_id may be reissued via timeout, but should appear at most once
            self.assertLessEqual(ids.count(work_id), 1)
        dt.close()

    def test_release_does_not_affect_other_items(self):
        """Releasing one work_id should not interfere with other issued items."""
        dt = DataTracker(self.infile.name, self.outfile.name, self.checkpoint,
                         work_timeout=WORK_TIMEOUT, checkpoint_interval=CHECKPOINT_INTERVAL)
        r0, = dt.get_work_batch()
        r1, = dt.get_work_batch()
        r2, = dt.get_work_batch()

        # Release only item 1
        dt.release_work([r1[0]])

        # Next batch should reissue item 1
        reissued, = dt.get_work_batch()
        self.assertEqual(reissued[0], r1[0])

        # Items 0 and 2 should still be in issued with their original issued_at
        self.assertIn(r0[0], dt.issued)
        self.assertIn(r2[0], dt.issued)
        _, _, retry_count_0, _ = dt.issued[r0[0]]
        _, _, retry_count_2, _ = dt.issued[r2[0]]
        self.assertEqual(retry_count_0, 0)
        self.assertEqual(retry_count_2, 0)
        dt.close()

    def test_release_max_retries_tombstone(self):
        """Repeated release+reissue cycles should eventually hit max_retries
        and write a tombstone.  The max_retries check fires BEFORE the
        reissue increments retry_count, so with max_retries=1 it takes
        two cycles: cycle 1 reissues (0→1), cycle 2 tombstones (1>=1)."""
        dt = DataTracker(self.infile.name, self.outfile.name, self.checkpoint,
                         work_timeout=WORK_TIMEOUT, checkpoint_interval=CHECKPOINT_INTERVAL,
                         max_retries=1)
        r0, = dt.get_work_batch()
        work_id = r0[0]

        # Release + reissue cycle 1 (retry_count checked at 0 < 1, reissued → 1)
        dt.release_work([work_id])
        dt.get_work_batch()
        _, _, retry_count, _ = dt.issued[work_id]
        self.assertEqual(retry_count, 1)

        # Release + reissue cycle 2 (retry_count checked at 1 >= 1 → tombstone)
        dt.release_work([work_id])
        dt.get_work_batch()

        # work_id=0 is the first item, so the tombstone is contiguous and
        # gets flushed immediately — removed from both pending_write and issued.
        self.assertNotIn(work_id, dt.issued)
        self.assertNotIn(work_id, dt.pending_write)

        # Verify tombstone was written to the output file
        with open(self.outfile.name, "r") as f:
            output = f.read()
        self.assertIn("max_retries_exceeded", output)
        dt.close()

    def test_issued_at_updated_on_reissue(self):
        """After reissue (via timeout or release), issued_at should match the
        new heap entry's timestamp."""
        dt = DataTracker(self.infile.name, self.outfile.name, self.checkpoint,
                         work_timeout=WORK_TIMEOUT, checkpoint_interval=CHECKPOINT_INTERVAL)
        r0, = dt.get_work_batch()
        work_id = r0[0]

        _, _, _, issued_at_original = dt.issued[work_id]
        self.assertGreater(issued_at_original, 0)

        # Release sets issued_at to 0
        dt.release_work([work_id])
        _, _, _, issued_at_released = dt.issued[work_id]
        self.assertEqual(issued_at_released, 0)

        # Reissue updates issued_at to a new timestamp
        dt.get_work_batch()
        _, _, _, issued_at_reissued = dt.issued[work_id]
        self.assertGreater(issued_at_reissued, 0)
        self.assertNotEqual(issued_at_reissued, issued_at_original)
        dt.close()

    def test_timeout_reissue_no_duplicate_in_batch(self):
        """In the timeout-only path (no release), a timed-out item should
        appear exactly once in the reissued batch — never duplicated."""
        dt = DataTracker(self.infile.name, self.outfile.name, self.checkpoint,
                         work_timeout=WORK_TIMEOUT, checkpoint_interval=CHECKPOINT_INTERVAL)
        r0, = dt.get_work_batch()
        work_id = r0[0]

        # Wait for timeout
        time.sleep(WORK_TIMEOUT + 0.5)

        # Request a large batch — only one reissue should happen
        batch = dt.get_work_batch(batch_size=5)
        self.assertIsNotNone(batch)
        ids_in_batch = [item[0] for item in batch]
        self.assertEqual(ids_in_batch.count(work_id), 1,
                         f"work_id {work_id} appeared {ids_in_batch.count(work_id)} times")
        dt.close()


if __name__ == "__main__":
    unittest.main()
