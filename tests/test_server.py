import os
import tempfile
import json
import unittest
from fastapi.testclient import TestClient
import dispatcher.server as server_mod
from dispatcher.data_tracker import DataTracker

class TestServer(unittest.TestCase):
    def setUp(self):
        # Create temporary input, output, and checkpoint files.
        self.infile = tempfile.NamedTemporaryFile(mode="w+", delete=False)
        self.outfile = tempfile.NamedTemporaryFile(mode="a+", delete=False)
        self.checkpoint = tempfile.mktemp()

        # Write a few lines to the input file.
        self.infile.write("content_0\n")
        self.infile.write("content_1\n")
        self.infile.write("content_2\n")
        self.infile.close()
        self.outfile.close()

        # Remove checkpoint if it exists
        if os.path.exists(self.checkpoint):
            os.remove(self.checkpoint)

        # Initialize global dt in the server module
        server_mod.dt = DataTracker(
            self.infile.name, 
            self.outfile.name, 
            self.checkpoint,
            work_timeout=2, 
            checkpoint_interval=1
        )
        self.client = TestClient(server_mod.app)

    def tearDown(self):
        server_mod.dt.close()
        os.remove(self.infile.name)
        os.remove(self.outfile.name)
        if os.path.exists(self.checkpoint):
            os.remove(self.checkpoint)

    def test_get_work_batch(self):
        """
        Test GET /work with batch_size=2
        """
        resp = self.client.get("/work?batch_size=2")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        # Expect status=OK, items have length up to 2
        self.assertEqual(data["status"], "OK")
        self.assertIn("items", data)
        items = data["items"]
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0]["content"], "content_0")
        self.assertEqual(items[1]["content"], "content_1")

        # Next call should retrieve the 3rd line
        resp2 = self.client.get("/work?batch_size=2")
        self.assertEqual(resp2.status_code, 200)
        data2 = resp2.json()
        self.assertEqual(data2["status"], "OK")
        self.assertEqual(len(data2["items"]), 1)
        self.assertEqual(data2["items"][0]["content"], "content_2")

        # Another call should return either "retry" or "all_work_complete"
        # since the input is exhausted but might not be "complete" if not all results posted.
        resp3 = self.client.get("/work?batch_size=2")
        data3 = resp3.json()
        self.assertIn(data3["status"], [ "retry", "all_work_complete" ])

    def test_submit_results(self):
        """
        Test POST /results with a batch of items.
        """
        # Grab a batch from /work
        resp = self.client.get("/work?batch_size=2")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "OK")
        items = data["items"]
        self.assertEqual(len(items), 2)

        # Fill in results for those items
        for i in items:
            i["result"] = f"processed_{i['content']}"

        # Submit them
        submit_resp = self.client.post("/results", json={"items": items})
        self.assertEqual(submit_resp.status_code, 200)
        submit_data = submit_resp.json()
        self.assertEqual(submit_data["status"], "OK")
        self.assertEqual(submit_data["count"], 2)

        # Check that the server acknowledges them as processed
        # E.g. if we call /work again for these lines, we won't get them
        # but let's do a simple check via the server_mod.dt or /status
        status_resp = self.client.get("/status")
        self.assertEqual(status_resp.status_code, 200)
        status_data = status_resp.json()
        # last_processed_work_id should now be at least 1
        self.assertGreaterEqual(status_data["last_processed_work_id"], 1)

    def test_all_work_complete(self):
        """
        Test processing all work. Then confirm we eventually get all_work_complete.
        """
        # There are 3 lines total. Let's fetch them in a batch of 3.
        r = self.client.get("/work?batch_size=3")
        data = r.json()
        self.assertEqual(data["status"], "OK")
        items = data["items"]
        self.assertEqual(len(items), 3)

        # Mark them complete
        for i in items:
            i["result"] = f"done_{i['content']}"

        post_resp = self.client.post("/results", json={"items": items})
        self.assertEqual(post_resp.status_code, 200)
        post_data = post_resp.json()
        self.assertEqual(post_data["status"], "OK")
        self.assertEqual(post_data["count"], 3)

        # Now if we ask for work again, we should get all_work_complete
        resp2 = self.client.get("/work?batch_size=3")
        data2 = resp2.json()
        self.assertEqual(data2["status"], "all_work_complete")
        self.assertEqual(len(data2["items"]), 0)

    def test_release_work(self):
        """Test POST /release endpoint."""
        # Get some work
        resp = self.client.get("/work?batch_size=2")
        data = resp.json()
        self.assertEqual(data["status"], "OK")
        items = data["items"]
        self.assertEqual(len(items), 2)

        work_ids = [i["work_id"] for i in items]

        # Release them
        release_resp = self.client.post("/release", json={"work_ids": work_ids})
        self.assertEqual(release_resp.status_code, 200)
        release_data = release_resp.json()
        self.assertEqual(release_data["status"], "OK")
        self.assertEqual(release_data["released_count"], 2)

        # The released items should be reissued immediately
        resp2 = self.client.get("/work?batch_size=2")
        data2 = resp2.json()
        self.assertEqual(data2["status"], "OK")
        reissued_ids = [i["work_id"] for i in data2["items"]]
        self.assertEqual(sorted(reissued_ids), sorted(work_ids))

    def test_release_unknown_ids(self):
        """Releasing unknown work_ids should return released_count=0."""
        release_resp = self.client.post("/release", json={"work_ids": [999, 1000]})
        self.assertEqual(release_resp.status_code, 200)
        release_data = release_resp.json()
        self.assertEqual(release_data["status"], "OK")
        self.assertEqual(release_data["released_count"], 0)

if __name__ == "__main__":
    unittest.main()
