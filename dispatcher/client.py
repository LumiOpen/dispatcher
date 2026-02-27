import requests
from typing import List
from dispatcher.models import (
    WorkItem,
    WorkStatus,
    BatchWorkResponse,
    BatchResultSubmission,
    BatchResultResponse,
    ReleaseWorkRequest,
    ReleaseWorkResponse,
)

class WorkClient:
    def __init__(self, server_url: str):
        if not (server_url.startswith("http://") or server_url.startswith("https://")):
            server_url = "http://" + server_url
        self.server_url = server_url.rstrip("/")

    def get_work(self, batch_size: int = 1) -> BatchWorkResponse:
        """
        Fetch up to batch_size work items from the server.
        Return a BatchWorkResponse with:
          - status: (OK, RETRY, ALL_WORK_COMPLETE, or SERVER_UNAVAILABLE)
          - retry_in: int if status=RETRY
          - items: list of WorkItem (if status=OK)
        """
        url = f"{self.server_url}/work"
        params = {"batch_size": batch_size}

        try:
            resp = requests.get(url, params=params)
        except requests.ConnectionError:
            # Return a "server unavailable" response
            return BatchWorkResponse(status=WorkStatus.SERVER_UNAVAILABLE, items=[])

        if resp.status_code == 404:
            # Not typical unless your server is returning 404 for no work
            return BatchWorkResponse(status=WorkStatus.ALL_WORK_COMPLETE, items=[])

        resp.raise_for_status()
        data = resp.json()
        return BatchWorkResponse(**data)

    def submit_results(self, items: List[WorkItem]) -> BatchResultResponse:
        """
        Post a list of WorkItem objects (with .result set) to /results.
        Returns a BatchResultResponse with:
          - status: OK or SERVER_UNAVAILABLE
          - count: number of items processed
        """
        url = f"{self.server_url}/results"
        # Build the JSON body
        submission = BatchResultSubmission(items=items)

        try:
            resp = requests.post(url, json=submission.dict())
        except requests.ConnectionError:
            return BatchResultResponse(status=WorkStatus.SERVER_UNAVAILABLE, count=0)

        resp.raise_for_status()
        data = resp.json()
        return BatchResultResponse(**data)

    def release_work(self, work_ids: List[int]) -> ReleaseWorkResponse:
        """Release in-flight work items back to the server for reissue.

        Best-effort: returns released_count=0 on connection failure.
        """
        url = f"{self.server_url}/release"
        body = ReleaseWorkRequest(work_ids=work_ids)

        try:
            resp = requests.post(url, json=body.dict(), timeout=5)
        except (requests.ConnectionError, requests.Timeout):
            return ReleaseWorkResponse(status=WorkStatus.SERVER_UNAVAILABLE, released_count=0)

        resp.raise_for_status()
        data = resp.json()
        return ReleaseWorkResponse(**data)
