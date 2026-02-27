from pydantic import BaseModel
from typing import Optional, List
from enum import Enum

class WorkStatus(str, Enum):
    OK = "OK"
    ALL_WORK_COMPLETE = "all_work_complete"
    RETRY = "retry"
    SERVER_UNAVAILABLE = "server_unavailable"

class WorkItem(BaseModel):
    work_id: int
    content: str
    result: Optional[str] = None  # This can be filled in after processing.

    def set_result(self, new_result: str):
        """Optional convenience method to store a result directly on the item."""
        self.result = new_result

class BatchWorkResponse(BaseModel):
    status: WorkStatus
    retry_in: Optional[int] = None
    items: List[WorkItem] = []

class BatchResultSubmission(BaseModel):
    """When submitting multiple results, just reuse the same WorkItem structure."""
    items: List[WorkItem]

class BatchResultResponse(BaseModel):
    status: WorkStatus
    count: int  # how many items were processed

class ReleaseWorkRequest(BaseModel):
    work_ids: List[int]

class ReleaseWorkResponse(BaseModel):
    status: WorkStatus
    released_count: int
