import json
import logging
from typing import List

from dispatcher.client import WorkClient
from dispatcher.models import WorkStatus

from ..task.base import Task
from .base import TaskSource

class DispatcherTaskSource(TaskSource):
    """Task source that uses a Dispatcher server for task distribution and result collection."""
    
    def __init__(self, dispatcher_server: str, task_class: type, batch_size: int = 1):
        """
        Initialize a Dispatcher-based task source.
        
        Args:
            dispatcher_server: Dispatcher server address (host:port)
            task_class: Task implementation class to instantiate
            batch_size: Maximum number of tasks to get in each request
        """
        self.dispatcher_server = dispatcher_server
        self.task_class = task_class
        self.batch_size = batch_size
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize client immediately
        try:
            self.client = WorkClient(self.dispatcher_server)
            self.logger.info(f"Initialized Dispatcher client for server {self.dispatcher_server}")
        except Exception as e:
            self.logger.error(f"Failed to initialize Dispatcher client: {e}")
            self._is_exhausted = True  # Mark as exhausted so we don't keep trying
            raise
        
        self._is_exhausted = False
        # Includes fetched items that have not reached TaskManager.active_tasks.
        self.outstanding_work_ids = set()
        self.fetch_in_progress = False
        self.deferred_signal = None
        self.shutdown_handler = None
    
    def get_next_tasks(self) -> List[Task]:
        """Get up to batch_size tasks from the Dispatcher server."""
        if self._is_exhausted:
            return []
        
        self.fetch_in_progress = True
        try:
            resp = self.client.get_work(batch_size=self.batch_size)
            
            if resp.status == WorkStatus.OK and resp.items:
                self.outstanding_work_ids.update(item.work_id for item in resp.items)
                tasks = []
                
                for work_item in resp.items:
                    try:
                        # Parse the JSON content
                        try:
                            task_data = json.loads(work_item.content)
                        except json.JSONDecodeError as e:
                            self.logger.error(f"Error parsing JSON for work item {work_item.work_id}: {e}")
                            # Return an error to the dispatcher
                            work_item.set_result(json.dumps({"error": f"Failed to parse JSON: {e}"}))
                            submitted = self.client.submit_results([work_item])
                            if submitted.status == WorkStatus.OK and submitted.count == 1:
                                self.outstanding_work_ids.discard(work_item.work_id)
                            continue
                        
                        # Create a task with the data and work_item as context
                        task = self.task_class(task_data, context=work_item)
                        tasks.append(task)
                        
                    except Exception as e:
                        self.logger.exception(f"Error creating task for work item {work_item.work_id}: {e}")
                        # Return an error to the dispatcher
                        work_item.set_result(json.dumps({"error": f"Failed to create task: {str(e)}"}))
                        submitted = self.client.submit_results([work_item])
                        if submitted.status == WorkStatus.OK and submitted.count == 1:
                            self.outstanding_work_ids.discard(work_item.work_id)
                
                if tasks:
                    self.logger.debug(f"Created {len(tasks)} new tasks from Dispatcher")
                return tasks
                
            elif resp.status == WorkStatus.ALL_WORK_COMPLETE:
                self.logger.info("Dispatcher reports all work is complete")
                self._is_exhausted = True
                
            # RETRY status means no work available right now, but maybe later
            
            return []  # Return empty list for no new tasks
            
        except Exception as e:
            self.logger.exception(f"Error getting work from Dispatcher: {e}")
            return []
        finally:
            self.fetch_in_progress = False
            if self.deferred_signal is not None:
                self.shutdown_handler(self.deferred_signal, None)
    
    def save_task_result(self, task: Task) -> None:
        """Save task result back to the Dispatcher server."""
        try:
            # Get the result and context from the task
            result, context = task.get_result()
            
            # The context should be the original work_item
            work_item = context

            # If the task should be retried, release the work item back to the dispatcher
            if task.should_retry():
                resp = self.client.release_work([work_item.work_id])
                if resp.status != WorkStatus.OK or resp.released_count != 1:
                    self.logger.warning(
                        "Failed to release retry work item %s immediately; status=%s released_count=%s. "
                        "It will be retried after dispatcher timeout.",
                        work_item.work_id,
                        resp.status,
                        resp.released_count,
                    )
                else:
                    self.outstanding_work_ids.discard(work_item.work_id)
                    self.logger.debug(f"Released work item {work_item.work_id} for retry: {task.retry_reason}")
                return

            work_item.set_result(json.dumps(result, ensure_ascii=False))
            
            # Submit back to the dispatcher
            resp = self.client.submit_results([work_item])
            if resp.status == WorkStatus.OK and resp.count == 1:
                self.outstanding_work_ids.discard(work_item.work_id)
                self.logger.debug(f"Submitted result for work item {work_item.work_id} back to Dispatcher")
            else:
                self.logger.warning("Result for work item %s was not confirmed; it remains pending for timeout recovery", work_item.work_id)
            
        except Exception as e:
            self.logger.exception(f"Error saving task result: {e}")
    
    @property
    def is_exhausted(self) -> bool:
        """Check if the Dispatcher has no more work available."""
        return self._is_exhausted
