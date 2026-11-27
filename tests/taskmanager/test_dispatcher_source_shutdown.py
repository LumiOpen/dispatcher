import signal
import unittest
from unittest.mock import Mock, patch

from dispatcher.models import BatchResultResponse, BatchWorkResponse, ReleaseWorkResponse, WorkItem, WorkStatus
from dispatcher.taskmanager.cli import _install_signal_handlers
from dispatcher.taskmanager.taskmanager import TaskManager
from dispatcher.taskmanager.tasksource.dispatcher import DispatcherTaskSource


class DummyTask:
    def __init__(self, data, context):
        self.context = context

    def get_result(self):
        return {"ok": True}, self.context

    def should_retry(self):
        return False


class TestFetchedWorkShutdown(unittest.TestCase):
    @patch("dispatcher.taskmanager.tasksource.dispatcher.WorkClient")
    def test_fetched_batch_released_before_activation(self, client_class):
        client = client_class.return_value
        client.get_work.return_value = BatchWorkResponse(status=WorkStatus.OK, items=[
            WorkItem(work_id=i, content='{"x": 1}') for i in range(3)
        ])
        client.release_work.return_value = ReleaseWorkResponse(status=WorkStatus.OK, released_count=3)
        source = DispatcherTaskSource("localhost:9999", DummyTask, batch_size=3)
        self.assertEqual(len(source.get_next_tasks()), 3)
        self.assertEqual(source.outstanding_work_ids, {0, 1, 2})

        handlers = {}
        with patch("dispatcher.taskmanager.cli.signal.signal", side_effect=lambda sig, fn: handlers.setdefault(sig, fn)):
            _install_signal_handlers(Mock(), TaskManager(), source)
            with self.assertRaises(SystemExit):
                handlers[signal.SIGTERM](signal.SIGTERM, None)
        client.release_work.assert_called_once_with([0, 1, 2])

    @patch("dispatcher.taskmanager.tasksource.dispatcher.WorkClient")
    def test_acknowledged_results_leave_only_unconfirmed_work(self, client_class):
        client = client_class.return_value
        client.get_work.return_value = BatchWorkResponse(status=WorkStatus.OK, items=[
            WorkItem(work_id=i, content='{"x": 1}') for i in range(2)
        ])
        client.submit_results.side_effect = [
            BatchResultResponse(status=WorkStatus.OK, count=1),
            BatchResultResponse(status=WorkStatus.SERVER_UNAVAILABLE, count=0),
        ]
        source = DispatcherTaskSource("localhost:9999", DummyTask, batch_size=2)
        tasks = source.get_next_tasks()
        source.save_task_result(tasks[0])
        source.save_task_result(tasks[1])
        self.assertEqual(source.outstanding_work_ids, {1})

    @patch("dispatcher.taskmanager.tasksource.dispatcher.WorkClient")
    def test_signal_during_fetch_releases_returned_batch(self, client_class):
        client = client_class.return_value
        client.release_work.return_value = ReleaseWorkResponse(status=WorkStatus.OK, released_count=2)
        source = DispatcherTaskSource("localhost:9999", DummyTask, batch_size=2)
        handlers = {}

        def delayed_get_work(batch_size):
            handlers[signal.SIGTERM](signal.SIGTERM, None)
            return BatchWorkResponse(status=WorkStatus.OK, items=[
                WorkItem(work_id=i, content='{"x": 1}') for i in range(2)
            ])

        client.get_work.side_effect = delayed_get_work
        with patch("dispatcher.taskmanager.cli.signal.signal", side_effect=lambda sig, fn: handlers.setdefault(sig, fn)):
            _install_signal_handlers(Mock(), TaskManager(), source)
            with self.assertRaises(SystemExit):
                source.get_next_tasks()
        client.release_work.assert_called_once_with([0, 1])
