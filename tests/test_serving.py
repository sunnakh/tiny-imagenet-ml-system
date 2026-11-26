import tempfile
import unittest
from concurrent.futures import TimeoutError
from pathlib import Path
from threading import Event
from unittest.mock import patch

import torch

from tiny_imagenet.service import App


class InferenceQueueTest(unittest.TestCase):
    def setUp(self):
        directory = self.enterContext(tempfile.TemporaryDirectory())
        self.app = App(Path(directory))
        self.addCleanup(self.app.close)
        self.running = Event()
        self.resume = Event()
        self.addCleanup(self.resume.set)
        self.image = torch.zeros((3, 64, 64))
        self.worker = self.app.active
        self.predictions = self.enterContext(
            patch.object(self.worker, "predict", side_effect=self.predict)
        )

    def predict(self, release_id, _):
        if release_id == "running":
            self.running.set()
            if not self.resume.wait(timeout=5):
                raise RuntimeError("test worker was not released")
        return 0, release_id

    def test_rejects_work_only_after_all_waiting_slots_are_used(self):
        running = self.worker.submit("running", self.image)
        self.assertTrue(self.running.wait(timeout=2))
        waiting = [
            self.worker.submit(f"waiting-{index}", self.image) for index in range(3)
        ]
        self.assertTrue(all(future is not None for future in waiting))
        self.assertIsNone(self.worker.submit("overflow", self.image))

        drained = Event()
        waiting[-1].add_done_callback(lambda _: drained.set())
        self.resume.set()
        self.assertEqual(running.result(timeout=2), (0, "running"))
        for index, future in enumerate(waiting):
            self.assertEqual(future.result(timeout=2), (0, f"waiting-{index}"))
        self.assertTrue(drained.wait(timeout=2))
        recovered = self.worker.submit("recovered", self.image)
        self.assertEqual(recovered.result(timeout=2), (0, "recovered"))

    def test_expired_work_keeps_its_slot_until_dequeued_and_skips_inference(self):
        running = self.worker.submit("running", self.image)
        self.assertTrue(self.running.wait(timeout=2))
        waiting = [
            self.worker.submit(f"expired-{index}", self.image, deadline=10)
            for index in range(3)
        ]
        self.assertTrue(all(future is not None for future in waiting))

        with self.assertRaises(TimeoutError):
            waiting[0].result(timeout=0)
        self.assertIsNone(self.worker.submit("overflow", self.image))

        with patch("tiny_imagenet.service.perf_counter", return_value=11):
            self.resume.set()
            self.assertEqual(running.result(timeout=2), (0, "running"))
            for future in waiting:
                with self.assertRaisesRegex(TimeoutError, "expired in the queue"):
                    future.result(timeout=2)
        self.predictions.assert_called_once_with("running", self.image)

    def test_discards_a_prediction_that_completes_after_its_deadline(self):
        with patch("tiny_imagenet.service.perf_counter", side_effect=(1, 11)):
            future = self.worker.submit("late", self.image, deadline=10)
            with self.assertRaisesRegex(TimeoutError, "completed after the deadline"):
                future.result(timeout=2)
        self.predictions.assert_called_once_with("late", self.image)
