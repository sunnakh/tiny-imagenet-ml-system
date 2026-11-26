import tempfile
import unittest
from pathlib import Path

from prometheus_client.parser import text_string_to_metric_families

from tiny_imagenet.monitor import Monitor, prometheus_metrics


class MonitoringTest(unittest.TestCase):
    @staticmethod
    def samples(monitor, ready):
        report = monitor.report("release", None, target_ms=250)
        return {
            (sample.name, sample.labels.get("kind")): sample.value
            for family in text_string_to_metric_families(
                prometheus_metrics(report, ready).decode()
            )
            for sample in family.samples
        }

    def test_exposes_zero_counts_and_omits_unmeasured_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            samples = self.samples(Monitor(Path(temporary)), ready=False)
        self.assertEqual(samples["tiny_imagenet_ready", None], 0)
        self.assertEqual(samples["tiny_imagenet_window_requests", None], 0)
        self.assertEqual(samples["tiny_imagenet_alert", "fallback_rate_high"], 0)
        self.assertNotIn(
            ("tiny_imagenet_window_request_duration_p95_seconds", None), samples
        )
        self.assertNotIn(("tiny_imagenet_window_feedback_accuracy", None), samples)

    def test_exports_window_units_and_alert_transitions(self):
        with tempfile.TemporaryDirectory() as temporary:
            monitor = Monitor(Path(temporary))
            monitor.record("failed", "release", "fallback", None, [0.5] * 3, 300)
            samples = self.samples(monitor, ready=True)
            self.assertEqual(samples["tiny_imagenet_ready", None], 1)
            self.assertEqual(samples["tiny_imagenet_window_requests", None], 1)
            self.assertEqual(
                samples["tiny_imagenet_window_request_duration_p95_seconds", None], 0.3
            )
            self.assertEqual(samples["tiny_imagenet_alert", "latency_high"], 1)
            self.assertEqual(samples["tiny_imagenet_alert", "fallback_rate_high"], 1)
            for index in range(20):
                request_id = f"ok-{index}"
                monitor.record(request_id, "release", "active", 0, [0.5] * 3, 10)
                monitor.feedback(request_id, 0)
            samples = self.samples(monitor, ready=True)
            self.assertEqual(
                samples["tiny_imagenet_window_request_duration_p95_seconds", None], 0.01
            )
            self.assertEqual(samples["tiny_imagenet_window_feedback_accuracy", None], 1)
            self.assertEqual(samples["tiny_imagenet_alert", "latency_high"], 0)
            self.assertEqual(samples["tiny_imagenet_alert", "fallback_rate_high"], 0)
