import json
from pathlib import Path
import tempfile
import unittest

from config import Config
from monitoring import Monitor, Observation
from reporting import SessionReport, evaluate_events, load_annotations


class ReportTests(unittest.TestCase):
    def test_event_matching_and_false_alarms(self):
        labels = {'duration_seconds': 60, 'events': [
            {'label': 'drowsy', 'start': 10, 'end': 15}, {'label': 'drowsy', 'start': 30, 'end': 35}]}
        predictions = [{'label': 'drowsy', 'start': 12, 'end': 16}, {'label': 'drowsy', 'start': 40, 'end': 42}]
        result = evaluate_events(predictions, labels)
        self.assertEqual(result['detected_events'], 1)
        self.assertEqual(result['missed_events'], 1)
        self.assertEqual(result['false_alert_events'], 1)
        self.assertEqual(result['mean_detection_delay_seconds'], 2)
        self.assertEqual(result['event_recall'], 0.5)
        self.assertEqual(result['false_alerts_per_hour'], 60)

    def test_single_long_alarm_cannot_match_multiple_events(self):
        labels = {'duration_seconds': 60, 'events': [
            {'label': 'drowsy', 'start': 10, 'end': 15}, {'label': 'drowsy', 'start': 30, 'end': 35}]}
        result = evaluate_events([{'label': 'drowsy', 'start': 11, 'end': 40}], labels)
        self.assertEqual(result['detected_events'], 1)

    def test_no_labels_does_not_invent_accuracy(self):
        result = evaluate_events([], {'duration_seconds': 10, 'events': []})
        self.assertIsNone(result['event_precision'])
        self.assertIsNone(result['event_recall'])

    def test_reject_invalid_annotation_intervals(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'labels.json'
            for event in ({'label': 'drowsy', 'start': 2, 'end': 1},
                          {'label': 'drowsy', 'start': 0, 'end': 11},
                          {'label': 'unknown', 'start': 0, 'end': 1}):
                path.write_text(json.dumps({'duration_seconds': 10, 'events': [event]}))
                with self.assertRaises(ValueError):
                    load_annotations(path)

    def test_session_files_and_incomplete_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            report = SessionReport(directory, Config(), {'backend': 'synthetic'})
            m = Monitor(calibrate=False)
            for t in (0, 0.1, 0.2):
                obs = Observation(t)
                report.add(m.update(obs), obs, 'AWAKE', 10)
            output = report.finish(0.3, {'duration_seconds': 10, 'events': []}, 'stopped_by_user')
            summary = json.loads((output / 'summary.json').read_text())
            events = json.loads((output / 'events.json').read_text())
            self.assertEqual(summary['frames'], 3)
            self.assertIsNone(summary['evaluation'])
            self.assertIn('evaluation_error', summary)
            self.assertEqual(events[0]['label'], 'visibility')
            self.assertAlmostEqual(events[0]['end'], 0.3)
            self.assertTrue((output / 'measurements.csv').exists())


if __name__ == '__main__':
    unittest.main()
