import sys
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spinepose.pose_estimator import SpinePoseEstimator
from spinepose.pose_tracker import PoseTracker, compute_iou, pose_to_bbox


class TrackerUtilityTests(unittest.TestCase):
    def test_compute_iou_for_overlapping_boxes(self):
        iou = compute_iou([0.0, 0.0, 10.0, 10.0], [5.0, 5.0, 15.0, 15.0])
        self.assertAlmostEqual(iou, 25.0 / 175.0)

    def test_pose_to_bbox_expands_around_keypoints(self):
        keypoints = np.array([[0.0, 0.0], [10.0, 20.0]])
        bbox = pose_to_bbox(keypoints, expansion=1.0)

        np.testing.assert_array_equal(bbox, np.array([0.0, 0.0, 10.0, 20.0]))


class _FakeSolution:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.num_keypoints = 2
        self.det_model = True
        self.detect_calls = 0

    def detect(self, image):
        self.detect_calls += 1
        return np.array([[0.0, 0.0, 20.0, 20.0]])

    def estimate(self, image, bboxes):
        return (
            np.array([[[0.0, 0.0], [20.0, 20.0]]], dtype=np.float32),
            np.array([[0.9, 0.8]], dtype=np.float32),
        )

    def visualize(self, image, keypoints, scores):
        return image + 1


class PoseTrackerTests(unittest.TestCase):
    def test_tracker_forwards_constructor_arguments_to_solution(self):
        tracker = PoseTracker(
            _FakeSolution,
            mode="small",
            detector="yolox",
            model_version="v1",
            hardware_acceleration=False,
        )

        self.assertEqual(tracker.solution.kwargs["mode"], "small")
        self.assertEqual(tracker.solution.kwargs["detector"], "yolox")
        self.assertEqual(tracker.solution.kwargs["model_version"], "v1")
        self.assertFalse(tracker.solution.kwargs["hardware_acceleration"])

    def test_tracker_call_returns_estimated_keypoints_and_scores(self):
        tracker = PoseTracker(_FakeSolution, tracking=False)

        keypoints, scores = tracker(np.zeros((4, 4, 3), dtype=np.uint8))

        self.assertEqual(tracker.frame_cnt, 1)
        self.assertEqual(keypoints.shape, (1, 2, 2))
        self.assertEqual(scores.shape, (1, 2))

    def test_track_by_iou_reuses_existing_track_id(self):
        tracker = PoseTracker(_FakeSolution)
        tracker.bboxes_last_frame = [np.array([0.0, 0.0, 10.0, 10.0])]
        tracker.track_ids_last_frame = [42]

        track_id, max_iou = tracker.track_by_iou(np.array([1.0, 1.0, 11.0, 11.0]))

        self.assertEqual(track_id, 42)
        self.assertGreater(max_iou, tracker.tracking_thr)

    def test_visualize_delegates_to_solution(self):
        tracker = PoseTracker(_FakeSolution)
        image = np.zeros((2, 2, 3), dtype=np.uint8)

        np.testing.assert_array_equal(
            tracker.visualize(image, np.empty((0, 2, 2)), np.empty((0, 2))),
            np.ones((2, 2, 3), dtype=np.uint8),
        )


class SpinePoseEstimatorTests(unittest.TestCase):
    def test_resolve_model_name_maps_public_versions(self):
        estimator = object.__new__(SpinePoseEstimator)

        self.assertEqual(estimator._resolve_model_name("latest"), "simspine")
        self.assertEqual(estimator._resolve_model_name("v2"), "simspine")
        self.assertEqual(estimator._resolve_model_name("v1"), "spinetrack")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertEqual(estimator._resolve_model_name("unknown"), "simspine")
        self.assertEqual(len(caught), 1)

    def test_postprocess_applies_v1_spine_smoothing_only_for_v1(self):
        estimator = object.__new__(SpinePoseEstimator)
        keypoints = np.zeros((1, 37, 2), dtype=np.float32)
        scores = np.ones((1, 37), dtype=np.float32)

        estimator.version = "v2"
        out_keypoints, out_scores = estimator.postprocess(keypoints, scores)
        self.assertIs(out_keypoints, keypoints)
        self.assertIs(out_scores, scores)

        estimator.version = "v1"
        estimator._smooth_spine = lambda k, s: (k + 1, s)
        out_keypoints, out_scores = estimator.postprocess(keypoints, scores)
        np.testing.assert_array_equal(out_keypoints, keypoints + 1)
        self.assertIs(out_scores, scores)

    def test_constructor_resolves_model_version_before_base_initialization(self):
        with patch(
            "spinepose.pose_estimator.BasePoseSolution.__init__",
            return_value=None,
        ) as base_init:
            estimator = SpinePoseEstimator(
                mode="small", model_version="v1", detector="yolox"
            )

        self.assertEqual(estimator.version, "v1")
        _, config = base_init.call_args.args[:2]
        self.assertIn("spinetrack", config["small"]["pose"])
        self.assertEqual(base_init.call_args.kwargs["mode"], "small")
        self.assertEqual(base_init.call_args.kwargs["detector"], "yolox")


if __name__ == "__main__":
    unittest.main()
