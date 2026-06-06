import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spinepose import inference


class InferenceHelperTests(unittest.TestCase):
    def test_file_type_helpers_detect_supported_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "frame.jpg"
            video = Path(tmp) / "clip.mp4"
            text = Path(tmp) / "notes.txt"
            image.write_text("")
            video.write_text("")
            text.write_text("")

            self.assertTrue(inference._is_image(str(image)))
            self.assertTrue(inference._is_video(str(video)))
            self.assertFalse(inference._is_image(str(text)))
            self.assertFalse(inference._is_video(str(text)))

    def test_write_frame_outputs_openpose_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "frame.json"
            keypoints = np.array([[[1.0, 2.0, 0.9], [3.0, 4.0, 0.8]]])

            inference._write_frame(keypoints, output)

            payload = json.loads(output.read_text())
            self.assertEqual(payload["version"], 1.0)
            self.assertEqual(
                payload["people"][0]["pose_keypoints_2d"],
                [1.0, 2.0, 0.9, 3.0, 4.0, 0.8],
            )


class _FakeEstimator:
    SPINE_IDS = [0, 2]
    init_args = None
    init_kwargs = None

    def __init__(self, *args, **kwargs):
        _FakeEstimator.init_args = args
        _FakeEstimator.init_kwargs = kwargs

    def __call__(self, image):
        return (
            np.array(
                [
                    [
                        [1.0, 2.0, 0.0],
                        [3.0, 4.0, 0.0],
                        [5.0, 6.0, 0.0],
                    ]
                ],
                dtype=np.float32,
            ),
            np.array([[0.9, 0.1, 0.8]], dtype=np.float32),
        )

    def visualize(self, image, keypoints, scores):
        return image


class _FakeVideoCapture:
    def __init__(self, path):
        self.path = path
        self.frames = [np.zeros((2, 2, 3), dtype=np.uint8)]

    def isOpened(self):
        return True

    def get(self, prop):
        return 25.0

    def read(self):
        if self.frames:
            return True, self.frames.pop(0)
        return False, None

    def release(self):
        self.released = True


class _FakePoseTracker:
    init_args = None
    init_kwargs = None

    def __init__(self, *args, **kwargs):
        _FakePoseTracker.init_args = args
        _FakePoseTracker.init_kwargs = kwargs
        self.solution = Mock(SPINE_IDS=[0])

    def __call__(self, image):
        return (
            np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32),
            np.array([[0.9, 0.1]], dtype=np.float32),
        )

    def visualize(self, image, keypoints, scores):
        return image


class InferenceAPITests(unittest.TestCase):
    def test_infer_image_forwards_runtime_options_and_filters_spine_only(self):
        with (
            patch.object(inference, "SpinePoseEstimator", _FakeEstimator),
            patch.object(
                inference.cv2,
                "imread",
                return_value=np.zeros((4, 4, 3), dtype=np.uint8),
            ),
            patch.object(inference.cv2, "imwrite", return_value=True),
        ):
            results = inference.infer_image(
                "input.jpg",
                mode="small",
                spine_only=True,
                vis_path="output.jpg",
                model_version="v1",
                detector="yolox",
                hardware_acceleration=False,
                mixed_precision=True,
            )

        self.assertEqual(_FakeEstimator.init_args, ("small",))
        self.assertEqual(_FakeEstimator.init_kwargs["detector"], "yolox")
        self.assertEqual(_FakeEstimator.init_kwargs["model_version"], "v1")
        self.assertFalse(_FakeEstimator.init_kwargs["hardware_acceleration"])
        self.assertTrue(_FakeEstimator.init_kwargs["mixed_precision"])
        self.assertEqual(results.shape, (1, 2, 4))
        np.testing.assert_array_equal(
            results[0, :, 3], np.array([0.9, 0.8], dtype=np.float32)
        )

    def test_infer_video_forwards_runtime_options_and_returns_frame_results(self):
        with (
            patch.object(inference, "PoseTracker", _FakePoseTracker),
            patch.object(
                inference.cv2,
                "VideoCapture",
                _FakeVideoCapture,
            ),
            patch.object(inference, "_imshow"),
            patch.object(
                inference.cv2,
                "waitKey",
                return_value=-1,
            ),
            patch.object(
                inference.cv2,
                "destroyAllWindows",
            ),
        ):
            results = inference.infer_video(
                "input.mp4",
                mode="medium",
                spine_only=True,
                use_smoothing=False,
                model_version="v2",
                detector="rfdetr",
                hardware_acceleration=False,
                mixed_precision=True,
            )

        self.assertEqual(_FakePoseTracker.init_kwargs["mode"], "medium")
        self.assertEqual(_FakePoseTracker.init_kwargs["detector"], "rfdetr")
        self.assertFalse(_FakePoseTracker.init_kwargs["smoothing"])
        self.assertFalse(_FakePoseTracker.init_kwargs["hardware_acceleration"])
        self.assertTrue(_FakePoseTracker.init_kwargs["mixed_precision"])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].shape, (1, 1, 3))


if __name__ == "__main__":
    unittest.main()
