import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spinepose.tools.smoothing import KeypointSmoothing
from spinepose.tools.visualization import draw_bbox, draw_skeleton


class VisualizationTests(unittest.TestCase):
    def test_draw_bbox_modifies_image_pixels(self):
        image = np.zeros((20, 20, 3), dtype=np.uint8)
        output = draw_bbox(image.copy(), np.array([[2, 2, 10, 10]]), color=(0, 255, 0))

        self.assertGreater(output.sum(), 0)
        np.testing.assert_array_equal(
            output[2, 2], np.array([0, 255, 0], dtype=np.uint8)
        )

    def test_draw_skeleton_draws_keypoints_and_links(self):
        image = np.zeros((30, 30, 3), dtype=np.uint8)
        metainfo = {
            "keypoint_info": {
                0: {"id": 0, "name": "a", "color": [255, 0, 0]},
                1: {"id": 1, "name": "b", "color": [0, 255, 0]},
            },
            "skeleton_info": {
                0: {"link": ("a", "b"), "color": [0, 0, 255]},
            },
        }
        keypoints = np.array([[[5.0, 5.0], [20.0, 20.0]]], dtype=np.float32)
        scores = np.array([[0.9, 0.9]], dtype=np.float32)

        output = draw_skeleton(image.copy(), keypoints, scores, metainfo, kpt_thr=0.5)

        self.assertGreater(output.sum(), 0)


class SmoothingTests(unittest.TestCase):
    def test_keypoint_smoothing_returns_smoothed_xy_coordinates(self):
        smoother = KeypointSmoothing(num_keypoints=2, freq=30.0)
        keypoints = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

        output = smoother(keypoints)

        self.assertEqual(output.shape, (2, 2))
        np.testing.assert_allclose(output, keypoints, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
