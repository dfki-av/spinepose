import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spinepose.tools.pose_estimation.post_processings import get_simcc_maximum
from spinepose.tools.pose_estimation.pre_processings import (
    _get_3rd_point,
    _rotate_point,
    bbox_xyxy2cs,
    get_warp_matrix,
    top_down_affine,
)
from spinepose.tools.pose_estimation.rtmpose import RTMPose


class PosePreProcessingTests(unittest.TestCase):
    def test_bbox_xyxy2cs_handles_single_and_batched_boxes(self):
        center, scale = bbox_xyxy2cs(np.array([0.0, 2.0, 10.0, 22.0]), padding=1.25)
        np.testing.assert_allclose(center, np.array([5.0, 12.0]))
        np.testing.assert_allclose(scale, np.array([12.5, 25.0]))

        centers, scales = bbox_xyxy2cs(
            np.array([[0.0, 0.0, 10.0, 10.0], [10.0, 10.0, 20.0, 30.0]])
        )
        self.assertEqual(centers.shape, (2, 2))
        self.assertEqual(scales.shape, (2, 2))

    def test_rotate_point_and_third_point_helpers(self):
        np.testing.assert_allclose(
            _rotate_point(np.array([1.0, 0.0]), np.pi / 2),
            np.array([0.0, 1.0]),
            atol=1e-7,
        )
        np.testing.assert_array_equal(
            _get_3rd_point(np.array([1.0, 1.0]), np.array([1.0, 0.0])),
            np.array([0.0, 0.0]),
        )

    def test_get_warp_matrix_maps_center_to_output_center(self):
        warp = get_warp_matrix(
            center=np.array([10.0, 20.0]),
            scale=np.array([20.0, 20.0]),
            rot=0,
            output_size=(40, 40),
        )
        mapped = cv2.transform(np.array([[[10.0, 20.0]]], dtype=np.float32), warp)

        np.testing.assert_allclose(mapped[0, 0], np.array([20.0, 20.0]), atol=1e-5)

    def test_top_down_affine_returns_warped_image_and_scale(self):
        img = np.zeros((20, 20, 3), dtype=np.uint8)
        warped, scale = top_down_affine(
            input_size=(8, 8),
            bbox_scale=np.array([10.0, 20.0]),
            bbox_center=np.array([10.0, 10.0]),
            img=img,
        )

        self.assertEqual(warped.shape, (8, 8, 3))
        self.assertEqual(scale.shape, (2,))


class PosePostProcessingTests(unittest.TestCase):
    def test_get_simcc_maximum_returns_peak_locations_and_scores(self):
        simcc_x = np.array([[[0.1, 0.9, 0.2], [0.0, 0.1, 0.8]]], dtype=np.float32)
        simcc_y = np.array([[[0.3, 0.4], [0.7, 0.1]]], dtype=np.float32)

        locs, vals = get_simcc_maximum(simcc_x, simcc_y)

        np.testing.assert_array_equal(locs, np.array([[[1.0, 1.0], [2.0, 0.0]]]))
        np.testing.assert_allclose(vals, np.array([[0.65, 0.75]], dtype=np.float32))

    def test_rtmpose_postprocess_rescales_simcc_coordinates(self):
        pose = object.__new__(RTMPose)
        pose.model_input_size = np.array([4.0, 4.0])
        simcc_x = np.array([[[0.0, 0.1, 1.0, 0.2]]], dtype=np.float32)
        simcc_y = np.array([[[0.0, 1.0, 0.1, 0.2]]], dtype=np.float32)

        keypoints, scores = pose.postprocess(
            [simcc_x, simcc_y],
            center=np.array([10.0, 20.0]),
            scale=np.array([8.0, 8.0]),
            simcc_split_ratio=2.0,
        )

        np.testing.assert_allclose(keypoints, np.array([[[8.0, 17.0]]]), atol=1e-6)
        np.testing.assert_allclose(scores, np.array([[1.0]], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
