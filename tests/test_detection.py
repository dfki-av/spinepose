import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spinepose.tools.object_detection.post_processings import multiclass_nms, nms
from spinepose.tools.object_detection.rfdetr import RFDETR
from spinepose.tools.object_detection.yolox import YOLOX


class DetectionPostProcessingTests(unittest.TestCase):
    def test_nms_suppresses_overlapping_boxes(self):
        boxes = np.array(
            [
                [0.0, 0.0, 10.0, 10.0],
                [1.0, 1.0, 11.0, 11.0],
                [30.0, 30.0, 40.0, 40.0],
            ]
        )
        scores = np.array([0.9, 0.8, 0.7])

        self.assertEqual(nms(boxes, scores, nms_thr=0.5), [0, 2])

    def test_multiclass_nms_returns_class_indices(self):
        boxes = np.array([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]])
        scores = np.array([[0.9, 0.1], [0.2, 0.95]])

        dets, keep = multiclass_nms(boxes, scores, nms_thr=0.5, score_thr=0.3)

        self.assertIsNotNone(keep)
        self.assertEqual(dets.shape, (2, 6))
        self.assertEqual(set(dets[:, 5].astype(int)), {0, 1})


class YOLOXTests(unittest.TestCase):
    def test_preprocess_pads_and_returns_resize_ratio(self):
        detector = object.__new__(YOLOX)
        detector.model_input_size = (8, 8)
        img = np.ones((4, 2, 3), dtype=np.uint8) * 255

        padded, ratio = detector.preprocess(img)

        self.assertEqual(padded.shape, (8, 8, 3))
        self.assertEqual(ratio, 2.0)
        np.testing.assert_array_equal(
            padded[:8, :4], np.ones((8, 4, 3), dtype=np.uint8) * 255
        )
        self.assertTrue(np.all(padded[:, 4:] == 114))

    def test_postprocess_nms_export_returns_scores_when_requested(self):
        detector = object.__new__(YOLOX)
        outputs = np.array(
            [
                [
                    [10.0, 20.0, 30.0, 40.0, 0.9],
                    [50.0, 60.0, 70.0, 80.0, 0.2],
                ]
            ],
            dtype=np.float32,
        )

        result = detector.postprocess(outputs, ratio=2.0, return_scores=True)

        np.testing.assert_allclose(
            result["xyxy"],
            np.array([[5.0, 10.0, 15.0, 20.0]], dtype=np.float32),
        )
        np.testing.assert_allclose(
            result["confidence"], np.array([0.9], dtype=np.float32)
        )


class RFDETRTests(unittest.TestCase):
    def _detector(self):
        detector = object.__new__(RFDETR)
        detector.input_size = (4, 4)
        detector._imagenet_inv_std = (1.0 / detector.imagenet_std).astype(np.float32)
        detector.score_thr = 0.1
        detector.num_select = 4
        detector.class_ids_np = np.array([0], dtype=np.int64)
        return detector

    def test_preprocess_validates_rgb_input_and_tracks_original_size(self):
        detector = self._detector()
        img = np.ones((2, 3, 3), dtype=np.uint8) * 255

        processed, target_sizes = detector.preprocess(img)

        self.assertEqual(processed.shape, (4, 4, 3))
        np.testing.assert_array_equal(
            target_sizes, np.array([[2, 3]], dtype=np.float32)
        )
        with self.assertRaises(ValueError):
            detector.preprocess(np.ones((2, 3), dtype=np.uint8))

    def test_postprocess_returns_scaled_boxes_and_scores(self):
        detector = self._detector()
        out_bbox = np.array([[[0.5, 0.5, 0.2, 0.4]]], dtype=np.float32)
        out_logits = np.array([[[-8.0, 8.0]]], dtype=np.float32)
        target_sizes = np.array([[100.0, 200.0]], dtype=np.float32)

        result = detector.postprocess(
            (out_bbox, out_logits), target_sizes, return_scores=True
        )

        np.testing.assert_allclose(
            result["xyxy"],
            np.array([[80.0, 30.0, 120.0, 70.0]], dtype=np.float32),
            rtol=1e-5,
        )
        self.assertGreater(result["confidence"][0], 0.99)

    def test_postprocess_returns_empty_payload_for_invalid_outputs(self):
        detector = self._detector()
        result = detector.postprocess(
            (np.array([1.0]),), np.array([[1.0, 1.0]]), return_scores=True
        )

        self.assertEqual(result["xyxy"].shape, (0, 4))
        self.assertEqual(result["confidence"].shape, (0,))


if __name__ == "__main__":
    unittest.main()
