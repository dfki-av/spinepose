# Code modified from https://github.com/IDEA-Research/DWPose/blob/opencv_onnx/ControlNet-v1-1-nightly/annotator/dwpose/cv_ox_det.py  # noqa
from __future__ import annotations

import cv2
import numpy as np

from ..base_tool import BaseTool
from .post_processings import multiclass_nms


class YOLOX(BaseTool):
    def __init__(
        self,
        onnx_model: str,
        model_input_size: tuple[int, int] = (640, 640),
        nms_thr: float = 0.45,
        score_thr: float = 0.7,
        backend: str = "onnxruntime",
        device: str = "cpu",
    ) -> None:
        """Initializes the YOLOX detector.

        Args:
            onnx_model: Path to the ONNX model.
            model_input_size: Model input size as ``(height, width)``.
            nms_thr: Non-maximum suppression threshold.
            score_thr: Minimum score required to keep a detection.
            backend: Inference backend name.
            device: Inference device name.
        """
        super().__init__(onnx_model, model_input_size, backend=backend, device=device)
        self.nms_thr = nms_thr
        self.score_thr = score_thr

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Runs detection and returns boxes only."""
        return self.predict(image)["xyxy"]

    def predict(self, image: np.ndarray) -> dict[str, np.ndarray]:
        """Runs detection and returns boxes with confidence scores."""
        image, ratio = self.preprocess(image)
        outputs = self.inference(image)[0]
        return self.postprocess(outputs, ratio, return_scores=True)

    def preprocess(self, img: np.ndarray) -> tuple[np.ndarray, float]:
        """Pads and resizes an image for YOLOX inference.

        Args:
            img: Input image array.

        Returns:
            tuple[np.ndarray, float]: Preprocessed image and resize ratio.
        """
        if len(img.shape) == 3:
            padded_img = (
                np.ones(
                    (self.model_input_size[0], self.model_input_size[1], 3),
                    dtype=np.uint8,
                )
                * 114
            )
        else:
            padded_img = np.ones(self.model_input_size, dtype=np.uint8) * 114

        ratio = min(
            self.model_input_size[0] / img.shape[0],
            self.model_input_size[1] / img.shape[1],
        )
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * ratio), int(img.shape[0] * ratio)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        padded_shape = (int(img.shape[0] * ratio), int(img.shape[1] * ratio))
        padded_img[: padded_shape[0], : padded_shape[1]] = resized_img

        return padded_img, ratio

    def postprocess(
        self,
        outputs: np.ndarray,
        ratio: float = 1.0,
        return_scores: bool = False,
    ) -> np.ndarray | dict[str, np.ndarray]:
        """Converts model outputs into image-space detections.

        Args:
            outputs: Raw model outputs.
            ratio: Resize ratio used during preprocessing.
            return_scores: When ``True``, returns boxes and confidences.

        Returns:
            np.ndarray | dict[str, np.ndarray]: Boxes only or a detection payload.
        """

        final_boxes = np.empty((0, 4), dtype=np.float32)
        final_scores = np.empty((0,), dtype=np.float32)

        if outputs.shape[-1] == 4:
            # onnx without nms module

            grids = []
            expanded_strides = []
            strides = [8, 16, 32]

            hsizes = [self.model_input_size[0] // stride for stride in strides]
            wsizes = [self.model_input_size[1] // stride for stride in strides]

            for hsize, wsize, stride in zip(hsizes, wsizes, strides):
                xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
                grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
                grids.append(grid)
                shape = grid.shape[:2]
                expanded_strides.append(np.full((*shape, 1), stride))

            grids = np.concatenate(grids, 1)
            expanded_strides = np.concatenate(expanded_strides, 1)
            outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
            outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides

            predictions = outputs[0]
            boxes = predictions[:, :4]
            scores = predictions[:, 4:5] * predictions[:, 5:]

            boxes_xyxy = np.ones_like(boxes)
            boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
            boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
            boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
            boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
            boxes_xyxy /= ratio
            dets, keep = multiclass_nms(
                boxes_xyxy, scores, nms_thr=self.nms_thr, score_thr=self.score_thr
            )
            if dets is not None:
                pack_dets = (dets[:, :4], dets[:, 4], dets[:, 5])
                final_boxes, final_scores, final_cls_inds = pack_dets
                isscore = final_scores > 0.3
                iscat = final_cls_inds == 0
                isbbox = [i and j for (i, j) in zip(isscore, iscat)]
                final_boxes = final_boxes[isbbox]
                final_scores = final_scores[isbbox]

        elif outputs.shape[-1] == 5:
            # onnx contains nms module

            pack_dets = (outputs[0, :, :4], outputs[0, :, 4])
            final_boxes, final_scores = pack_dets
            final_boxes /= ratio
            isscore = final_scores > 0.3
            isbbox = list(isscore)
            final_boxes = final_boxes[isbbox]
            final_scores = final_scores[isbbox]

        if return_scores:
            return {
                "xyxy": final_boxes,
                "confidence": final_scores,
            }
        return final_boxes
