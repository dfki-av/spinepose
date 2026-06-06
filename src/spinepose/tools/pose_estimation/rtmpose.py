from typing import List, Tuple

import numpy as np

from ..base_tool import BaseTool
from .post_processings import get_simcc_maximum
from .pre_processings import bbox_xyxy2cs, top_down_affine


class RTMPose(BaseTool):
    """RTMPose keypoint estimator."""

    def __init__(
        self,
        onnx_model: str,
        model_input_size: tuple = (288, 384),
        mean: tuple = (123.675, 116.28, 103.53),
        std: tuple = (58.395, 57.12, 57.375),
        **kwargs,
    ):
        """Initializes the pose estimator.

        Args:
            onnx_model: Path to the ONNX model.
            model_input_size: Model input size as ``(width, height)``.
            mean: Channel-wise normalization mean.
            std: Channel-wise normalization standard deviation.
            **kwargs: Additional arguments forwarded to ``BaseTool``.
        """
        super().__init__(onnx_model, model_input_size, mean, std, **kwargs)

    def __call__(self, image: np.ndarray, bboxes: list = []):
        """Runs pose estimation for one or more bounding boxes.

        Args:
            image: Input image.
            bboxes: Bounding boxes in ``xyxy`` format.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Keypoints and confidence scores.
        """
        if len(bboxes) == 0:
            bboxes = [[0, 0, image.shape[1], image.shape[0]]]

        keypoints, scores = [], []
        for bbox in bboxes:
            img, center, scale = self.preprocess(image, bbox)
            outputs = self.inference(img)
            kpts, score = self.postprocess(outputs, center, scale)

            keypoints.append(kpts)
            scores.append(score)

        keypoints = np.concatenate(keypoints, axis=0)
        scores = np.concatenate(scores, axis=0)

        return keypoints, scores

    def preprocess(self, img: np.ndarray, bbox: list):
        """Preprocesses an image crop for RTMPose.

        Args:
            img: Input image.
            bbox: Target bounding box in ``xyxy`` format.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Image crop, center, and scale.
        """
        bbox = np.array(bbox)

        # get center and scale
        center, scale = bbox_xyxy2cs(bbox, padding=1.25)

        # do affine transformation
        resized_img, scale = top_down_affine(self.model_input_size, scale, center, img)
        # normalize image
        if self.mean is not None:
            self.mean = np.array(self.mean)
            self.std = np.array(self.std)
            resized_img = (resized_img - self.mean) / self.std

        return resized_img, center, scale

    def postprocess(
        self,
        outputs: List[np.ndarray],
        center: Tuple[int, int],
        scale: Tuple[int, int],
        simcc_split_ratio: float = 2.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Postprocesses RTMPose outputs.

        Args:
            outputs: Raw model outputs.
            center: Bounding-box center.
            scale: Bounding-box scale.
            simcc_split_ratio: SimCC coordinate split ratio.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Rescaled keypoints and scores.
        """
        # decode simcc
        simcc_x, simcc_y = outputs
        locs, scores = get_simcc_maximum(simcc_x, simcc_y)
        keypoints = locs / simcc_split_ratio

        # rescale keypoints
        keypoints = keypoints / self.model_input_size * scale
        keypoints = keypoints + center - scale / 2

        return keypoints, scores
