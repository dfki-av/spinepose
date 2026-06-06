import logging
from typing import Tuple

import numpy as np

from .object_detection import RFDETR, YOLOX
from .pose_estimation import RTMPose
from .utils.multithreading import concurrent_forloop
from .visualization import draw_skeleton


class BasePoseSolution:
    """Base class for single-frame pose estimation pipelines."""

    def __init__(
        self,
        metainfo: dict,
        config: dict,
        mode: str = "performance",
        detector: str = "yolox",
        **kwargs,
    ):
        """Initializes the pose solution.

        Args:
            metainfo: Skeleton metadata used for visualization and outputs.
            config: Model configuration mapping by mode.
            mode: Model preset to load.
            detector: Detector name to use.
            **kwargs: Additional arguments forwarded to the model tools.
        """
        self.metainfo = metainfo
        self.num_keypoints = len(metainfo["keypoint_info"])

        mode_config = config.get(mode)
        if mode_config is None:
            logging.warning(
                f"Mode '{mode}' is not supported by {self.__class__.__name__}. Falling back to 'lightweight' mode."
            )
            mode_config = config.get("lightweight")
            if mode_config is None:
                raise ValueError(
                    f"No supported mode found for {self.__class__.__name__}."
                )

        detector_map = {
            "yolox": YOLOX,
            "rfdetr": RFDETR,
        }
        detector_cls = detector_map.get(detector.lower())
        if detector_cls is None:
            raise ValueError(
                f"Unsupported detector '{detector}'. Choose from: {list(detector_map.keys())}."
            )

        # Initialize detection and pose models
        self.det_model = detector_cls(
            mode_config[f"det_{detector.lower()}"],
            model_input_size=mode_config[f"det_{detector.lower()}_input_size"],
            **kwargs,
        )
        self.pose_model = RTMPose(
            mode_config["pose"],
            model_input_size=mode_config["pose_input_size"],
            **kwargs,
        )

    def detect(self, image: np.ndarray) -> np.ndarray:
        """Runs person detection on an image.

        Args:
            image: Input image.

        Returns:
            np.ndarray: Detected bounding boxes.
        """
        return self.det_model(image)

    def estimate(
        self, image: np.ndarray, bboxes: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Runs pose estimation for a set of bounding boxes.

        Args:
            image: Input image.
            bboxes: Bounding boxes to estimate poses for.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Keypoints and confidence scores.
        """
        # Process each bounding box concurrently
        results = concurrent_forloop(
            lambda bbox: self.pose_model(image, bboxes=[bbox]),
            bboxes,
        )

        if len(results) == 0:
            # No bounding boxes detected
            keypoints = np.zeros((0, self.num_keypoints, 2))
            scores = np.zeros((0, self.num_keypoints))
            return keypoints, scores

        # Concatenate results
        keypoints, scores = zip(*results)
        keypoints = np.concatenate(keypoints, axis=0)
        scores = np.concatenate(scores, axis=0)

        # Postprocess the results
        return self.postprocess(keypoints, scores)

    def postprocess(
        self, keypoints: np.ndarray, scores: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Postprocesses predicted keypoints and scores.

        Args:
            keypoints: Predicted keypoints.
            scores: Predicted confidence scores.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Postprocessed keypoints and scores.
        """
        return keypoints, scores

    def visualize(
        self, image: np.ndarray, keypoints: np.ndarray, scores: np.ndarray
    ) -> np.ndarray:
        """Draws pose predictions on an image.

        Args:
            image: Input image.
            keypoints: Predicted keypoints.
            scores: Predicted confidence scores.

        Returns:
            np.ndarray: Annotated image.
        """
        scale = image.shape[1] / 800
        radius = int(4 * scale)
        line_width = int(2 * scale)
        return draw_skeleton(
            image,
            keypoints,
            scores,
            self.metainfo,
            radius=radius,
            line_width=line_width,
        )

    def __call__(
        self, image: np.ndarray, bboxes: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Runs the full single-frame pose pipeline.

        Args:
            image: Input image.
            bboxes: Optional precomputed bounding boxes.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Keypoints and confidence scores.
        """
        if bboxes is None:
            bboxes = self.detect(image)
        return self.estimate(image, bboxes)
