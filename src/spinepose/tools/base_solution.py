import logging
from typing import Tuple

import numpy as np

from .object_detection import RFDETR, YOLOX
from .pose_estimation import RTMPose
from .utils.multithreading import concurrent_forloop
from .visualization import draw_skeleton


class BasePoseSolution:
    """
    Single-frame pose estimation solution.

    This class is responsible for:
      - Running the detection model (if available) to generate bounding boxes.
      - Running the pose estimation model on provided bounding boxes.
      - Optionally postprocessing the output keypoints and scores.

    The __call__ method accepts an image and, optionally, precomputed bounding boxes.
    If no bounding boxes are provided, detection is performed automatically.
    """

    def __init__(
        self,
        metainfo: dict,
        config: dict,
        mode: str = "performance",
        detector: str = "yolox",
        **kwargs,
    ):
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
        """Run detection to get bounding boxes from the image."""
        return self.det_model(image)

    def estimate(
        self, image: np.ndarray, bboxes: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run pose estimation on the image using the given bounding boxes.

        Returns:
            A tuple (keypoints, scores) after postprocessing.
        """
        # Process each bounding box concurrently
        results = concurrent_forloop(
            lambda bbox: self.pose_model(image, bboxes=[bbox]),
            bboxes,
        )

        if len(results) == 0:
            # No bounding boxes detected
            return np.zeros((0, self.num_keypoints, 3)), np.zeros(
                (0, self.num_keypoints)
            )

        # Concatenate results
        keypoints, scores = zip(*results)
        keypoints = np.concatenate(keypoints, axis=0)
        scores = np.concatenate(scores, axis=0)

        # Postprocess the results
        return self.postprocess(keypoints, scores)

    def postprocess(
        self, keypoints: np.ndarray, scores: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Optionally postprocess the keypoints and scores.

        Override this method if additional postprocessing is needed.
        """
        return keypoints, scores

    def visualize(
        self, image: np.ndarray, keypoints: np.ndarray, scores: np.ndarray
    ) -> np.ndarray:
        """
        Visualize the keypoints and scores on the image.

        Override this method if custom visualization is needed.
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
        """
        Run the full single-frame pipeline.

        If `bboxes` is provided, it is used directly; otherwise, detection is performed.
        """
        if bboxes is None:
            bboxes = self.detect(image)
        return self.estimate(image, bboxes)
