from __future__ import annotations

import warnings
from copy import deepcopy

import numpy as np

from .metainfo import metainfo
from .tools.base_solution import BasePoseSolution
from .tools.pose_lifting import DefaultPoseLifter


class SpinePoseEstimator(BasePoseSolution):
    """Spine-aware pose estimator built on top of the base solution."""

    MODE = {
        "xlarge": {
            "det_yolox": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_x_8xb8-300e_humanart-a39d44ed.zip",
            "det_yolox_input_size": (640, 640),
            "det_rfdetr": "https://huggingface.co/saifkhichi96/opendetect/resolve/main/rfdetr/rfdetr_l_v142_704x704.onnx",
            "det_rfdetr_input_size": (704, 704),
            "pose": "https://huggingface.co/dfki-av/spinepose/resolve/main/spinepose-x_32xb128-10e_spinetrack-384x288.onnx",
            "pose_input_size": (288, 384),
        },
        "large": {
            "det_yolox": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_x_8xb8-300e_humanart-a39d44ed.zip",
            "det_yolox_input_size": (640, 640),
            "det_rfdetr": "https://huggingface.co/saifkhichi96/opendetect/resolve/main/rfdetr/rfdetr_m_v142_576x576.onnx",
            "det_rfdetr_input_size": (576, 576),
            "pose": "https://huggingface.co/dfki-av/spinepose/resolve/main/spinepose-l_32xb256-10e_%s-256x192.onnx",
            "pose_input_size": (192, 256),
        },
        "medium": {
            "det_yolox": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip",
            "det_yolox_input_size": (640, 640),
            "det_rfdetr": "https://huggingface.co/saifkhichi96/opendetect/resolve/main/rfdetr/rfdetr_s_v142_512x512.onnx",
            "det_rfdetr_input_size": (512, 512),
            "pose": "https://huggingface.co/dfki-av/spinepose/resolve/main/spinepose-m_32xb256-10e_%s-256x192.onnx",
            "pose_input_size": (192, 256),
        },
        "small": {
            "det_yolox": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_tiny_8xb8-300e_humanart-6f3252f9.zip",
            "det_yolox_input_size": (416, 416),
            "det_rfdetr": "https://huggingface.co/saifkhichi96/opendetect/resolve/main/rfdetr/rfdetr_n_v142_384x384.onnx",
            "det_rfdetr_input_size": (384, 384),
            "pose": "https://huggingface.co/dfki-av/spinepose/resolve/main/spinepose-s_32xb256-10e_%s-256x192.onnx",
            "pose_input_size": (192, 256),
        },
    }

    LIFTING_MODEL_CONFIG = {
        "onnx_model": "https://huggingface.co/dfki-av/spinepose/resolve/main/simspine/3d/onnx/spinepose_lifter_8xb256-90e_simspine-37x2.onnx",
        "model_input_size": (37, 2),
        "mean": (0.0, 0.0),
        "std": (1.0, 1.0),
    }

    SPINE_IDS = [19, 26, 27, 28, 29, 30, 18, 35, 36]

    def __init__(
        self,
        mode: str = "large",
        model_version: str = "latest",
        detector: str = "rfdetr",
        enable_lifting: bool = False,
        lifting_thr: float = 0.5,
        camera_intrinsics: np.ndarray | None = None,
        camera_field_of_view: float | None = 84.0,
        estimate_metric_scale: bool = True,
        primary_subject_height: float = 1.84,
        **kwargs,
    ):
        """Initializes the SpinePose estimator.

        Args:
            mode: Model preset to load.
            model_version: Model version to use.
            detector: Detector name to use.
            enable_lifting: Whether to enable 2D-to-3D pose lifting.
            lifting_thr: Minimum keypoint score used as visible for lifting.
            camera_intrinsics: Camera intrinsic matrix for 3D lifting.
            camera_field_of_view: Diagonal field of view used to estimate intrinsics.
            estimate_metric_scale: Whether to estimate metric scale from subject height.
            primary_subject_height: Primary subject height in meters.
            **kwargs: Additional arguments forwarded to the base solution.
        """
        model_name = self._resolve_model_name(model_version)
        config = deepcopy(self.MODE)
        for key in config:
            pose_model = config[key]["pose"]
            if "%s" in pose_model:
                config[key]["pose"] = pose_model % model_name

        super().__init__(
            metainfo,
            config,
            mode=mode,
            detector=detector,
            **kwargs,
        )
        self.version = model_version

        self._lifting_model = None
        if enable_lifting:
            self._lifting_model = DefaultPoseLifter(
                **self.LIFTING_MODEL_CONFIG,
                lifting_thr=lifting_thr,
                camera_intrinsics=camera_intrinsics,
                camera_field_of_view=camera_field_of_view,
                estimate_metric_scale=estimate_metric_scale,
                primary_subject_height=primary_subject_height,
                **kwargs,
            )

    @property
    def camera_intrinsics(self) -> np.ndarray | None:
        """Camera intrinsic matrix used for 3D lifting."""
        if self._lifting_model is not None:
            return self._lifting_model.camera_intrinsics
        return None

    def _resolve_model_name(self, model_version: str) -> str:
        """Maps a public model version to the underlying model family.

        Args:
            model_version: User-facing model version string.

        Returns:
            str: Internal model family name.
        """
        if model_version in ["latest", "v2"]:
            return "simspine"
        elif model_version == "v1":
            return "spinetrack"
        else:
            warnings.warn(
                f"Unknown model version '{model_version}', defaulting to 'simspine'"
            )
            return "simspine"

    def _smooth_spine(
        self, keypoints: np.ndarray, scores: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Applies a simple smoothing rule to spine keypoints.

        Args:
            keypoints: Predicted keypoints.
            scores: Predicted confidence scores.

        Returns:
            tuple: Smoothed keypoints and confidence scores.
        """
        spine_ids = self.SPINE_IDS[:9]
        spine_keypoints = keypoints[:, spine_ids]
        spine_scores = scores[:, spine_ids]

        # Smooth by averaging consecutive points
        for i in range(1, len(spine_keypoints[0]) - 1):
            spine_keypoints[:, i] = (
                spine_keypoints[:, i - 1] + spine_keypoints[:, i + 1]
            ) / 2

        # Replace in global keypoints
        keypoints[:, spine_ids] = spine_keypoints
        scores[:, spine_ids] = spine_scores

        return keypoints, scores

    def postprocess(
        self, keypoints: np.ndarray, scores: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Postprocesses predicted keypoints and scores.

        Args:
            keypoints: Predicted keypoints.
            scores: Predicted confidence scores.

        Returns:
            tuple: Postprocessed keypoints and confidence scores.
        """
        if self.version == "v1":
            keypoints, scores = self._smooth_spine(keypoints, scores)

        return keypoints, scores

    def estimate(
        self, image: np.ndarray, bboxes: np.ndarray
    ) -> (
        tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray | None]
    ):
        """Runs pose estimation for a set of bounding boxes.

        Args:
            image: Input image.
            bboxes: Bounding boxes to estimate poses for.

        Returns:
            tuple: Keypoints and scores, with camera-space 3D keypoints when
            lifting is enabled.
        """
        keypoints, scores = super().estimate(image, bboxes)
        if self._lifting_model is None:
            return keypoints, scores

        if keypoints.shape[0] == 0:
            keypoints_3d = np.zeros((0, self.num_keypoints, 3))
        else:
            self._lifting_model.estimate_intrinsics(
                width=image.shape[1],
                height=image.shape[0],
            )
            keypoints_3d = self._lifting_model(keypoints, scores)
        return keypoints, scores, keypoints_3d

    def __call__(
        self, image: np.ndarray, bboxes: np.ndarray | None = None
    ) -> (
        tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray | None]
    ):
        """Runs the full SpinePose estimation pipeline.

        Args:
            image: Input image.
            bboxes: Optional precomputed bounding boxes.

        Returns:
            tuple: Keypoints and scores, with camera-space 3D keypoints when
            lifting is enabled.
        """
        return super().__call__(image, bboxes)

    def close(self) -> None:
        """Releases model resources held by the estimator."""
        super().close()
        close = getattr(self._lifting_model, "close", None)
        if close is not None:
            close()
