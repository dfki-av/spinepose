from __future__ import annotations

import logging

import numpy as np

from ..base_tool import BaseTool
from .post_processings import (
    estimate_camera_intrinsics,
    estimate_metric_scale,
    estimate_root_translation,
)


class DefaultPoseLifter(BaseTool):
    """2D-to-3D pose lifter for SpinePose."""

    def __init__(
        self,
        onnx_model: str,
        model_input_size: tuple[int, int] = (37, 2),
        mean: tuple[float, float] = (0.0, 0.0),
        std: tuple[float, float] = (1.0, 1.0),
        lifting_thr: float = 0.5,
        camera_intrinsics: np.ndarray | None = None,
        camera_field_of_view: float | None = 84.0,
        estimate_metric_scale: bool = True,
        primary_subject_height: float = 1.84,
        **kwargs,
    ) -> None:
        """Initializes the pose lifter.

        Args:
            onnx_model: Path to the ONNX model.
            model_input_size: Model input size as ``(K, 2)``.
            mean: Coordinate normalization mean.
            std: Coordinate normalization standard deviation.
            lifting_thr: Minimum keypoint score treated as visible.
            camera_intrinsics: Camera intrinsic matrix.
            camera_field_of_view: Diagonal field of view used to estimate intrinsics.
            estimate_metric_scale: Whether to estimate metric scale from height.
            primary_subject_height: Primary subject height in meters.
            **kwargs: Additional arguments forwarded to ``BaseTool``.
        """
        super().__init__(onnx_model, model_input_size, mean, std, **kwargs)
        self.lifting_thr = lifting_thr
        self.camera_intrinsics = camera_intrinsics
        self.camera_field_of_view = camera_field_of_view
        self.estimate_metric_scale = estimate_metric_scale
        self.primary_subject_height = primary_subject_height
        self.metric_scale = 1.0
        self.metric_scale_estimated = False

        if camera_intrinsics is None:
            logging.info(
                "Camera intrinsics not provided; estimating them from image size."
            )

        if estimate_metric_scale:
            logging.info(
                f"Metric scale estimation enabled with subject height "
                f"{primary_subject_height:.2f} m."
            )

    def reset(self) -> None:
        """Resets stateful metric scale estimation."""
        self.metric_scale = 1.0
        self.metric_scale_estimated = False

    def estimate_intrinsics(
        self, width: int, height: int, override: bool = False
    ) -> np.ndarray:
        """Estimates camera intrinsics from the image size if not already set.

        Args:
            width: Image width in pixels.
            height: Image height in pixels.
            override: Whether to replace existing intrinsics.

        Returns:
            np.ndarray: Camera intrinsic matrix.
        """
        if self.camera_intrinsics is not None and not override:
            return self.camera_intrinsics

        self.camera_intrinsics = estimate_camera_intrinsics(
            width,
            height,
            dfov=self.camera_field_of_view,
        )
        logging.info(f"Estimated camera intrinsics from {width}x{height} image.")
        return self.camera_intrinsics

    def preprocess(
        self, keypoints: np.ndarray, scores: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Normalizes 2D keypoints and computes visibility masks.

        Args:
            keypoints: 2D keypoints with shape ``(N, K, 2)``.
            scores: Keypoint scores with shape ``(N, K)``.

        Returns:
            tuple: Normalized 2D keypoints and visibility mask.
        """
        intrinsics = self.camera_intrinsics

        fx = intrinsics[0, 0]
        fy = intrinsics[1, 1]
        cx = intrinsics[0, 2]
        cy = intrinsics[1, 2]
        xy_norm = np.concatenate(
            [
                (keypoints[..., 0:1] - cx) / fx,
                (keypoints[..., 1:2] - cy) / fy,
            ],
            axis=-1,
        )

        visible = scores >= self.lifting_thr
        xy_norm = xy_norm * visible[:, :, np.newaxis]
        if self.mean is not None and self.std is not None:
            xy_norm = (xy_norm - self.mean) / self.std

        return xy_norm, visible

    def __call__(self, keypoints: np.ndarray, scores: np.ndarray) -> np.ndarray | None:
        """Lifts 2D keypoints into camera-space 3D keypoints.

        Args:
            keypoints: 2D keypoints with shape ``(N, K, 2)``.
            scores: Keypoint scores with shape ``(N, K)``.

        Returns:
            np.ndarray | None: 3D keypoints with shape ``(N, K, 3)``, or
            ``None`` if camera intrinsics are unavailable.
        """
        if self.camera_intrinsics is None:
            logging.warning("Camera intrinsics not set; skipping pose lifting.")
            return None
        if keypoints.shape[0] == 0:
            return np.zeros((0, keypoints.shape[1], 3), dtype=np.float32)

        xy_norm, visible = self.preprocess(keypoints, scores)

        keypoints_3d_rel = self.inference(xy_norm)[0]
        keypoints_3d_rel = keypoints_3d_rel - keypoints_3d_rel[:, 19:20, :]

        if self.estimate_metric_scale and not self.metric_scale_estimated:
            self.metric_scale = estimate_metric_scale(
                keypoints_3d_rel[0],  # first person only
                subject_height=self.primary_subject_height,
            )
            self.metric_scale_estimated = True
            logging.info(
                f"Estimated metric scale: {self.metric_scale:.4f} meters per unit"
            )
        keypoints_3d_rel = keypoints_3d_rel * self.metric_scale

        root_translations = []
        for pose_rel, xy_n, vis in zip(keypoints_3d_rel, xy_norm, visible):
            root_transl = estimate_root_translation(
                P_rel=pose_rel,
                xy_norm=xy_n,
                visible=vis,
            )
            root_translations.append(root_transl)
        root_translations = np.stack(root_translations, axis=0)

        keypoints_3d = keypoints_3d_rel + root_translations[:, None, :]

        return keypoints_3d
