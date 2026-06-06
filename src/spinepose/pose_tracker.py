from __future__ import annotations

import logging
import warnings

import numpy as np

from .camera_transforms import (
    build_camera_to_world_transform,
    estimate_camera_pose,
    project_cam_to_world,
    project_world_to_img,
)
from .tools.smoothing import KeypointSmoothing


def compute_iou(bboxA: np.ndarray, bboxB: np.ndarray) -> float:
    """Computes the intersection-over-union between two boxes.

    Args:
        bboxA: First bounding box in ``xyxy`` format.
        bboxB: Second bounding box in ``xyxy`` format.

    Returns:
        float: Intersection-over-union score.
    """

    x1 = max(bboxA[0], bboxB[0])
    y1 = max(bboxA[1], bboxB[1])
    x2 = min(bboxA[2], bboxB[2])
    y2 = min(bboxA[3], bboxB[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)

    bboxA_area = (bboxA[2] - bboxA[0]) * (bboxA[3] - bboxA[1])
    bboxB_area = (bboxB[2] - bboxB[0]) * (bboxB[3] - bboxB[1])
    union_area = float(bboxA_area + bboxB_area - inter_area)
    if union_area == 0:
        union_area = 1e-5
        warnings.warn("union_area=0 is unexpected")

    return inter_area / union_area


def pose_to_bbox(keypoints: np.ndarray, expansion: float = 1.25) -> np.ndarray:
    """Builds a bounding box around a set of keypoints.

    Args:
        keypoints: Keypoints for one pose.
        expansion: Expansion factor applied to the box.

    Returns:
        np.ndarray: Bounding box in ``xyxy`` format.
    """
    x = keypoints[:, 0]
    y = keypoints[:, 1]
    bbox = np.array([x.min(), y.min(), x.max(), y.max()])
    center = np.array([bbox[0] + bbox[2], bbox[1] + bbox[3]]) / 2
    return np.concatenate(
        [
            center - (center - bbox[:2]) * expansion,
            center + (bbox[2:] - center) * expansion,
        ]
    )


class PoseTracker:
    """Tracks poses across frames for temporal consistency."""

    MIN_AREA = 1000

    def __init__(
        self,
        solution: type,
        mode: str = "large",
        det_frequency: int = 1,
        max_detections: int = 10,
        # Tracking parameters
        tracking: bool = True,
        tracking_thr: float = 0.3,
        # Smoother parameters
        smoothing: bool = False,
        smoothing_freq: float = 30.0,
        smoothing_mincutoff: float = 0.1,
        smoothing_beta: float = 0.1,
        smoothing_dcutoff: float = 1.0,  # Derivative cutoff frequency
        model_version: str = "latest",
        detector: str = "rfdetr",
        # Lifting parameters
        enable_lifting: bool = False,
        camera_intrinsics: np.ndarray | None = None,
        camera_field_of_view: float | None = 84.0,
        estimate_metric_scale: bool = True,
        estimate_camera_pose: bool = True,
        estimate_ground_plane: bool = True,
        primary_subject_height: float = 1.84,
        warmup_frames: int = 30,
        **kwargs,
    ) -> None:
        """Initializes the pose tracker.

        Args:
            solution: Pose solution class to instantiate.
            mode: Model preset to load.
            det_frequency: Detection frequency in frames.
            max_detections: Maximum number of detections to consider.
            tracking: Whether to enable tracking.
            tracking_thr: IoU threshold for track association.
            smoothing: Whether to smooth keypoints over time.
            smoothing_freq: Expected frame rate for smoothing.
            smoothing_mincutoff: Minimum cutoff for smoothing.
            smoothing_beta: Speed coefficient for smoothing.
            smoothing_dcutoff: Derivative cutoff for smoothing.
            model_version: Model version to use.
            detector: Detector name to use.
            enable_lifting: Whether to enable 2D-to-3D pose lifting.
            camera_intrinsics: Camera intrinsic matrix for 3D lifting.
            camera_field_of_view: Diagonal field of view used to estimate intrinsics.
            estimate_metric_scale: Whether to estimate metric scale from height.
            estimate_camera_pose: Whether to estimate camera-to-world orientation.
            estimate_ground_plane: Whether to align world orientation to the ground.
            primary_subject_height: Primary subject height in meters.
            warmup_frames: Number of frames used to stabilize camera calibration.
            **kwargs: Additional arguments forwarded to the solution.
        """
        self.solution = solution(
            mode=mode,
            detector=detector,
            model_version=model_version,
            enable_lifting=enable_lifting,
            lifting_thr=tracking_thr,
            camera_intrinsics=camera_intrinsics,
            estimate_metric_scale=estimate_metric_scale,
            camera_field_of_view=camera_field_of_view,
            primary_subject_height=primary_subject_height,
            **kwargs,
        )

        self.det_frequency = det_frequency
        self.max_detections = max_detections
        self.smoothing = smoothing
        self.smoothing_cfg = dict(
            num_keypoints=self.solution.num_keypoints,
            freq=smoothing_freq,
            mincutoff=smoothing_mincutoff,
            beta=smoothing_beta,
            dcutoff=smoothing_dcutoff,
        )
        self.filters = {}
        self.tracking = tracking or smoothing
        self.tracking_thr = tracking_thr

        # Calibration parameters for 3D lifting
        self.estimate_camera_pose = estimate_camera_pose
        self.estimate_ground_plane = estimate_ground_plane
        self.warmup_frames = max(1, int(warmup_frames))

        self.reset()

    def reset(self) -> None:
        """Resets the internal tracking state."""
        self.frame_cnt = 0
        self.next_id = 0
        self.bboxes_last_frame = []
        self.track_ids_last_frame = []

        # Reset calibration state for lifting
        self._camera_to_world = None  # (4,4) camera-to-world transform
        self._world_to_pixels = None  # (3,4) camera projection matrix
        self._calibration_anchor_root = None
        self._calibration_up_axis_accum = np.zeros(3, dtype=np.float32)
        self._calibration_up_axis_weight_sum = 0.0
        self._calibration_ground_point_accum = np.zeros(3, dtype=np.float32)
        self._calibration_ground_weight_sum = 0.0
        self._calibration_frames = 0

    def _set_camera_transform(self, camera_to_world: np.ndarray) -> None:
        """Sets camera/world transforms used for 3D output and reprojection.

        Args:
            camera_to_world: Camera-to-world transform with shape ``(4, 4)``.

        Raises:
            ValueError: If camera intrinsics are unavailable.
        """
        intrinsics = getattr(self.solution, "camera_intrinsics", None)
        if intrinsics is None:
            raise ValueError(
                "Camera intrinsics must be set before setting camera pose."
            )

        R_cw = camera_to_world[:3, :3]
        t_cw = camera_to_world[:3, 3]

        R_wc = R_cw.T
        t_wc = -R_wc @ t_cw
        world_to_pixels = intrinsics @ np.concatenate(
            [R_wc, t_wc[:, None]],
            axis=1,
        )

        self._camera_to_world = camera_to_world.astype(np.float32)
        self._world_to_pixels = world_to_pixels.astype(np.float32)

    def _set_identity_camera_transform(self) -> None:
        """Uses camera coordinates directly as world coordinates."""
        intrinsics = getattr(self.solution, "camera_intrinsics", None)
        camera_to_world = np.eye(4, dtype=np.float32)
        self._camera_to_world = camera_to_world
        if intrinsics is None:
            self._world_to_pixels = None
            return

        self._world_to_pixels = (
            intrinsics
            @ np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                ],
                dtype=np.float32,
            )
        ).astype(np.float32)

    def _log_camera_state(
        self, title: str, up_axis: np.ndarray, origin: np.ndarray
    ) -> None:
        """Logs the current camera calibration state."""
        logging.info(title)
        logging.info(f"  Up Axis: {up_axis}")
        logging.info(f"  World Origin: {origin}")
        logging.info(f"  Camera-to-World Transform:\n{self._camera_to_world}")
        logging.info(f"  World-to-Pixel Projection:\n{self._world_to_pixels}")

    def _update_camera_pose(self, points_cam: np.ndarray, scores: np.ndarray) -> None:
        """Updates camera pose from one lifted subject.

        Args:
            points_cam: Camera-space 3D keypoints with shape ``(K, 3)``.
            scores: Keypoint confidence scores with shape ``(K,)``.
        """
        if not self.estimate_camera_pose:
            if self._camera_to_world is None:
                self._set_identity_camera_transform()
            return

        if not self.estimate_ground_plane:
            if self._camera_to_world is None:
                initial_camera_pose = estimate_camera_pose(
                    points_cam,
                    scores,
                    estimate_ground_plane=False,
                )
                initial_transform = initial_camera_pose.transform
                self._set_camera_transform(initial_transform)
                self._log_camera_state(
                    "Camera pose initialized without ground-plane alignment:",
                    initial_camera_pose.up_axis,
                    initial_camera_pose.world_origin,
                )
            return

        if (
            self._camera_to_world is not None
            and self._calibration_frames >= self.warmup_frames
        ):
            if self._calibration_frames == self.warmup_frames:
                up_axis = self._calibration_up_axis_accum / max(
                    self._calibration_up_axis_weight_sum,
                    1e-6,
                )
                origin = self._calibration_ground_point_accum / max(
                    self._calibration_ground_weight_sum,
                    1e-6,
                )
                self._log_camera_state("Camera pose warmup complete:", up_axis, origin)
                self._calibration_frames += 1
            return

        camera_pose = estimate_camera_pose(
            points_cam,
            scores,
            estimate_ground_plane=True,
        )
        if self._calibration_anchor_root is None:
            self._calibration_anchor_root = np.asarray(
                points_cam[19],
                dtype=np.float32,
            ).copy()

        up_axis = camera_pose.up_axis
        if (
            self._calibration_up_axis_weight_sum > 0.0
            and float(np.dot(self._calibration_up_axis_accum, up_axis)) < 0.0
        ):
            up_axis = -up_axis

        orientation_weight = max(0.25, float(camera_pose.confidence))
        ground_weight = max(0.05, float(camera_pose.confidence))
        self._calibration_up_axis_accum += orientation_weight * up_axis
        self._calibration_up_axis_weight_sum += orientation_weight
        self._calibration_ground_point_accum += ground_weight * camera_pose.ground_point
        self._calibration_ground_weight_sum += ground_weight
        self._calibration_frames += 1

        averaged_up_axis = self._calibration_up_axis_accum / max(
            self._calibration_up_axis_weight_sum,
            1e-6,
        )
        averaged_up_axis_norm = float(np.linalg.norm(averaged_up_axis))
        if averaged_up_axis_norm < 1e-6:
            averaged_up_axis = camera_pose.up_axis
        else:
            averaged_up_axis = averaged_up_axis / averaged_up_axis_norm

        averaged_ground_point = self._calibration_ground_point_accum / max(
            self._calibration_ground_weight_sum,
            1e-6,
        )
        anchor_root = self._calibration_anchor_root
        plane_offset = float(np.dot(averaged_ground_point, averaged_up_axis))
        anchor_height = float(np.dot(anchor_root, averaged_up_axis) - plane_offset)
        world_origin = anchor_root - anchor_height * averaged_up_axis

        updated_transform = build_camera_to_world_transform(
            averaged_up_axis,
            world_origin,
            estimate_ground_plane=True,
        )
        self._set_camera_transform(updated_transform)

    def visualize(
        self, image: np.ndarray, keypoints: np.ndarray, scores: np.ndarray
    ) -> np.ndarray:
        """Draws the tracked pose predictions on an image.

        Args:
            image: Input image.
            keypoints: Predicted keypoints.
            scores: Predicted confidence scores.

        Returns:
            np.ndarray: Annotated image.
        """
        return self.solution.visualize(image, keypoints, scores)

    def __call__(
        self, image: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Processes a single frame and updates tracking state.

        Args:
            image: Input image.

        Returns:
            tuple: 2D keypoints and scores, plus world-space 3D keypoints when
            lifting is enabled and succeeds.
        """
        # Determine boxes using detection or reuse boxes from the last frame.
        if self.solution.det_model:
            if self.frame_cnt % self.det_frequency == 0:
                bboxes = self.solution.detect(image)
            else:
                bboxes = self.bboxes_last_frame
            bboxes = bboxes[: self.max_detections]
        else:
            bboxes = None  # For solutions that don't use detection

        # Run pose estimation (detection + pose + postprocessing)
        results = self.solution.estimate(image, bboxes)
        if len(results) == 3:
            keypoints, scores, keypoints_3d_cam = results
            if keypoints_3d_cam is None:
                keypoints_3d = None
            else:
                if len(keypoints_3d_cam) > 0:
                    self._update_camera_pose(keypoints_3d_cam[0], scores[0])
                elif self._camera_to_world is None:
                    self._set_identity_camera_transform()

                keypoints_3d = project_cam_to_world(
                    keypoints_3d_cam,
                    self._camera_to_world,
                )
                if self._world_to_pixels is not None and len(keypoints_3d) > 0:
                    reprojection = project_world_to_img(
                        keypoints_3d,
                        self._world_to_pixels,
                    )
                    error = np.linalg.norm(reprojection - keypoints, axis=-1)
                    error = error[scores >= self.tracking_thr]
                    error = error.mean() if len(error) > 0 else 0.0
                    logging.debug(f"Mean reprojection error: {error:.2f} px")
        else:
            keypoints, scores = results
            keypoints_3d = None

        if not self.tracking:
            # Without tracking, simply compute bounding boxes from keypoints
            bboxes_current_frame = [pose_to_bbox(kpts) for kpts in keypoints]
        else:
            # With tracking, assign track IDs based on IoU matching
            if not self.track_ids_last_frame:
                # Initialize track IDs for the first frame
                self.track_ids_last_frame = (
                    list(range(len(bboxes))) if bboxes is not None else []
                )
                self.next_id = len(self.track_ids_last_frame)

            bboxes_current_frame = []
            new_track_ids = []
            for kpts in keypoints:
                bbox = pose_to_bbox(kpts)
                track_id, _ = self.track_by_iou(bbox)
                if track_id >= 0:
                    new_track_ids.append(track_id)
                    bboxes_current_frame.append(bbox)

            self.track_ids_last_frame = new_track_ids

        # Smooth keypoints if enabled
        if self.smoothing:
            # Map detections to track IDs, then update per-track filters.
            for i, (kpts, track_id) in enumerate(
                zip(keypoints, self.track_ids_last_frame)
            ):
                if track_id < 0:
                    # Skip smoothing for untracked bboxes
                    continue
                if track_id not in self.filters:
                    # Create a new smoother for this track
                    self.filters[track_id] = KeypointSmoothing(**self.smoothing_cfg)
                # Update the filter with the current keypoints
                smoothed_kpts = self.filters[track_id](kpts)
                keypoints[i] = smoothed_kpts

            # (Optional) Remove filters for track IDs that disappeared this frame
            current_tracks = set(self.track_ids_last_frame)
            disappeared = [tid for tid in self.filters if tid not in current_tracks]
            for tid in disappeared:
                del self.filters[tid]

        # Save state for the next frame and increment frame counter
        self.bboxes_last_frame = bboxes_current_frame
        self.frame_cnt += 1

        if keypoints_3d is not None:
            return keypoints, scores, keypoints_3d

        return keypoints, scores

    def track_by_iou(self, bbox: np.ndarray) -> tuple[int, float]:
        """Assigns a track ID using IoU against the previous frame.

        Args:
            bbox: Current bounding box in ``xyxy`` format.

        Returns:
            tuple: Assigned track ID and the best IoU score.
        """
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        max_iou = -1
        max_index = -1
        for index, prev_bbox in enumerate(self.bboxes_last_frame):
            iou = compute_iou(bbox, prev_bbox)
            if iou > max_iou:
                max_iou = iou
                max_index = index

        if max_iou > self.tracking_thr:
            # Match found: reuse the corresponding track ID
            track_id = self.track_ids_last_frame.pop(max_index)
            self.bboxes_last_frame.pop(
                max_index
            )  # remove matched bbox to avoid duplicate matches
        elif area >= self.MIN_AREA:
            # No good match and bbox is large: assign a new track ID
            track_id = self.next_id
            self.next_id += 1
        else:
            # Bbox is too small to track
            track_id = -1

        return track_id, max_iou
