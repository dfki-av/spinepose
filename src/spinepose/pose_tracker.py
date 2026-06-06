import warnings
from typing import Tuple

import numpy as np

from .tools.smoothing import KeypointSmoothing


def compute_iou(bboxA, bboxB):
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
        smoothing_freq: float = 30.0,  # Default frequency of the input data (e.g., 30 FPS video)
        smoothing_mincutoff: float = 0.1,  # Lower cutoff for smoothing (higher = less smoothing)
        smoothing_beta: float = 0.1,  # Speed coefficient (higher = more dynamic adaptation)
        smoothing_dcutoff: float = 1.0,  # Derivative cutoff frequency
        model_version: str = "latest",
        detector: str = "rfdetr",
        **kwargs,
    ):
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
            **kwargs: Additional arguments forwarded to the solution.
        """
        self.solution = solution(
            mode=mode,
            detector=detector,
            model_version=model_version,
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
        self.reset()

    def reset(self):
        """Resets the internal tracking state."""
        self.frame_cnt = 0
        self.next_id = 0
        self.bboxes_last_frame = []
        self.track_ids_last_frame = []

    def visualize(self, image: np.ndarray, keypoints: np.ndarray, scores: np.ndarray):
        """Draws the tracked pose predictions on an image.

        Args:
            image: Input image.
            keypoints: Predicted keypoints.
            scores: Predicted confidence scores.

        Returns:
            np.ndarray: Annotated image.
        """
        return self.solution.visualize(image, keypoints, scores)

    def __call__(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Processes a single frame and updates tracking state.

        Args:
            image: Input image.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Keypoints and confidence scores.
        """
        # Determine bounding boxes using detection (if available) or reuse from last frame
        if self.solution.det_model:
            if self.frame_cnt % self.det_frequency == 0:
                bboxes = self.solution.detect(image)
            else:
                bboxes = self.bboxes_last_frame
        else:
            bboxes = None  # For solutions that don't use detection

        # Run pose estimation (detection + pose + postprocessing)
        keypoints, scores = self.solution.estimate(image, bboxes)

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
            # Map each detection to its track ID, then create or update the smoothing filter
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

        return keypoints, scores

    def track_by_iou(self, bbox):
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
