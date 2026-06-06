from __future__ import annotations

import numpy as np

from spinepose.metainfo import metainfo

keypoint_info = list(metainfo["keypoint_info"].values())
keypoint_names = [kpt["name"] for kpt in keypoint_info]


def estimate_camera_intrinsics(
    width: int,
    height: int,
    dfov: float | None = 84.0,
    hfov: float | None = None,
    vfov: float | None = None,
    principal_point: tuple[float, float] | None = None,
) -> np.ndarray:
    """Builds a pinhole camera intrinsic matrix.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.
        dfov: Diagonal field of view in degrees.
        hfov: Horizontal field of view in degrees.
        vfov: Vertical field of view in degrees.
        principal_point: Optional principal point as ``(cx, cy)``.

    Returns:
        np.ndarray: Camera intrinsic matrix with shape ``(3, 3)``.

    Raises:
        ValueError: If no field-of-view value is provided.
    """
    if principal_point is None:
        cx, cy = 0.5 * width, 0.5 * height
    else:
        cx, cy = principal_point

    if hfov is None and vfov is None:
        if dfov is None:
            raise ValueError("Provide dfov, hfov, or vfov.")

        fd = np.deg2rad(dfov)
        d = np.hypot(width, height)
        f = 0.5 * d / np.tan(0.5 * fd)
        fx = fy = f

    elif hfov is not None and vfov is None:
        fh = np.deg2rad(hfov)
        fx = 0.5 * width / np.tan(0.5 * fh)
        fy = fx  # square-pixel assumption

    elif hfov is None and vfov is not None:
        fv = np.deg2rad(vfov)
        fy = 0.5 * height / np.tan(0.5 * fv)
        fx = fy  # square-pixel assumption

    else:
        fh = np.deg2rad(hfov)
        fv = np.deg2rad(vfov)
        fx = 0.5 * width / np.tan(0.5 * fh)
        fy = 0.5 * height / np.tan(0.5 * fv)

    return np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def estimate_root_translation(
    P_rel: np.ndarray,  # (K, 3), root-relative 3D
    xy_norm: np.ndarray,  # (K, 2), normalized undistorted 2D
    visible: np.ndarray,  # (K,)
) -> np.ndarray:
    """Estimates camera-space root translation from 2D/3D correspondences.

    Args:
        P_rel: Root-relative 3D keypoints with shape ``(K, 3)``.
        xy_norm: Normalized 2D keypoints with shape ``(K, 2)``.
        visible: Visibility mask with shape ``(K,)``.

    Returns:
        np.ndarray: Root translation vector with shape ``(3,)``.
    """
    rows = []
    rhs = []

    for (X, Y, Z), (x, y), vis in zip(P_rel, xy_norm, visible):
        if not vis:
            continue

        # x * (Z + tz) = X + tx
        # tx - x * tz = x * Z - X
        rows.append([1.0, 0.0, -x])
        rhs.append(x * Z - X)

        # y * (Z + tz) = Y + ty
        # ty - y * tz = y * Z - Y
        rows.append([0.0, 1.0, -y])
        rhs.append(y * Z - Y)

    if len(rows) < 6:
        return np.array([0.0, 0.0, 3.0], dtype=np.float32)

    A = np.asarray(rows, dtype=np.float32)
    b = np.asarray(rhs, dtype=np.float32)

    t, *_ = np.linalg.lstsq(A, b, rcond=None)

    # Prevent invalid camera placement.
    if t[2] <= 0.1:
        t[2] = 3.0

    return t.astype(np.float32)


def estimate_metric_scale(
    keypoints_3d: np.ndarray,
    subject_height: float = 1.84,
) -> float:
    """Estimates metric scale from a predicted skeleton height.

    Args:
        keypoints_3d: Relative 3D keypoints with shape ``(K, 3)``.
        subject_height: Reference subject height in meters.

    Returns:
        float: Meters per model unit.
    """
    pelvis_to_headtop = ["hip", "neck", "head"]
    left_leg = ["hip", "left_knee", "left_ankle", "left_heel"]
    right_leg = ["hip", "right_knee", "right_ankle", "right_heel"]

    def segment_length(joint_sequence: list[str]) -> float:
        """Computes the cumulative length of a joint chain."""
        length = 0.0
        for i in range(len(joint_sequence) - 1):
            joint_a = joint_sequence[i]
            joint_b = joint_sequence[i + 1]
            idx_a = keypoint_names.index(joint_a)
            idx_b = keypoint_names.index(joint_b)
            point_a = keypoints_3d[idx_a]
            point_b = keypoints_3d[idx_b]
            length += np.linalg.norm(point_b - point_a)
        return length

    length_pelvis_head = segment_length(pelvis_to_headtop)
    length_left_leg = segment_length(left_leg)
    length_right_leg = segment_length(right_leg)
    computed_height = length_pelvis_head + 0.5 * (length_left_leg + length_right_leg)
    if computed_height < 1e-5:
        return 1.0

    metric_scale = subject_height / computed_height
    return float(metric_scale)
