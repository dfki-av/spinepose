from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from spinepose.metainfo import metainfo

keypoint_info = list(metainfo["keypoint_info"].values())
keypoint_names = [kpt["name"] for kpt in keypoint_info]

ROOT_JOINT_ID = keypoint_names.index("hip")
FOOT_JOINT_NAMES = [
    "left_big_toe",
    "right_big_toe",
    "left_small_toe",
    "right_small_toe",
    "left_heel",
    "right_heel",
]
FOOT_JOINT_IDS = np.array(
    [keypoint_names.index(name) for name in FOOT_JOINT_NAMES], dtype=np.int64
)
LEFT_FOOT_JOINT_IDS = np.array(
    [
        keypoint_names.index(name)
        for name in ["left_big_toe", "left_small_toe", "left_heel"]
    ],
    dtype=np.int64,
)
RIGHT_FOOT_JOINT_IDS = np.array(
    [
        keypoint_names.index(name)
        for name in ["right_big_toe", "right_small_toe", "right_heel"]
    ],
    dtype=np.int64,
)
FOOT_GROUP_JOINT_IDS = [LEFT_FOOT_JOINT_IDS, RIGHT_FOOT_JOINT_IDS]
BODY_UP_SEGMENT_IDS = [
    (keypoint_names.index("hip"), keypoint_names.index("neck")),
    (keypoint_names.index("hip"), keypoint_names.index("head")),
    (keypoint_names.index("left_hip"), keypoint_names.index("hip")),
    (keypoint_names.index("left_knee"), keypoint_names.index("left_hip")),
    (keypoint_names.index("left_ankle"), keypoint_names.index("left_knee")),
    (keypoint_names.index("right_hip"), keypoint_names.index("hip")),
    (keypoint_names.index("right_knee"), keypoint_names.index("right_hip")),
    (keypoint_names.index("right_ankle"), keypoint_names.index("right_knee")),
]
GROUND_CLEARANCE = 0.02
SUPPORT_FOOT_HEIGHT_TOL = 0.12
PLANE_RESIDUAL_TOL = 0.04
GROUND_HEIGHT_QUANTILE = 0.25
VISIBLE_HEIGHT_QUANTILE = 0.1


@dataclass
class CameraPoseEstimate:
    """Estimated camera-to-world transform and calibration metadata.

    Attributes:
        transform: Camera-to-world transform with shape ``(4, 4)``.
        up_axis: Estimated world up direction in camera coordinates.
        world_origin: Estimated world origin in camera coordinates.
        ground_point: Estimated ground-plane point in camera coordinates.
        confidence: Confidence score for the estimated orientation.
    """

    transform: np.ndarray
    up_axis: np.ndarray
    world_origin: np.ndarray
    ground_point: np.ndarray
    confidence: float


def normalize_vector(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Normalizes a vector.

    Args:
        x: Input vector.
        eps: Minimum allowed vector norm.

    Returns:
        np.ndarray: Unit vector.

    Raises:
        ValueError: If the input vector is near zero.
    """
    x = np.asarray(x, dtype=np.float32)
    norm = float(np.linalg.norm(x))
    if norm < eps:
        raise ValueError("Cannot normalize a near-zero vector.")
    return x / norm


def estimate_body_up_axis(
    points_cam: np.ndarray,
    scores: np.ndarray,
    threshold: float = 0.3,
) -> np.ndarray:
    """Estimates the body up axis from visible body segments.

    Args:
        points_cam: Camera-space 3D keypoints with shape ``(K, 3)``.
        scores: Keypoint confidence scores with shape ``(K,)``.
        threshold: Minimum score used to select visible joints.

    Returns:
        np.ndarray: Unit up-axis vector in camera coordinates.
    """
    candidate_vectors = []
    candidate_weights = []
    for lower_joint_id, upper_joint_id in BODY_UP_SEGMENT_IDS:
        lower_score = float(scores[lower_joint_id])
        upper_score = float(scores[upper_joint_id])
        if lower_score < threshold or upper_score < threshold:
            continue

        segment = np.asarray(
            points_cam[upper_joint_id] - points_cam[lower_joint_id],
            dtype=np.float32,
        )
        segment_norm = float(np.linalg.norm(segment))
        if segment_norm < 1e-4:
            continue

        candidate_vectors.append(segment / segment_norm)
        candidate_weights.append(
            max(1e-3, 0.5 * (lower_score + upper_score) * segment_norm)
        )

    if candidate_vectors:
        stacked_vectors = np.stack(candidate_vectors, axis=0)
        stacked_weights = np.asarray(candidate_weights, dtype=np.float32)
        return normalize_vector(
            np.sum(stacked_vectors * stacked_weights[:, None], axis=0)
        )

    fallback = np.asarray(
        points_cam[keypoint_names.index("head")] - points_cam[ROOT_JOINT_ID],
        dtype=np.float32,
    )
    if float(np.linalg.norm(fallback)) >= 1e-4:
        return normalize_vector(fallback)

    return np.array([0.0, -1.0, 0.0], dtype=np.float32)


def select_support_joint_ids(
    points_cam: np.ndarray,
    scores: np.ndarray,
    body_up_axis: np.ndarray,
    root_joint_id: int = ROOT_JOINT_ID,
    threshold: float = 0.3,
) -> np.ndarray:
    """Selects foot joints likely to support the body on the ground.

    Args:
        points_cam: Camera-space 3D keypoints with shape ``(K, 3)``.
        scores: Keypoint confidence scores with shape ``(K,)``.
        body_up_axis: Estimated body up-axis vector.
        root_joint_id: Root joint index.
        threshold: Minimum score used to select visible joints.

    Returns:
        np.ndarray: Selected support joint indices.
    """
    root_position = np.asarray(points_cam[root_joint_id], dtype=np.float32)
    visible_foot_ids = FOOT_JOINT_IDS[scores[FOOT_JOINT_IDS] >= threshold]
    if len(visible_foot_ids) < 3:
        return visible_foot_ids

    support_groups = []
    for foot_joint_ids in FOOT_GROUP_JOINT_IDS:
        visible = scores[foot_joint_ids] >= threshold
        if int(np.count_nonzero(visible)) < 2:
            continue

        foot_visible_ids = foot_joint_ids[visible]
        foot_points = np.asarray(points_cam[foot_visible_ids], dtype=np.float32)
        foot_heights = (foot_points - root_position[None, :]) @ body_up_axis
        support_groups.append((float(np.median(foot_heights)), foot_visible_ids))

    if not support_groups:
        return visible_foot_ids

    min_height = min(group_height for group_height, _ in support_groups)
    selected_ids = [
        foot_ids
        for group_height, foot_ids in support_groups
        if group_height <= min_height + SUPPORT_FOOT_HEIGHT_TOL
    ]
    if not selected_ids:
        return visible_foot_ids

    support_joint_ids = np.concatenate(selected_ids).astype(np.int64)
    if len(support_joint_ids) < 3:
        return visible_foot_ids
    return support_joint_ids


def build_camera_to_world_transform(
    up_axis: np.ndarray,
    world_origin: np.ndarray,
    estimate_ground_plane: bool = True,
) -> np.ndarray:
    """Builds a camera-to-world transform from orientation estimates.

    Args:
        up_axis: World up direction in camera coordinates.
        world_origin: World origin in camera coordinates.
        estimate_ground_plane: Whether to align world axes to the ground plane.

    Returns:
        np.ndarray: Camera-to-world transform with shape ``(4, 4)``.
    """
    if estimate_ground_plane:
        camera_x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        world_x_axis = camera_x_axis - float(np.dot(camera_x_axis, up_axis)) * up_axis
        if float(np.linalg.norm(world_x_axis)) < 1e-4:
            camera_z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            world_x_axis = np.cross(camera_z_axis, up_axis)
        world_x_axis = normalize_vector(world_x_axis)
        world_z_axis = normalize_vector(np.cross(world_x_axis, up_axis))
        rotation = np.stack([world_x_axis, up_axis, world_z_axis], axis=0).astype(
            np.float32
        )
    else:
        rotation = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = rotation
    transform[:3, 3] = -(world_origin @ rotation.T).astype(np.float32)
    return transform


def fit_support_plane(
    points_cam: np.ndarray,
    scores: np.ndarray,
    root_joint_id: int = ROOT_JOINT_ID,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fits a support plane from visible foot keypoints.

    Args:
        points_cam: Camera-space 3D keypoints with shape ``(K, 3)``.
        scores: Keypoint confidence scores with shape ``(K,)``.
        root_joint_id: Root joint index.

    Returns:
        tuple: Support points, centroid, plane normal, body up axis, and
        per-point plane residuals.
    """
    body_up_axis = estimate_body_up_axis(points_cam, scores)
    support_joint_ids = select_support_joint_ids(
        points_cam,
        scores,
        body_up_axis,
        root_joint_id=root_joint_id,
    )
    if len(support_joint_ids) < 3:
        support_joint_ids = FOOT_JOINT_IDS[scores[FOOT_JOINT_IDS] >= 0.3]
    if len(support_joint_ids) < 3:
        root_position = np.asarray(points_cam[root_joint_id], dtype=np.float32)
        return (
            root_position[None, :],
            root_position,
            body_up_axis,
            body_up_axis,
            np.zeros(1, dtype=np.float32),
        )

    support_points = np.asarray(points_cam[support_joint_ids], dtype=np.float32)
    support_scores = np.asarray(scores[support_joint_ids], dtype=np.float32)
    support_weights = np.clip(support_scores, 1e-3, None)
    support_weights /= support_weights.sum()

    support_centroid = np.sum(support_points * support_weights[:, None], axis=0)
    support_centered = support_points - support_centroid
    weighted_support = support_centered * np.sqrt(support_weights)[:, None]
    _, _, vh = np.linalg.svd(weighted_support, full_matrices=False)
    plane_normal = normalize_vector(vh[-1])
    if float(np.dot(body_up_axis, plane_normal)) < 0.0:
        plane_normal = -plane_normal

    root_offset = (
        np.asarray(points_cam[root_joint_id], dtype=np.float32) - support_centroid
    )
    if float(np.dot(root_offset, plane_normal)) < 0.0:
        plane_normal = -plane_normal

    plane_residuals = np.abs((support_points - support_centroid) @ plane_normal)
    return support_points, support_centroid, plane_normal, body_up_axis, plane_residuals


def estimate_up_axis(
    points_cam: np.ndarray,
    scores: np.ndarray,
    root_joint_id: int = ROOT_JOINT_ID,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Estimates the world up axis and supporting ground points.

    Args:
        points_cam: Camera-space 3D keypoints with shape ``(K, 3)``.
        scores: Keypoint confidence scores with shape ``(K,)``.
        root_joint_id: Root joint index.

    Returns:
        tuple: Up axis, support points, support centroid, and confidence score.
    """
    support_points, support_centroid, plane_normal, body_up_axis, plane_residuals = (
        fit_support_plane(
            points_cam,
            scores,
            root_joint_id=root_joint_id,
        )
    )

    support_feet = sum(
        int(np.any(scores[foot_joint_ids] >= 0.3))
        for foot_joint_ids in FOOT_GROUP_JOINT_IDS
    )
    plane_weight = np.clip(
        1.0 - float(np.mean(plane_residuals)) / PLANE_RESIDUAL_TOL,
        0.0,
        1.0,
    )
    if support_feet < 2:
        plane_weight *= 0.45
    if len(support_points) < 4:
        plane_weight *= 0.7

    plane_alignment = float(abs(np.dot(plane_normal, body_up_axis)))
    plane_weight *= np.clip(0.5 + 0.5 * plane_alignment, 0.0, 1.0)
    plane_weight = float(np.clip(plane_weight, 0.0, 0.85))

    if plane_weight <= 1e-3:
        up_axis = body_up_axis
    else:
        up_axis = normalize_vector(
            (1.0 - plane_weight) * body_up_axis + plane_weight * plane_normal
        )

    return up_axis, support_points, support_centroid, plane_weight


def estimate_camera_pose(
    points_cam: np.ndarray,
    scores: np.ndarray,
    root_joint_id: int = ROOT_JOINT_ID,
    ground_clearance: float = GROUND_CLEARANCE,
    estimate_ground_plane: bool = True,
) -> CameraPoseEstimate:
    """Estimates camera pose from one lifted subject.

    Args:
        points_cam: Camera-space 3D keypoints with shape ``(K, 3)``.
        scores: Keypoint confidence scores with shape ``(K,)``.
        root_joint_id: Root joint index.
        ground_clearance: Clearance added above the estimated support plane.
        estimate_ground_plane: Whether to use foot support to align the ground.

    Returns:
        CameraPoseEstimate: Estimated camera pose and calibration metadata.
    """
    up_axis, support_points, support_centroid, confidence = estimate_up_axis(
        points_cam,
        scores,
        root_joint_id=root_joint_id,
    )

    support_heights = (support_points - support_centroid) @ up_axis
    ground_offset = ground_clearance - float(
        np.quantile(support_heights, GROUND_HEIGHT_QUANTILE)
    )
    ground_point = support_centroid - ground_offset * up_axis

    root_position = np.asarray(points_cam[root_joint_id], dtype=np.float32)
    root_height = float(np.dot(root_position - ground_point, up_axis))
    world_origin = root_position - root_height * up_axis

    visible = np.asarray(scores, dtype=np.float32) >= 0.3
    visible_points = np.asarray(points_cam[visible], dtype=np.float32)
    if len(visible_points) == 0:
        visible_points = np.asarray(points_cam, dtype=np.float32)
    all_heights = (visible_points - world_origin[None, :]) @ up_axis
    min_height = float(np.quantile(all_heights, VISIBLE_HEIGHT_QUANTILE))
    if min_height < 0.0:
        world_origin = world_origin + min_height * up_axis
        ground_point = ground_point + min_height * up_axis

    transform = build_camera_to_world_transform(
        up_axis,
        world_origin,
        estimate_ground_plane=estimate_ground_plane,
    )
    return CameraPoseEstimate(
        transform=transform,
        up_axis=up_axis,
        world_origin=world_origin.astype(np.float32),
        ground_point=ground_point.astype(np.float32),
        confidence=confidence,
    )


def project_cam_to_world(
    points_cam: np.ndarray,
    camera_to_world_transform: np.ndarray,
) -> np.ndarray:
    """Projects camera-space points into world coordinates.

    Args:
        points_cam: Camera-space points with shape ``(..., 3)``.
        camera_to_world_transform: Camera-to-world transform with shape ``(4, 4)``.

    Returns:
        np.ndarray: World-space points with shape ``(..., 3)``.
    """
    rotation = camera_to_world_transform[:3, :3]
    translation = camera_to_world_transform[:3, 3]
    return np.asarray(points_cam, dtype=np.float32) @ rotation.T + translation


def project_world_to_img(
    points_world: np.ndarray,
    world_to_pixel_transform: np.ndarray,
) -> np.ndarray:
    """Projects world-space points into image coordinates.

    Args:
        points_world: World-space points with shape ``(..., 3)``.
        world_to_pixel_transform: World-to-pixel projection with shape ``(3, 4)``.

    Returns:
        np.ndarray: Image points with shape ``(..., 2)``.
    """
    points_world = np.asarray(points_world, dtype=np.float32)
    points_homog = np.concatenate(
        [points_world, np.ones((*points_world.shape[:-1], 1), dtype=np.float32)],
        axis=-1,
    )
    points_proj = points_homog @ world_to_pixel_transform.T
    points_2d = points_proj[..., :2] / points_proj[..., 2:3]
    return points_2d
