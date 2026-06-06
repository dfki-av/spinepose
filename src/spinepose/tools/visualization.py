from __future__ import annotations

import cv2
import numpy as np


def draw_bbox(img, bboxes, color=(0, 255, 0)):
    """Draws bounding boxes on an image.

    Args:
        img: Input image.
        bboxes: Bounding boxes in ``xyxy`` format.
        color: Box color in BGR format.

    Returns:
        The annotated image.
    """
    for bbox in bboxes:
        img = cv2.rectangle(
            img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2
        )
    return img


def draw_world_pose_panel(
    points_world: np.ndarray,
    scores: np.ndarray,
    metainfo: dict,
    panel_size: tuple[int, int],
    center: np.ndarray | None = None,
    radius: float | None = None,
    threshold: float = 0.5,
) -> np.ndarray:
    """Draws an X/Y world-space pose panel.

    Args:
        points_world: World-space keypoints with shape ``(K, 3)`` or ``(N, K, 3)``.
        scores: Keypoint scores with shape ``(K,)`` or ``(N, K)``.
        metainfo: Skeleton metadata.
        panel_size: Panel size as ``(width, height)``.
        center: Optional X/Y panel center.
        radius: Optional X/Y panel radius.
        threshold: Minimum score required to draw a keypoint.

    Returns:
        np.ndarray: Rendered BGR panel.
    """
    width, height = panel_size
    panel = np.full((height, width, 3), 18, dtype=np.uint8)
    points_world = np.asarray(points_world, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    if points_world.ndim == 2:
        points_world = points_world[None, :, :]
        scores = scores[None, :]

    visible_xy = points_world[..., :2][scores >= threshold]
    if center is None:
        if len(visible_xy) == 0:
            center = np.zeros(2, dtype=np.float32)
        else:
            min_xy = visible_xy.min(axis=0)
            max_xy = visible_xy.max(axis=0)
            center = 0.5 * (min_xy + max_xy)
    else:
        center = np.asarray(center, dtype=np.float32)

    if radius is None:
        if len(visible_xy) == 0:
            radius = 1.0
        else:
            span = np.abs(visible_xy - center[None, :]).max()
            radius = max(1.0, float(span) * 1.2)

    def project(point: np.ndarray) -> tuple[int, int]:
        x = (float(point[0]) - float(center[0])) / (2.0 * radius) + 0.5
        y = 0.5 - (float(point[1]) - float(center[1])) / (2.0 * radius)
        return int(round(x * width)), int(round(y * height))

    ground_y = project(np.array([center[0], 0.0], dtype=np.float32))[1]
    if 0 <= ground_y < height:
        cv2.line(panel, (0, ground_y), (width, ground_y), (80, 80, 80), 1, cv2.LINE_AA)

    keypoint_info = metainfo["keypoint_info"]
    skeleton_info = metainfo["skeleton_info"]
    link_dict = {info["name"]: info["id"] for info in keypoint_info.values()}

    for pose_points, pose_scores in zip(points_world, scores):
        for ske_info in skeleton_info.values():
            pt0 = link_dict[ske_info["link"][0]]
            pt1 = link_dict[ske_info["link"][1]]
            if pose_scores[pt0] < threshold or pose_scores[pt1] < threshold:
                continue
            color = tuple(int(c) for c in ske_info["color"][::-1])
            cv2.line(
                panel,
                project(pose_points[pt0]),
                project(pose_points[pt1]),
                color,
                2,
                cv2.LINE_AA,
            )

        for joint_id, point in enumerate(pose_points):
            if pose_scores[joint_id] < threshold:
                continue
            color = tuple(int(c) for c in keypoint_info[joint_id]["color"][::-1])
            cv2.circle(panel, project(point), 4, color, -1, cv2.LINE_AA)

    cv2.putText(
        panel,
        "World X/Y",
        (14, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (230, 230, 230),
        2,
        cv2.LINE_AA,
    )
    return panel


def draw_skeleton(
    img, keypoints, scores, metainfo, kpt_thr=0.5, radius=2, line_width=2
):
    """Draws pose skeletons on an image.

    Args:
        img: Input image.
        keypoints: Keypoint coordinates for one or more poses.
        scores: Keypoint confidence scores.
        metainfo: Skeleton metadata.
        kpt_thr: Minimum score required to draw a keypoint.
        radius: Keypoint circle radius.
        line_width: Skeleton line width.

    Returns:
        The annotated image.
    """
    keypoint_info = metainfo["keypoint_info"]
    skeleton_info = metainfo["skeleton_info"]

    if len(keypoints.shape) == 2:
        keypoints = keypoints[None, :, :]
        scores = scores[None, :, :]

    num_instance = keypoints.shape[0]
    for i in range(num_instance):
        img = draw_mmpose(
            img,
            keypoints[i],
            scores[i],
            keypoint_info,
            skeleton_info,
            kpt_thr,
            radius,
            line_width,
        )
    return img


def draw_mmpose(
    img,
    keypoints,
    scores,
    keypoint_info,
    skeleton_info,
    kpt_thr=0.5,
    radius=2,
    line_width=2,
):
    """Draws a single pose using MMPose-style metadata.

    Args:
        img: Input image.
        keypoints: Keypoint coordinates for one pose.
        scores: Keypoint confidence scores for one pose.
        keypoint_info: Keypoint metadata mapping.
        skeleton_info: Skeleton edge metadata mapping.
        kpt_thr: Minimum score required to draw a keypoint.
        radius: Keypoint circle radius.
        line_width: Skeleton line width.

    Returns:
        The annotated image.
    """
    assert len(keypoints.shape) == 2

    vis_kpt = [s >= kpt_thr for s in scores]

    link_dict = {}
    for i, kpt_info in keypoint_info.items():
        link_dict[kpt_info["name"]] = kpt_info["id"]

    for i, ske_info in skeleton_info.items():
        link = ske_info["link"]
        link_color = ske_info["color"]
        pt0, pt1 = link_dict[link[0]], link_dict[link[1]]

        if vis_kpt[pt0] and vis_kpt[pt1]:
            kpt0 = keypoints[pt0]
            kpt1 = keypoints[pt1]

            img = cv2.line(
                img,
                (int(kpt0[0]), int(kpt0[1])),
                (int(kpt1[0]), int(kpt1[1])),
                link_color[::-1],
                thickness=line_width,
            )

    stroke_color = (255, 255, 255)  # White
    for i, kpt_info in keypoint_info.items():
        kpt = keypoints[i]
        fill_color = kpt_info["color"]

        if vis_kpt[i]:
            center = (int(kpt[0]), int(kpt[1]))

            # Draw outer circle (stroke)
            img = cv2.circle(
                img, center, int(radius + line_width), stroke_color[::-1], -1
            )

            # Draw inner circle (fill)
            img = cv2.circle(img, center, int(radius), fill_color[::-1], -1)

    return img
