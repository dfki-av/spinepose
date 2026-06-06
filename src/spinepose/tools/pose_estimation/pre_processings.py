from typing import Tuple

import cv2
import numpy as np


def bbox_xyxy2cs(
    bbox: np.ndarray, padding: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Converts bounding boxes to center and scale.

    Args:
        bbox: Bounding box array in ``xyxy`` format.
        padding: Scale multiplier applied to the box size.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Bounding-box center and scale.
    """
    # convert single bbox from (4, ) to (1, 4)
    dim = bbox.ndim
    if dim == 1:
        bbox = bbox[None, :]

    # get bbox center and scale
    x1, y1, x2, y2 = np.hsplit(bbox, [1, 2, 3])
    center = np.hstack([x1 + x2, y1 + y2]) * 0.5
    scale = np.hstack([x2 - x1, y2 - y1]) * padding

    if dim == 1:
        center = center[0]
        scale = scale[0]

    return center, scale


def _rotate_point(pt: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rotates a 2D point.

    Args:
        pt: Point coordinates.
        angle_rad: Rotation angle in radians.

    Returns:
        np.ndarray: Rotated point.
    """
    sn, cs = np.sin(angle_rad), np.cos(angle_rad)
    rot_mat = np.array([[cs, -sn], [sn, cs]])
    return rot_mat @ pt


def _get_3rd_point(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Computes the third point used for affine transforms.

    Args:
        a: First point.
        b: Second point.

    Returns:
        np.ndarray: Derived third point.
    """
    direction = a - b
    return b + np.r_[-direction[1], direction[0]]


def get_warp_matrix(
    center: np.ndarray,
    scale: np.ndarray,
    rot: float,
    output_size: Tuple[int, int],
    shift: Tuple[float, float] = (0.0, 0.0),
    inv: bool = False,
) -> np.ndarray:
    """Calculates an affine warp matrix.

    Args:
        center: Bounding-box center.
        scale: Bounding-box scale.
        rot: Rotation angle in degrees.
        output_size: Output image size as ``(width, height)``.
        shift: Relative translation applied before warping.
        inv: Whether to invert the transform direction.

    Returns:
        np.ndarray: Affine transformation matrix.
    """

    shift = np.array(shift)
    src_w = scale[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    # compute transformation matrix
    rot_rad = np.deg2rad(rot)
    src_dir = _rotate_point(np.array([0.0, src_w * -0.5]), rot_rad)
    dst_dir = np.array([0.0, dst_w * -0.5])

    # get four corners of the src rectangle in the original image
    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale * shift
    src[1, :] = center + src_dir + scale * shift
    src[2, :] = _get_3rd_point(src[0, :], src[1, :])

    # get four corners of the dst rectangle in the input image
    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]

    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir
    dst[2, :] = _get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        warp_mat = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        warp_mat = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return warp_mat


def top_down_affine(
    input_size: dict, bbox_scale: dict, bbox_center: dict, img: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Applies top-down affine preprocessing to an image crop.

    Args:
        input_size: Model input size.
        bbox_scale: Bounding-box scale.
        bbox_center: Bounding-box center.
        img: Original image.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Warped image and adjusted scale.
    """
    w, h = input_size
    warp_size = (int(w), int(h))

    # reshape bbox to fixed aspect ratio
    aspect_ratio = w / h
    b_w, b_h = np.hsplit(bbox_scale, [1])
    bbox_scale = np.where(
        b_w > b_h * aspect_ratio,
        np.hstack([b_w, b_w / aspect_ratio]),
        np.hstack([b_h * aspect_ratio, b_h]),
    )

    # get the affine matrix
    center = bbox_center
    scale = bbox_scale
    rot = 0
    warp_mat = get_warp_matrix(center, scale, rot, output_size=(w, h))

    # do affine transform
    img = cv2.warpAffine(img, warp_mat, warp_size, flags=cv2.INTER_LINEAR)

    return img, bbox_scale
