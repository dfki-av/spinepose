import numpy as np


def nms(boxes, scores, nms_thr):
    """Applies non-maximum suppression for one class.

    Args:
        boxes: Bounding boxes in ``xyxy`` format.
        scores: Confidence scores for the boxes.
        nms_thr: IoU threshold for suppression.

    Returns:
        list[int]: Indices of the boxes to keep.
    """
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= nms_thr)[0]
        order = order[inds + 1]

    return keep


def multiclass_nms(boxes, scores, nms_thr, score_thr):
    """Applies class-aware non-maximum suppression.

    Args:
        boxes: Bounding boxes in ``xyxy`` format.
        scores: Per-class confidence scores.
        nms_thr: IoU threshold for suppression.
        score_thr: Minimum class score required to keep a box.

    Returns:
        tuple: Final detections and the last set of kept indices, or ``(None, None)``.
    """
    final_dets = []
    num_classes = scores.shape[1]
    for cls_ind in range(num_classes):
        cls_scores = scores[:, cls_ind]
        valid_score_mask = cls_scores > score_thr
        if valid_score_mask.sum() == 0:
            continue
        valid_scores = cls_scores[valid_score_mask]
        valid_boxes = boxes[valid_score_mask]
        keep = nms(valid_boxes, valid_scores, nms_thr)
        if len(keep) > 0:
            cls_inds = np.ones((len(keep), 1)) * cls_ind
            dets = np.concatenate(
                [valid_boxes[keep], valid_scores[keep, None], cls_inds], 1
            )
            final_dets.append(dets)
    if len(final_dets) == 0:
        return None, None
    return np.concatenate(final_dets, 0), keep
