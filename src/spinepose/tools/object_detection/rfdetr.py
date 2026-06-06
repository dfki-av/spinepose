from __future__ import annotations

import cv2
import numpy as np

from ..base_tool import BaseTool


class RFDETR(BaseTool):
    imagenet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    imagenet_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    pixel_scale = np.float32(1.0 / 255.0)

    def __init__(
        self,
        onnx_model: str,
        model_input_size: tuple[int, int] = (576, 576),
        score_thr: float = 0.3,
        num_select: int = 300,
        class_ids: list[int] | None = [0],
        device: str = "cpu",
    ) -> None:
        """Initializes the RF-DETR detector.

        Args:
            onnx_model: Path to the ONNX model.
            model_input_size: Model input size as ``(height, width)``.
            score_thr: Minimum score required to keep a detection.
            num_select: Maximum number of candidates to rank per image.
            class_ids: Foreground class IDs to keep. Uses all classes when ``None``.
            device: Inference device name.
        """
        super().__init__(onnx_model, model_input_size, device=device)
        self.input_size = (int(model_input_size[0]), int(model_input_size[1]))
        if self.input_size[0] <= 0 or self.input_size[1] <= 0:
            raise ValueError(f"Invalid input_size: {self.input_size}")

        self.score_thr = score_thr
        self.num_select = num_select
        self.class_ids = class_ids if class_ids else None
        self.class_ids_np = (
            np.asarray(self.class_ids, dtype=np.int64)
            if self.class_ids is not None
            else None
        )

        self.input_name = self.session.get_inputs()[0].name
        self._imagenet_inv_std = (1.0 / self.imagenet_std).astype(np.float32)

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        """Applies the sigmoid function elementwise."""
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def _box_cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
        """Converts boxes from center-width-height to corner format."""
        cx = boxes[..., 0]
        cy = boxes[..., 1]
        w = np.clip(boxes[..., 2], a_min=0.0, a_max=None)
        h = np.clip(boxes[..., 3], a_min=0.0, a_max=None)
        return np.stack(
            [cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], axis=-1
        )

    @staticmethod
    def _empty_detections() -> dict[str, np.ndarray]:
        """Returns an empty detection payload."""
        return {
            "xyxy": np.empty((0, 4), dtype=np.float32),
            "confidence": np.empty((0,), dtype=np.float32),
        }

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Runs detection and returns boxes only."""
        return self.predict(image)["xyxy"]

    def predict(self, image: np.ndarray) -> dict[str, np.ndarray]:
        """Runs detection and returns boxes with confidence scores."""
        image, target_sizes = self.preprocess(image)
        outputs = self.inference(image)
        return self.postprocess(outputs, target_sizes, return_scores=True)

    def preprocess(self, image_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Resizes and normalizes an RGB image for inference.

        Args:
            image_rgb: Input RGB image with shape ``[H, W, 3]``.

        Returns:
            tuple[np.ndarray, np.ndarray]: Normalized image and original size tensor.
        """
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError("Expected image_rgb shape [H, W, 3].")

        original_h, original_w = image_rgb.shape[:2]
        resized = cv2.resize(
            image_rgb,
            (self.input_size[1], self.input_size[0]),
            interpolation=cv2.INTER_LINEAR,
        )

        image_array = resized.astype(np.float32)
        np.multiply(image_array, self.pixel_scale, out=image_array)
        np.subtract(image_array, self.imagenet_mean, out=image_array)
        np.multiply(image_array, self._imagenet_inv_std, out=image_array)
        target_sizes = np.array([[original_h, original_w]], dtype=np.float32)
        return image_array, target_sizes

    def postprocess(
        self,
        outputs: tuple[np.ndarray, ...],
        target_sizes: np.ndarray,
        return_scores: bool = False,
    ) -> np.ndarray | dict[str, np.ndarray]:
        """Converts model outputs into image-space detections.

        Args:
            outputs: Raw model outputs containing boxes and class logits.
            target_sizes: Original image sizes as ``[[height, width], ...]``.
            return_scores: When ``True``, returns boxes and confidences.

        Returns:
            np.ndarray | dict[str, np.ndarray]: Boxes only or a detection payload.
        """
        if not isinstance(outputs, (list, tuple)) or len(outputs) < 2:
            empty = self._empty_detections()
            return empty if return_scores else empty["xyxy"]

        out_bbox, out_logits = outputs

        if out_bbox.ndim != 3 or out_logits.ndim != 3:
            empty = self._empty_detections()
            return empty if return_scores else empty["xyxy"]

        probs = self._sigmoid(out_logits)
        # RF-DETR exports include a background class at index 0.
        # Exclude it from ranking so class IDs match YOLOX-style 0-based foreground IDs.
        probs[..., 0] = -1.0
        batch_size, _, num_classes = probs.shape

        flat_probs = probs.reshape(batch_size, -1)
        num_topk = min(self.num_select, flat_probs.shape[1])
        if num_topk <= 0:
            empty = self._empty_detections()
            return empty if return_scores else empty["xyxy"]

        if target_sizes.ndim != 2 or target_sizes.shape[1] != 2:
            empty = self._empty_detections()
            return empty if return_scores else empty["xyxy"]

        if (
            out_bbox.shape[0] != out_logits.shape[0]
            or out_logits.shape[0] != target_sizes.shape[0]
        ):
            empty = self._empty_detections()
            return empty if return_scores else empty["xyxy"]

        topk_unsorted_indices = np.argpartition(flat_probs, -num_topk, axis=1)[
            :, -num_topk:
        ]
        topk_unsorted_scores = np.take_along_axis(
            flat_probs, topk_unsorted_indices, axis=1
        )
        sort_order = np.argsort(-topk_unsorted_scores, axis=1)
        topk_indices = np.take_along_axis(topk_unsorted_indices, sort_order, axis=1)
        scores = np.take_along_axis(topk_unsorted_scores, sort_order, axis=1)

        topk_boxes = topk_indices // num_classes
        labels = topk_indices % num_classes
        mapped_labels = labels - 1

        boxes = self._box_cxcywh_to_xyxy(out_bbox)
        batch_indices = np.arange(batch_size)[:, None]
        boxes = boxes[batch_indices, topk_boxes]

        img_h = target_sizes[:, 0]
        img_w = target_sizes[:, 1]
        scale_factors = np.stack([img_w, img_h, img_w, img_h], axis=1)
        boxes = boxes * scale_factors[:, None, :]

        detections = []
        for i in range(batch_size):
            keep = scores[i] > self.score_thr
            keep &= labels[i] > 0
            if self.class_ids_np is not None:
                keep &= np.isin(mapped_labels[i], self.class_ids_np)

            detections.append(
                {
                    "xyxy": boxes[i][keep].astype(np.float32),
                    "confidence": scores[i][keep].astype(np.float32),
                }
            )

        kept_detections = [d for d in detections if len(d["xyxy"]) > 0]
        if not kept_detections:
            empty = self._empty_detections()
            return empty if return_scores else empty["xyxy"]

        merged_detections = {
            "xyxy": np.concatenate([d["xyxy"] for d in kept_detections], axis=0),
            "confidence": np.concatenate(
                [d["confidence"] for d in kept_detections], axis=0
            ),
        }
        return merged_detections if return_scores else merged_detections["xyxy"]
