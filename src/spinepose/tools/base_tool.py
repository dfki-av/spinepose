"""Utilities for the ``posetrack.models.base`` module."""

from __future__ import annotations

import logging
import os
from abc import ABCMeta, abstractmethod
from typing import Any, Optional, Tuple

import numpy as np

from .utils.file import download_checkpoint
from .utils.deprecation import deprecated_arg
from .utils.session import create_ort_session


class BaseTool(metaclass=ABCMeta):
    """Base class for ONNX-backed tool components."""

    @deprecated_arg("backend")
    @deprecated_arg("device", "hardware_acceleration", default=True)
    def __init__(
        self,
        onnx_model: Optional[str] = None,
        model_input_size: Optional[Tuple[int, int]] = None,
        mean: Optional[Tuple[float, ...]] = None,
        std: Optional[Tuple[float, ...]] = None,
        hardware_acceleration: bool = True,
        mixed_precision: bool = False,
    ):
        """Initializes the base tool.

        Args:
            onnx_model: Path or URL to the ONNX model.
            model_input_size: Model input size as ``(height, width)``.
            mean: Optional channel-wise normalization mean.
            std: Optional channel-wise normalization standard deviation.
            hardware_acceleration: Whether to use non-CPU execution providers.
            mixed_precision: Whether to enable lower-precision execution.
        """
        if not os.path.exists(onnx_model):
            onnx_model = download_checkpoint(onnx_model)

        self.session = create_ort_session(
            model_path=onnx_model,
            hardware_acceleration=hardware_acceleration,
            tensor_rt=False,
            mixed_precision=mixed_precision,
        )

        model_name = os.path.splitext(os.path.basename(onnx_model))[0]
        prov_list = getattr(self.session, "get_providers", list)()
        logging.info(
            f"Model '{model_name}' initialized. Primary EP: '{prov_list[0] if prov_list else 'unknown'}'. "
            f"Providers in use: {prov_list}"
        )

        self.onnx_model = onnx_model
        self.model_input_size = model_input_size
        self.mean = np.array(mean) if mean is not None else None
        self.std = np.array(std) if std is not None else None

    @abstractmethod
    def __call__(self, *args, **kwargs) -> Any:
        """Runs the tool-specific forward pass.

        Args:
            *args: Positional arguments for the tool.
            **kwargs: Keyword arguments for the tool.

        Returns:
            Any: Tool-specific outputs.
        """
        raise NotImplementedError

    def close(self) -> None:
        """Releases the underlying ONNX Runtime session reference."""
        self.session = None

    @staticmethod
    def _shape_dim_to_int(value):
        """Converts a static shape dimension to an integer.

        Args:
            value: Shape dimension value.

        Returns:
            Any: Positive integer dimension or ``None``.
        """
        if isinstance(value, int) and value > 0:
            return value
        return None

    def _infer_input_layout(self, input_shape) -> str:
        """Infers the model input layout.

        Args:
            input_shape: Input tensor shape metadata.

        Returns:
            str: Either ``"nchw"`` or ``"nhwc"``.
        """
        # Default to NCHW to preserve existing behavior for legacy models.
        if not isinstance(input_shape, (list, tuple)) or len(input_shape) != 4:
            return "nchw"

        dim1 = self._shape_dim_to_int(input_shape[1])
        dim3 = self._shape_dim_to_int(input_shape[3])
        if dim1 in (1, 3, 4) and dim3 not in (1, 3, 4):
            return "nchw"
        if dim3 in (1, 3, 4) and dim1 not in (1, 3, 4):
            return "nhwc"

        # Fallback hint from symbolic axis names.
        name1 = str(input_shape[1]).lower()
        name3 = str(input_shape[3]).lower()
        if "channel" in name1:
            return "nchw"
        if "channel" in name3:
            return "nhwc"
        return "nchw"

    @staticmethod
    def _merge_batched_outputs(chunk_outputs):
        """Merges per-item outputs into batched outputs.

        Args:
            chunk_outputs: Sequence of outputs from single-item runs.

        Returns:
            Any: Batched output tuple.
        """
        if len(chunk_outputs) == 0:
            return tuple()
        num_outputs = len(chunk_outputs[0])
        merged = []
        for i in range(num_outputs):
            parts = [chunk[i] for chunk in chunk_outputs]
            merged.append(np.concatenate(parts, axis=0))
        return tuple(merged)

    def _run_session(self, x: np.ndarray):
        """Runs the ONNX session.

        Args:
            x: Batched model input.

        Returns:
            Any: Session outputs.
        """
        input_meta = self.session.get_inputs()[0]
        input_name = input_meta.name
        output_names = [o.name for o in self.session.get_outputs()]

        expected_batch = (
            self._shape_dim_to_int(input_meta.shape[0])
            if len(input_meta.shape) > 0
            else None
        )
        if expected_batch == 1 and x.shape[0] != 1:
            outputs = []
            for i in range(x.shape[0]):
                outputs.append(
                    self.session.run(output_names, {input_name: x[i : i + 1]})
                )
            return self._merge_batched_outputs(outputs)

        return tuple(self.session.run(output_names, {input_name: x}))

    def inference(self, img: np.ndarray):
        """Runs inference on an image or coordinate batch.

        Args:
            img: Input image in HWC/BHWC format, or coordinate batch with shape
                ``(N, K, 2)``.

        Returns:
            Any: Session outputs.

        Raises:
            ValueError: If the input shape is not supported.
        """
        if img.ndim == 3 and img.shape[-1] == 2:
            return self._run_session(img.astype(np.float32, copy=False))

        if img.ndim not in [3, 4] or img.shape[-1] not in (1, 3, 4):
            raise ValueError(
                f"Expected HxWxC/BHxWxC image or NxKx2 coordinates, got {img.shape}"
            )

        # Normalize to BHWC first.
        if img.ndim == 3:
            x_bhwc = np.expand_dims(img, axis=0)
        else:
            x_bhwc = img

        # If 4 channels (e.g., RGBA), drop alpha
        if x_bhwc.shape[-1] == 4:
            x_bhwc = x_bhwc[..., :3]
        # If 1 channel (e.g., grayscale), repeat channel
        elif x_bhwc.shape[-1] == 1:
            x_bhwc = np.repeat(x_bhwc, 3, axis=-1)

        input_shape = self.session.get_inputs()[0].shape
        layout = self._infer_input_layout(input_shape)
        if layout == "nhwc":
            x = x_bhwc.astype(np.float32, copy=False)
        else:
            x = x_bhwc.transpose(0, 3, 1, 2).astype(np.float32, copy=False)

        return self._run_session(x)
