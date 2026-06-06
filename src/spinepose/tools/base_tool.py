import logging
import os
from abc import ABCMeta, abstractmethod
from typing import Any

import numpy as np
import onnxruntime as ort

from .utils.file import download_checkpoint


def check_mps_support():
    providers = ort.get_available_providers()
    return "CoreMLExecutionProvider" in providers


RTMLIB_SETTINGS = {
    "cpu": "CPUExecutionProvider",
    "cuda": "CUDAExecutionProvider",
    "rocm": "ROCMExecutionProvider",
    "mps": "CoreMLExecutionProvider" if check_mps_support() else "CPUExecutionProvider",
}


class BaseTool(metaclass=ABCMeta):
    def __init__(
        self,
        onnx_model: str = None,
        model_input_size: tuple = None,
        mean: tuple = None,
        std: tuple = None,
        backend: str = "onnxruntime",
        device: str = "cpu",
    ):
        if not os.path.exists(onnx_model):
            onnx_model = download_checkpoint(onnx_model)

        providers = RTMLIB_SETTINGS[device]
        self.session = ort.InferenceSession(
            path_or_bytes=onnx_model, providers=[providers]
        )

        logging.info(f"load {onnx_model} with {backend} backend")

        self.onnx_model = onnx_model
        self.model_input_size = model_input_size
        self.mean = mean
        self.std = std
        self.backend = backend
        self.device = device

    @abstractmethod
    def __call__(self, *args, **kwargs) -> Any:
        """Implement the actual function here."""
        raise NotImplementedError

    def inference(self, img: np.ndarray):
        """Inference model.

        Args:
            img (np.ndarray): Input image in shape.

        Returns:
            outputs (np.ndarray): Output of RTMPose model.
        """
        # build input to (1, 3, H, W)
        img = img.transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        input = img[None, :, :, :]

        # run model
        sess_input = {self.session.get_inputs()[0].name: input}
        sess_output = []
        for out in self.session.get_outputs():
            sess_output.append(out.name)

        outputs = self.session.run(sess_output, sess_input)

        return outputs
