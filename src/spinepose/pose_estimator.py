import warnings
from copy import deepcopy

from .tools.base_solution import BasePoseSolution
from .metainfo import metainfo


class SpinePoseEstimator(BasePoseSolution):
    """Spine-aware pose estimator built on top of the base solution."""

    MODE = {
        "xlarge": {
            "det_yolox": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_x_8xb8-300e_humanart-a39d44ed.zip",
            "det_yolox_input_size": (640, 640),
            "det_rfdetr": "https://huggingface.co/saifkhichi96/opendetect/resolve/main/rfdetr/rfdetr_l_v142_704x704.onnx",
            "det_rfdetr_input_size": (704, 704),
            "pose": "https://huggingface.co/dfki-av/spinepose/resolve/main/spinepose-x_32xb128-10e_spinetrack-384x288.onnx",
            "pose_input_size": (288, 384),
        },
        "large": {
            "det_yolox": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_x_8xb8-300e_humanart-a39d44ed.zip",
            "det_yolox_input_size": (640, 640),
            "det_rfdetr": "https://huggingface.co/saifkhichi96/opendetect/resolve/main/rfdetr/rfdetr_m_v142_576x576.onnx",
            "det_rfdetr_input_size": (576, 576),
            "pose": "https://huggingface.co/dfki-av/spinepose/resolve/main/spinepose-l_32xb256-10e_%s-256x192.onnx",
            "pose_input_size": (192, 256),
        },
        "medium": {
            "det_yolox": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip",
            "det_yolox_input_size": (640, 640),
            "det_rfdetr": "https://huggingface.co/saifkhichi96/opendetect/resolve/main/rfdetr/rfdetr_s_v142_512x512.onnx",
            "det_rfdetr_input_size": (512, 512),
            "pose": "https://huggingface.co/dfki-av/spinepose/resolve/main/spinepose-m_32xb256-10e_%s-256x192.onnx",
            "pose_input_size": (192, 256),
        },
        "small": {
            "det_yolox": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_tiny_8xb8-300e_humanart-6f3252f9.zip",
            "det_yolox_input_size": (416, 416),
            "det_rfdetr": "https://huggingface.co/saifkhichi96/opendetect/resolve/main/rfdetr/rfdetr_n_v142_384x384.onnx",
            "det_rfdetr_input_size": (384, 384),
            "pose": "https://huggingface.co/dfki-av/spinepose/resolve/main/spinepose-s_32xb256-10e_%s-256x192.onnx",
            "pose_input_size": (192, 256),
        },
    }

    SPINE_IDS = [19, 26, 27, 28, 29, 30, 18, 35, 36]

    def __init__(
        self,
        mode: str = "large",
        model_version: str = "latest",
        detector: str = "rfdetr",
        **kwargs,
    ):
        """Initializes the SpinePose estimator.

        Args:
            mode: Model preset to load.
            model_version: Model version to use.
            detector: Detector name to use.
            **kwargs: Additional arguments forwarded to the base solution.
        """
        model_name = self._resolve_model_name(model_version)
        config = deepcopy(self.MODE)
        for key in config:
            pose_model = config[key]["pose"]
            if "%s" in pose_model:
                config[key]["pose"] = pose_model % model_name

        super().__init__(
            metainfo,
            config,
            mode=mode,
            detector=detector,
            **kwargs,
        )
        self.version = model_version

    def _resolve_model_name(self, model_version: str) -> str:
        """Maps a public model version to the underlying model family.

        Args:
            model_version: User-facing model version string.

        Returns:
            str: Internal model family name.
        """
        if model_version in ["latest", "v2"]:
            return "simspine"
        elif model_version == "v1":
            return "spinetrack"
        else:
            warnings.warn(
                f"Unknown model version '{model_version}', defaulting to 'simspine'"
            )
            return "simspine"

    def postprocess(self, keypoints, scores):
        """Postprocesses predicted keypoints and scores.

        Args:
            keypoints: Predicted keypoints.
            scores: Predicted confidence scores.

        Returns:
            tuple: Postprocessed keypoints and scores.
        """
        if self.version == "v1":
            keypoints, scores = self._smooth_spine(keypoints, scores)

        return keypoints, scores

    def _smooth_spine(self, keypoints, scores):
        """Applies a simple smoothing rule to spine keypoints.

        Args:
            keypoints: Predicted keypoints.
            scores: Predicted confidence scores.

        Returns:
            tuple: Smoothed keypoints and original scores.
        """
        spine_ids = self.SPINE_IDS[:9]
        spine_keypoints = keypoints[:, spine_ids]
        spine_scores = scores[:, spine_ids]

        # Smooth by averaging consecutive points
        for i in range(1, len(spine_keypoints[0]) - 1):
            spine_keypoints[:, i] = (
                spine_keypoints[:, i - 1] + spine_keypoints[:, i + 1]
            ) / 2

        # Replace in global keypoints
        keypoints[:, spine_ids] = spine_keypoints
        scores[:, spine_ids] = spine_scores

        return keypoints, scores
