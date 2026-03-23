import warnings

from .tools.base_solution import BasePoseSolution
from .metainfo import metainfo


class SpinePoseEstimator(BasePoseSolution):
    """
    SpinePose: Body + Spine pose estimation using SpineTrack keypoints.

    Combines HALPE-26 keypoints with additional spine keypoints obtained
    from an auxiliary spine pose model.
    """

    MODE = {
        "xlarge": {
            "det": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_x_8xb8-300e_humanart-a39d44ed.zip",
            "det_input_size": (640, 640),
            "pose": "https://huggingface.co/dfki-av/spinepose/resolve/main/spinepose-x_32xb128-10e_spinetrack-384x288.onnx",
            "pose_input_size": (288, 384),
        },
        "large": {
            "det": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_x_8xb8-300e_humanart-a39d44ed.zip",
            "det_input_size": (640, 640),
            "pose": "https://huggingface.co/dfki-av/spinepose/resolve/main/spinepose-l_32xb256-10e_%s-256x192.onnx",
            "pose_input_size": (192, 256),
        },
        "medium": {
            "det": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip",
            "det_input_size": (640, 640),
            "pose": "https://huggingface.co/dfki-av/spinepose/resolve/main/spinepose-m_32xb256-10e_%s-256x192.onnx",
            "pose_input_size": (192, 256),
        },
        "small": {
            "det": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_tiny_8xb8-300e_humanart-6f3252f9.zip",
            "det_input_size": (416, 416),
            "pose": "https://huggingface.co/dfki-av/spinepose/resolve/main/spinepose-s_32xb256-10e_%s-256x192.onnx",
            "pose_input_size": (192, 256),
        },
    }

    SPINE_IDS = [19, 26, 27, 28, 29, 30, 18, 35, 36]

    def __init__(
        self,
        mode: str = "large",
        backend="onnxruntime",
        device: str = "auto",
        model_version: str = "latest",
    ):
        model_name = self._resolve_model_name(model_version)
        config = self.MODE
        for key in config:
            pose_model = config[key]["pose"]
            if "%s" in pose_model:
                config[key]["pose"] =pose_model % model_name

        super().__init__(
            metainfo,
            config,
            mode=mode,
            backend=backend,
            device=device,
        )
        self.version = model_version

    def _resolve_model_name(self, model_version: str) -> str:
        if model_version in ["latest", "v2"]:
            return "simspine"
        elif model_version == "v1":
            return "spinetrack"
        else:
            warnings.warn(f"Unknown model version '{model_version}', defaulting to 'simspine'")
            return "simspine"

    def postprocess(self, keypoints, scores):
        if self.version == "v1":
            keypoints, scores = self._smooth_spine(keypoints, scores)

        return keypoints, scores

    def _smooth_spine(self, keypoints, scores):
        """Smooth spine keypoints based on domain-specific rule."""
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
