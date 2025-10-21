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
            "pose": "https://huggingface.co/dfki-av/spinepose/resolve/main/spinepose-l_32xb256-10e_spinetrack-256x192.onnx",
            "pose_input_size": (192, 256),
        },
        "medium": {
            "det": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip",
            "det_input_size": (640, 640),
            "pose": "https://huggingface.co/dfki-av/spinepose/resolve/main/spinepose-m_32xb256-10e_spinetrack-256x192.onnx",
            "pose_input_size": (192, 256),
        },
        "small": {
            "det": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_tiny_8xb8-300e_humanart-6f3252f9.zip",
            "det_input_size": (416, 416),
            "pose": "https://huggingface.co/dfki-av/spinepose/resolve/main/spinepose-s_32xb256-10e_spinetrack-256x192.onnx",
            "pose_input_size": (192, 256),
        },
    }

    def __init__(
        self,
        mode: str = "large",
        backend="onnxruntime",
        device: str = "auto",
    ):
        super().__init__(
            metainfo,
            self.MODE,
            mode=mode,
            backend=backend,
            device=device,
        )

    def postprocess(self, keypoints, scores):
        keypoints, scores = self._smooth_spine(keypoints, scores)

        # Hide latissimus dorsi and clavicle keypoints
        # These keypoints are not used in the final output
        hidden_ids = [31, 32, 33, 34]
        scores[:, hidden_ids] = 0
        return keypoints, scores

    def _smooth_spine(self, keypoints, scores):
        """Smooth spine keypoints based on domain-specific rule."""
        spine_ids = [36, 35, 18, 30, 29, 28, 27, 26, 19]
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
