import sys
import tempfile
import unittest
import warnings
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spinepose.tools.base_tool import BaseTool
from spinepose.tools.utils.deprecation import deprecated_arg
from spinepose.tools.utils.file import _get_cache_dir, extract_zip
from spinepose.tools.utils.multithreading import concurrent_forloop
from spinepose.tools.utils.types import BodyResult, Keypoint, PoseResult


class DummyTool(BaseTool):
    def __call__(self, *args, **kwargs):
        return args, kwargs


class _Meta:
    def __init__(self, name, shape):
        self.name = name
        self.shape = shape


class _FakeSession:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.runs = []

    def get_inputs(self):
        return [_Meta("input", self.input_shape)]

    def get_outputs(self):
        return [_Meta("output", None)]

    def run(self, output_names, feed):
        x = feed["input"]
        self.runs.append(x.copy())
        return [x.sum(axis=tuple(range(1, x.ndim)))]


class BaseToolTests(unittest.TestCase):
    def test_shape_dim_to_int_handles_static_and_dynamic_dims(self):
        self.assertEqual(DummyTool._shape_dim_to_int(3), 3)
        self.assertIsNone(DummyTool._shape_dim_to_int(0))
        self.assertIsNone(DummyTool._shape_dim_to_int("batch"))

    def test_infer_input_layout_uses_channel_position(self):
        tool = object.__new__(DummyTool)
        self.assertEqual(tool._infer_input_layout([1, 3, 224, 224]), "nchw")
        self.assertEqual(tool._infer_input_layout([1, 224, 224, 3]), "nhwc")
        self.assertEqual(tool._infer_input_layout(["N", "C", "H", "W"]), "nchw")
        self.assertEqual(tool._infer_input_layout(["N", "H", "W", "channel"]), "nhwc")

    def test_merge_batched_outputs_concatenates_by_output_index(self):
        merged = DummyTool._merge_batched_outputs(
            [
                (np.array([[1]]), np.array([[10]])),
                (np.array([[2]]), np.array([[20]])),
            ]
        )

        np.testing.assert_array_equal(merged[0], np.array([[1], [2]]))
        np.testing.assert_array_equal(merged[1], np.array([[10], [20]]))

    def test_inference_converts_bhwc_to_nchw_and_splits_static_batch(self):
        tool = object.__new__(DummyTool)
        tool.session = _FakeSession([1, 3, 2, 2])
        img = np.ones((2, 2, 2, 1), dtype=np.uint8)

        outputs = tool.inference(img)

        self.assertEqual(len(tool.session.runs), 2)
        self.assertEqual(tool.session.runs[0].shape, (1, 3, 2, 2))
        np.testing.assert_array_equal(
            outputs[0], np.array([12.0, 12.0], dtype=np.float32)
        )

    def test_inference_preserves_nhwc_layout_when_model_expects_it(self):
        tool = object.__new__(DummyTool)
        tool.session = _FakeSession([None, 2, 2, 3])
        img = np.ones((2, 2, 3), dtype=np.uint8)

        tool.inference(img)

        self.assertEqual(tool.session.runs[0].shape, (1, 2, 2, 3))


class DeprecationTests(unittest.TestCase):
    def test_deprecated_arg_without_replacement_removes_keyword(self):
        @deprecated_arg("backend")
        def func(**kwargs):
            return kwargs

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertEqual(func(backend="opencv"), {})

        self.assertEqual(len(caught), 1)
        self.assertIn("backend", str(caught[0].message))

    def test_deprecated_device_preserves_cpu_behavior(self):
        @deprecated_arg("device", "hardware_acceleration", default=True)
        def func(**kwargs):
            return kwargs

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.assertEqual(func(device="cpu"), {"hardware_acceleration": False})
            self.assertEqual(func(device="cuda"), {"hardware_acceleration": True})
            self.assertEqual(
                func(device="cpu", hardware_acceleration=True),
                {"hardware_acceleration": True},
            )


class UtilityTests(unittest.TestCase):
    def test_concurrent_forloop_returns_results_in_order(self):
        results = concurrent_forloop(lambda x, y: x + y, [1, 2, 3], [10, 20, 30])
        self.assertEqual(results, [11, 22, 33])

    def test_extract_zip_extracts_archive_contents(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            archive = tmp_path / "archive.zip"
            output_dir = tmp_path / "out"
            with zipfile.ZipFile(archive, "w") as zf:
                zf.writestr("nested/file.txt", "payload")

            extract_zip(archive, output_dir)

            self.assertEqual(
                (output_dir / "nested" / "file.txt").read_text(), "payload"
            )

    def test_get_cache_dir_points_to_spinepose_hub_cache(self):
        self.assertTrue(_get_cache_dir().endswith(".cache/spinepose/hub"))

    def test_named_tuple_types_store_pose_results(self):
        keypoint = Keypoint(1.0, 2.0, score=0.9, id=3)
        body = BodyResult([keypoint], total_score=0.9, total_parts=1)
        pose = PoseResult(body=body, left_hand=None, right_hand=None, face=None)

        self.assertEqual(pose.body.keypoints[0].id, 3)


if __name__ == "__main__":
    unittest.main()
