import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spinepose.tools.utils import session


class SessionUtilityTests(unittest.TestCase):
    def test_sha256_file_returns_content_digest(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.onnx"
            path.write_bytes(b"abc")

            self.assertEqual(
                session._sha256_file(str(path)),
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            )

    def test_resolve_execution_providers_prefers_accelerators_then_cpu(self):
        with patch.object(
            session,
            "_available_providers",
            return_value=[
                "CPUExecutionProvider",
                "CUDAExecutionProvider",
                "CoreMLExecutionProvider",
                "DmlExecutionProvider",
            ],
        ):
            providers = session.resolve_execution_providers()

        self.assertEqual(
            providers,
            [
                "CoreMLExecutionProvider",
                "CUDAExecutionProvider",
                "DmlExecutionProvider",
                "CPUExecutionProvider",
            ],
        )

    def test_resolve_execution_providers_can_force_cpu(self):
        self.assertEqual(
            session.resolve_execution_providers(hardware_acceleration=False),
            ["CPUExecutionProvider"],
        )

    def test_resolve_execution_providers_includes_tensorrt_when_enabled(self):
        with patch.object(
            session,
            "_available_providers",
            return_value=[
                "CPUExecutionProvider",
                "CUDAExecutionProvider",
                "TensorrtExecutionProvider",
            ],
        ):
            providers = session.resolve_execution_providers(tensor_rt=True)

        self.assertEqual(
            providers,
            [
                "TensorrtExecutionProvider",
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ],
        )

    def test_provider_options_use_expected_cache_directories(self):
        with patch.object(session, "_ep_cache_dir", return_value="/tmp/ep-cache"):
            coreml_entry = session.provider_options_for(
                "CoreMLExecutionProvider",
                onnx_model="/tmp/model.onnx",
                mixed_precision=True,
            )
            trt_entry = session.provider_options_for(
                "TensorrtExecutionProvider",
                onnx_model="/tmp/model.onnx",
                mixed_precision=True,
            )

        self.assertEqual(coreml_entry[0], "CoreMLExecutionProvider")
        self.assertEqual(coreml_entry[1]["ModelCacheDirectory"], "/tmp/ep-cache")
        self.assertEqual(coreml_entry[1]["AllowLowPrecisionAccumulationOnGPU"], "1")
        self.assertEqual(trt_entry[1]["trt_engine_cache_path"], "/tmp/ep-cache")
        self.assertTrue(trt_entry[1]["trt_fp16_enable"])

    def test_build_provider_entries_keeps_cpu_as_string_entry(self):
        entries = session.build_provider_entries(
            ["CPUExecutionProvider"],
            onnx_model="/tmp/model.onnx",
            mixed_precision=False,
        )

        self.assertEqual(entries, ["CPUExecutionProvider"])

    def test_create_ort_session_uses_prepared_coreml_model_path(self):
        fake_so = SimpleNamespace(graph_optimization_level=None)
        fake_session = object()

        with (
            patch.object(
                session,
                "resolve_execution_providers",
                return_value=["CoreMLExecutionProvider", "CPUExecutionProvider"],
            ),
            patch.object(
                session,
                "_prepare_coreml_model_copy",
                return_value="/cache/prepared.onnx",
            ) as prepare,
            patch.object(
                session,
                "build_provider_entries",
                return_value=["CPUExecutionProvider"],
            ) as build,
            patch.object(
                session.ort,
                "SessionOptions",
                return_value=fake_so,
            ),
            patch.object(
                session.ort,
                "InferenceSession",
                return_value=fake_session,
            ) as inference_session,
        ):
            created = session.create_ort_session("/models/source.onnx")

        self.assertIs(created, fake_session)
        prepare.assert_called_once_with("/models/source.onnx")
        build.assert_called_once_with(
            ["CoreMLExecutionProvider", "CPUExecutionProvider"],
            onnx_model="/cache/prepared.onnx",
            mixed_precision=True,
        )
        self.assertEqual(
            inference_session.call_args.kwargs["path_or_bytes"],
            "/cache/prepared.onnx",
        )

    def test_create_ort_session_retries_with_plain_providers_on_runtime_error(self):
        fake_so = SimpleNamespace(graph_optimization_level=None)
        fake_session = object()
        inference_session = Mock(
            side_effect=[RuntimeError("bad provider"), fake_session]
        )

        with (
            patch.object(
                session,
                "resolve_execution_providers",
                side_effect=[
                    ["CUDAExecutionProvider", "CPUExecutionProvider"],
                    ["CPUExecutionProvider"],
                ],
            ),
            patch.object(
                session,
                "build_provider_entries",
                return_value=[("CUDAExecutionProvider", {"bad": "option"})],
            ),
            patch.object(
                session.ort,
                "SessionOptions",
                return_value=fake_so,
            ),
            patch.object(
                session.ort,
                "InferenceSession",
                inference_session,
            ),
        ):
            created = session.create_ort_session("/models/source.onnx")

        self.assertIs(created, fake_session)
        self.assertEqual(inference_session.call_count, 2)
        self.assertEqual(
            inference_session.call_args.kwargs["providers"],
            ["CPUExecutionProvider"],
        )


if __name__ == "__main__":
    unittest.main()
