# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, with an `Unreleased` section for work
that exists in development branches but has not yet been merged into a tagged
release on `main`.

## [Unreleased]

This section tracks changes after `v2.0.2` (`f0f378f`). These changes are not
part of any released PyPI version yet.

### Added
- RF-DETR and YOLOX detectors now expose a consistent `predict()` API and can
  optionally return confidence scores alongside bounding boxes.
- Added an experimental 2D-to-3D pose lifter that can return camera-space
  keypoints from `SpinePoseEstimator` when lifting is enabled.
- Added world-space 3D keypoint output through `PoseTracker`, including camera
  pose estimation and optional ground-plane alignment.
- Exposed pose lifting through `infer_image`, `infer_video`, and the CLI.
- Exposed the `PoseTracker` maximum detection limit through `infer_video` and
  the CLI.
- Added OpenPose-style `pose_keypoints_3d` JSON export and an OpenCV-based
  world-pose visualization panel for lifted video/image outputs.
- Added ONNX Runtime session utilities for provider selection, provider tuning,
  caching, and hardware-aware session creation.
- Added deprecation utilities for managing public API argument transitions.

### Changed
- Replaced the old `device` / `backend` execution selection model with
  `hardware_acceleration` and `mixed_precision`.
- Removed internal usage of unsupported OpenCV and OpenVINO backends.
- Exposed hardware acceleration and mixed-precision controls through the CLI and
  convenience inference helpers.
- CoreML session preparation now uses cached ONNX copies instead of mutating the
  source model file in place.
- `BaseTool.inference()` now accepts batched coordinate tensors for pose-lifting
  models in addition to image tensors.
- Inference helpers now release ONNX Runtime sessions before process shutdown to
  reduce provider cleanup warnings on accelerated backends.

### Deprecated
- Deprecated the public `backend` argument.
- Deprecated the public `device` argument in favor of `hardware_acceleration`.

### Documentation
- Added consistent Google-style docstrings across the package.
- Updated README content for the current inference surface and hardware control
  options.
- Documented the 2D-to-3D lifting workflow and OpenPose-style 3D export format.

## [2.0.2] - 2026-03-23

### Added
- Added detector selection in the CLI and Python API via
  `--detector rfdetr|yolox` and `detector="rfdetr"|"yolox"`.
- Integrated RF-DETR as an alternative person detector.
- Aligned RF-DETR with the existing YOLOX-style detection interface.

### Changed
- Updated release notes and user-facing documentation for multi-detector usage.

## [2.0.1] - 2026-03-23

### Added
- Added SpinePose V2 models trained on the SIMSPINE dataset.
- Added webcam support to `--input_path` for live camera inference.
- Added `--model-version` to the CLI to select `v1`, `v2`, or `latest`.
- Added `model_version` support to `SpinePoseEstimator`, `PoseTracker`,
  `infer_image`, and `infer_video`.
- Added model-zoo documentation for available V1 and V2 checkpoints.

### Changed
- Default predictions now include 37 keypoints instead of 33.
- Expanded the skeleton with four additional keypoints:
  `left_clavicle`, `right_clavicle`, `left_latissimus`, and
  `right_latissimus`.
- Refreshed documentation and package metadata for the V2 model line.

## [2.0.0] - 2025-10-21

### Added
- Added a more user-friendly public inference API.
- Added annotation export support.

### Changed
- Introduced the SpinePose 2.x release line with breaking changes relative to
  1.x.

### Documentation
- Refreshed project links, citations, and README content for the 2.0 release.

## [1.0.2] - 2025-06-11

### Fixed
- Corrected the `use_smoothing` parameter typo.
- Fixed the arXiv badge link.
- Updated version metadata for the release.

## [1.0.1] - 2025-04-11

### Added
- Added the `spine_only` option to `infer_image` and `infer_video`.

### Changed
- Updated keypoint fill coloring in pose visualization.

### Documentation
- Added SpineTrack download instructions.
- Fixed project links in the README.

## [1.0.0] - 2025-04-08

### Added
- Initial public PyPI release of SpinePose.
- Added the package `VERSION` file for runtime version discovery.

### Changed
- Updated packaging and requirements for ONNX Runtime-based inference.
- Added README guidance for GPU installation.
