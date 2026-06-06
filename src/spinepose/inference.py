from __future__ import annotations

import argparse
import gc
import json
import os
import warnings
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from spinepose._version import __version__
from spinepose.pose_estimator import SpinePoseEstimator
from spinepose.pose_tracker import PoseTracker
from spinepose.tools.visualization import draw_world_pose_panel


def _stack_2d_keypoints(keypoints: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Stacks 2D keypoints and scores into OpenPose-style arrays."""
    if len(keypoints) == 0:
        return np.array([])
    return np.concatenate([keypoints, scores[..., np.newaxis]], axis=-1)


def _stack_3d_keypoints(
    keypoints_3d: np.ndarray | None,
    scores: np.ndarray,
) -> np.ndarray:
    """Stacks 3D keypoints and scores into OpenPose-style arrays."""
    if keypoints_3d is None or len(keypoints_3d) == 0:
        return np.array([])
    return np.concatenate([keypoints_3d, scores[..., np.newaxis]], axis=-1)


def _filter_spine_only(
    keypoints: np.ndarray,
    scores: np.ndarray,
    keypoints_3d: np.ndarray | None,
    spine_ids: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Masks non-spine keypoints for visualization."""
    keypoints = keypoints.copy()
    scores = scores.copy()
    keypoints_3d = None if keypoints_3d is None else keypoints_3d.copy()
    if len(scores) == 0:
        return keypoints, scores, keypoints_3d

    non_spine_ids = list(set(range(len(scores[0]))) - set(spine_ids))
    scores[:, non_spine_ids] = 0
    keypoints[:, non_spine_ids, :] = 0
    if keypoints_3d is not None:
        keypoints_3d[:, non_spine_ids, :] = 0
    return keypoints, scores, keypoints_3d


def _slice_spine_only(results: np.ndarray, spine_ids: list[int]) -> np.ndarray:
    """Slices OpenPose-style results to the spine keypoint subset."""
    if results.size == 0:
        return results
    return results[:, spine_ids, :]


def _append_world_panel(
    image: np.ndarray,
    keypoints_3d: np.ndarray | None,
    scores: np.ndarray,
    metainfo: dict | None,
    enabled: bool,
) -> np.ndarray:
    """Appends a world-space X/Y panel when 3D keypoints are available."""
    if (
        not enabled
        or metainfo is None
        or keypoints_3d is None
        or len(keypoints_3d) == 0
    ):
        return image

    panel_width = max(320, image.shape[1] // 3)
    panel = draw_world_pose_panel(
        keypoints_3d,
        scores,
        metainfo,
        panel_size=(panel_width, image.shape[0]),
    )
    return np.concatenate([image, panel], axis=1)


def _release_model(model: object) -> None:
    """Releases model resources before process shutdown."""
    close = getattr(model, "close", None)
    if close is not None:
        close()
    gc.collect()


def infer_image(
    input_path: str,
    mode: str = "medium",
    spine_only: bool = False,
    vis_path: str | None = None,
    model_version: str = "latest",
    detector: str = "rfdetr",
    hardware_acceleration: bool = True,
    mixed_precision: bool = False,
    enable_lifting: bool = False,
    camera_intrinsics: np.ndarray | None = None,
    camera_field_of_view: float | None = 84.0,
    estimate_metric_scale: bool = True,
    estimate_camera_pose: bool = True,
    estimate_ground_plane: bool = True,
    primary_subject_height: float = 1.84,
    warmup_frames: int = 1,
    show_lifting_panel: bool = True,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Runs pose estimation on a single image.

    Args:
        input_path: Path to the input image file.
        mode: Model size to use. One of: 'xlarge', 'large', 'medium', 'small'.
        detector: Detector to use. One of: 'rfdetr', 'yolox'.
        spine_only: Whether to include only spine keypoints.
        vis_path: Optional path to save the output visualization.
        model_version: Model version to use. One of: 'latest', 'v2', 'v1'.
        hardware_acceleration: Whether to use non-CPU execution providers when
            available.
        mixed_precision: Whether to enable lower-precision execution when supported.
        enable_lifting: Whether to return lifted 3D keypoints.
        camera_intrinsics: Camera intrinsic matrix for 3D lifting.
        camera_field_of_view: Diagonal field of view used to estimate intrinsics.
        estimate_metric_scale: Whether to estimate metric scale from subject height.
        estimate_camera_pose: Whether to convert lifted keypoints into world space.
        estimate_ground_plane: Whether to align world orientation to the ground.
        primary_subject_height: Primary subject height in meters.
        warmup_frames: Number of frames used to stabilize camera calibration.
        show_lifting_panel: Whether to append a 3D world-pose panel to visualization.

    Returns:
        np.ndarray | tuple[np.ndarray, np.ndarray]: 2D keypoints in ``(N, K, 3)``
        format, or ``(keypoints_2d, keypoints_3d)`` when lifting is enabled.
    """
    img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if enable_lifting:
        model = PoseTracker(
            SpinePoseEstimator,
            mode=mode,
            detector=detector,
            tracking=False,
            smoothing=False,
            model_version=model_version,
            hardware_acceleration=hardware_acceleration,
            mixed_precision=mixed_precision,
            enable_lifting=True,
            camera_intrinsics=camera_intrinsics,
            camera_field_of_view=camera_field_of_view,
            estimate_metric_scale=estimate_metric_scale,
            estimate_camera_pose=estimate_camera_pose,
            estimate_ground_plane=estimate_ground_plane,
            primary_subject_height=primary_subject_height,
            warmup_frames=warmup_frames,
        )
    else:
        model = SpinePoseEstimator(
            mode,
            detector=detector,
            model_version=model_version,
            hardware_acceleration=hardware_acceleration,
            mixed_precision=mixed_precision,
        )

    model_results = model(img)
    if len(model_results) == 3:
        keypoints, scores, keypoints_3d = model_results
    else:
        keypoints, scores = model_results
        keypoints_3d = None

    if len(keypoints) == 0:
        empty = np.array([])
        _release_model(model)
        return (empty, empty) if enable_lifting else empty

    if spine_only:
        solution = model.solution if enable_lifting else model
        spine_ids = solution.SPINE_IDS
        vis_keypoints, vis_scores, vis_keypoints_3d = _filter_spine_only(
            keypoints,
            scores,
            keypoints_3d,
            spine_ids,
        )
    else:
        solution = model.solution if enable_lifting else model
        spine_ids = None
        vis_keypoints = keypoints
        vis_scores = scores
        vis_keypoints_3d = keypoints_3d

    vis = model.visualize(img, vis_keypoints, vis_scores)
    vis = _append_world_panel(
        vis,
        vis_keypoints_3d,
        vis_scores,
        getattr(solution, "metainfo", None),
        enable_lifting and show_lifting_panel,
    )
    if vis_path is None:
        _imshow(vis, "SpinePose Image Inference")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        cv2.imwrite(vis_path, vis)

    results = _stack_2d_keypoints(vis_keypoints, vis_scores)
    results_3d = _stack_3d_keypoints(vis_keypoints_3d, vis_scores)
    if spine_only:
        results = _slice_spine_only(results, spine_ids)
        results_3d = _slice_spine_only(results_3d, spine_ids)
    if enable_lifting:
        _release_model(model)
        return results, results_3d

    _release_model(model)
    return results


def infer_video(
    input_path: str,
    mode: str = "medium",
    spine_only: bool = False,
    use_smoothing: bool = True,
    vis_path: str | None = None,
    model_version: str = "latest",
    detector: str = "rfdetr",
    max_detections: int = 10,
    hardware_acceleration: bool = True,
    mixed_precision: bool = False,
    enable_lifting: bool = False,
    camera_intrinsics: np.ndarray | None = None,
    camera_field_of_view: float | None = 84.0,
    estimate_metric_scale: bool = True,
    estimate_camera_pose: bool = True,
    estimate_ground_plane: bool = True,
    primary_subject_height: float = 1.84,
    warmup_frames: int = 30,
    show_lifting_panel: bool = True,
) -> list[np.ndarray | tuple[np.ndarray, np.ndarray]]:
    """Runs pose estimation on a video file.

    Args:
        input_path: Path to the input video file or 'webcam' for live video.
        mode: Model size to use. One of: 'xlarge', 'large', 'medium', 'small'.
        detector: Detector to use. One of: 'rfdetr', 'yolox'.
        spine_only: Whether to include only spine keypoints.
        use_smoothing: Whether to apply smoothing to keypoints over time.
        vis_path: Optional path to save the output video.
        model_version: Model version to use. One of: 'latest', 'v2', 'v1'.
        hardware_acceleration: Whether to use non-CPU execution providers when
            available.
        max_detections: Maximum number of detected people to track per frame.
        mixed_precision: Whether to enable lower-precision execution when supported.
        enable_lifting: Whether to return lifted 3D keypoints.
        camera_intrinsics: Camera intrinsic matrix for 3D lifting.
        camera_field_of_view: Diagonal field of view used to estimate intrinsics.
        estimate_metric_scale: Whether to estimate metric scale from subject height.
        estimate_camera_pose: Whether to convert lifted keypoints into world space.
        estimate_ground_plane: Whether to align world orientation to the ground.
        primary_subject_height: Primary subject height in meters.
        warmup_frames: Number of frames used to stabilize camera calibration.
        show_lifting_panel: Whether to append a 3D world-pose panel to visualization.

    Returns:
        list: Per-frame 2D keypoints, or ``(keypoints_2d, keypoints_3d)``
        tuples when lifting is enabled.
    """
    if input_path.lower() == "webcam":
        input_path = 0  # OpenCV uses 0 for the default webcam

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        warnings.warn("FPS not detected, defaulting to 30")
        fps = 30.0

    pose_tracker = PoseTracker(
        SpinePoseEstimator,
        mode=mode,
        detector=detector,
        max_detections=max_detections,
        smoothing=use_smoothing,
        smoothing_freq=fps,
        model_version=model_version,
        hardware_acceleration=hardware_acceleration,
        mixed_precision=mixed_precision,
        enable_lifting=enable_lifting,
        camera_intrinsics=camera_intrinsics,
        camera_field_of_view=camera_field_of_view,
        estimate_metric_scale=estimate_metric_scale,
        estimate_camera_pose=estimate_camera_pose,
        estimate_ground_plane=estimate_ground_plane,
        primary_subject_height=primary_subject_height,
        warmup_frames=warmup_frames,
    )

    writer = None
    if vis_path is not None:
        try:
            import imageio

            writer = imageio.get_writer(vis_path, fps=int(fps))
        except ImportError:
            warnings.warn(
                "Please run `pip install imageio[ffmpeg]` to enable video saving."
                " The video will not be saved.",
                UserWarning,
            )
            writer = None

    all_results = []
    while True:
        try:
            ret, img = cap.read()
            if not ret:
                break

            tracker_results = pose_tracker(img)
            if len(tracker_results) == 3:
                keypoints, scores, keypoints_3d = tracker_results
            else:
                keypoints, scores = tracker_results
                keypoints_3d = None

            if spine_only and len(scores) > 0:
                spine_ids = pose_tracker.solution.SPINE_IDS
                vis_keypoints, vis_scores, vis_keypoints_3d = _filter_spine_only(
                    keypoints,
                    scores,
                    keypoints_3d,
                    spine_ids,
                )
            else:
                spine_ids = None
                vis_keypoints = keypoints
                vis_scores = scores
                vis_keypoints_3d = keypoints_3d

            vis = pose_tracker.visualize(img, vis_keypoints, vis_scores)
            vis = _append_world_panel(
                vis,
                vis_keypoints_3d,
                vis_scores,
                getattr(pose_tracker.solution, "metainfo", None),
                enable_lifting and show_lifting_panel,
            )

            # Display the result
            _imshow(vis, "SpinePose Video Inference")
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            # Save the frame if requested
            if writer is not None:
                vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
                writer.append_data(vis_rgb)

            if len(keypoints) == 0:
                empty = np.array([])
                all_results.append((empty, empty) if enable_lifting else empty)
                continue

            frame_results = _stack_2d_keypoints(vis_keypoints, vis_scores)
            frame_results_3d = _stack_3d_keypoints(vis_keypoints_3d, vis_scores)
            if spine_only:
                frame_results = _slice_spine_only(frame_results, spine_ids)
                frame_results_3d = _slice_spine_only(frame_results_3d, spine_ids)

            if enable_lifting:
                all_results.append((frame_results, frame_results_3d))
            else:
                all_results.append(frame_results)
        except KeyboardInterrupt:
            print("Inference interrupted by user.")
            break

    cap.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.close()
    _release_model(pose_tracker)

    return all_results


def _imshow(img, title="Image"):
    """Displays an image resized to a maximum dimension of 1024 pixels.

    Args:
        img: Image to display.
        title: Window title.
    """
    h, w = img.shape[:2]
    scale = 1024 / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    img = cv2.resize(img, (new_w, new_h))
    cv2.imshow(title, img)


def _write_frame(
    keypoints_array: np.ndarray,
    save_path: str | Path,
    keypoints_3d_array: np.ndarray | None = None,
) -> None:
    """Writes one frame of keypoints in OpenPose JSON format.

    Args:
        keypoints_array: Keypoints in ``(num_people, num_keypoints, 3)`` format.
        save_path: Output JSON path.
        keypoints_3d_array: Optional 3D keypoints in ``(N, K, 4)`` format.
    """
    people = []

    if keypoints_array.size > 0:
        for person_id, person in enumerate(keypoints_array):
            person_data = {"pose_keypoints_2d": person.reshape(-1).tolist()}
            if keypoints_3d_array is not None and keypoints_3d_array.size > 0:
                person_data["pose_keypoints_3d"] = (
                    keypoints_3d_array[person_id].reshape(-1).tolist()
                )
            people.append(person_data)

    output_data = {"version": 1.0, "people": people}

    with open(save_path, "w") as f:
        json.dump(output_data, f)


def _write_result_frame(
    frame_results: np.ndarray | tuple[np.ndarray, np.ndarray],
    save_path: str | Path,
) -> None:
    """Writes 2D-only or 2D+3D frame results."""
    if isinstance(frame_results, tuple):
        _write_frame(frame_results[0], save_path, frame_results[1])
    else:
        _write_frame(frame_results, save_path)


def _exists(filepath):
    """Checks whether a file exists.

    Args:
        filepath: File path to check.

    Returns:
        bool: ``True`` if the file exists.
    """
    return os.path.isfile(filepath)


def _is_valid(filepath, formats):
    """Checks whether a file has an allowed extension.

    Args:
        filepath: File path to check.
        formats: Allowed file extensions.

    Returns:
        bool: ``True`` if the file extension is allowed.
    """
    _, ext = os.path.splitext(filepath)
    return ext.lower() in formats


def _is_image(filename):
    """Checks whether a path points to a supported image file.

    Args:
        filename: File path to check.

    Returns:
        bool: ``True`` if the file is a supported image.
    """
    img_exts = [".jpg", ".jpeg", ".png", ".bmp"]
    return _exists(filename) and _is_valid(filename, img_exts)


def _is_video(filename):
    """Checks whether a path points to a supported video file.

    Args:
        filename: File path to check.

    Returns:
        bool: ``True`` if the file is a supported video.
    """
    video_exts = [".mp4", ".avi", ".mov", ".mkv"]
    return _exists(filename) and _is_valid(filename, video_exts)


def main():
    """Parses CLI arguments and runs image or video inference."""
    parser = argparse.ArgumentParser(description="SpinePose Inference")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--version", "-V", action="store_true", help="Print the version and exit."
    )
    group.add_argument(
        "--input_path",
        "-i",
        type=str,
        help="Path to the input image or video",
    )
    parser.add_argument(
        "--vis-path",
        "-o",
        type=str,
        default=None,
        help="Path to save the output image or video",
    )
    parser.add_argument(
        "--save-path",
        "-s",
        type=str,
        default=None,
        help=(
            "Save predictions in OpenPose format (.json for image or folder for video)."
        ),
    )
    parser.add_argument(
        "--mode",
        "-m",
        choices=["xlarge", "large", "medium", "small"],
        default="medium",
        help="Model size. Choose from: xlarge, large, medium, small (default: medium)",
    )
    parser.add_argument(
        "--detector",
        type=str,
        choices=["rfdetr", "yolox"],
        default="rfdetr",
        help="Detector backend. One of: 'rfdetr', 'yolox' (default: rfdetr)",
    )
    parser.add_argument(
        "--max-detections",
        type=int,
        default=10,
        help="Maximum number of detected people to track per video frame.",
    )
    parser.add_argument(
        "--hardware-acceleration",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable non-CPU execution providers when available (default: enabled)",
    )
    parser.add_argument(
        "--mixed-precision",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable lower-precision execution when supported (default: disabled)",
    )
    parser.add_argument(
        "--enable-lifting",
        action="store_true",
        help="Enable 2D-to-3D pose lifting and return world-space 3D keypoints.",
    )
    parser.add_argument(
        "--camera-field-of-view",
        type=float,
        default=84.0,
        help="Diagonal camera field of view in degrees for intrinsic estimation.",
    )
    parser.add_argument(
        "--no-metric-scale",
        dest="estimate_metric_scale",
        action="store_false",
        default=True,
        help="Disable metric scale estimation from the primary subject height.",
    )
    parser.add_argument(
        "--no-camera-pose",
        dest="estimate_camera_pose",
        action="store_false",
        default=True,
        help=(
            "Keep lifted keypoints in camera orientation instead of world orientation."
        ),
    )
    parser.add_argument(
        "--no-ground-plane",
        dest="estimate_ground_plane",
        action="store_false",
        default=True,
        help="Disable ground-plane alignment for world orientation.",
    )
    parser.add_argument(
        "--primary-subject-height",
        type=float,
        default=1.84,
        help="Primary subject height in meters for metric scale estimation.",
    )
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=30,
        help="Number of video frames used to stabilize camera calibration.",
    )
    parser.add_argument(
        "--no-lifting-panel",
        dest="show_lifting_panel",
        action="store_false",
        default=True,
        help="Do not append the 3D world-pose panel when lifting is enabled.",
    )
    parser.add_argument(
        "--nosmooth",
        action="store_false",
        help="Disable keypoint smoothing for video inference (default: enabled)",
    )
    parser.add_argument(
        "--spine-only",
        action="store_true",
        help="Only use 9 spine keypoints (default: use all 37 keypoints)",
    )
    parser.add_argument(
        "--model-version",
        type=str,
        default="latest",
        help="Model version to use. One of: 'latest', 'v2', 'v1' (default: latest)",
    )
    args = parser.parse_args()

    if args.version:
        print(f"SpinePose {__version__}")
        return

    # Check if the input path is a valid image or video
    if _is_image(args.input_path):
        image_mode = True
        results = infer_image(
            args.input_path,
            args.mode,
            detector=args.detector,
            spine_only=args.spine_only,
            vis_path=args.vis_path,
            model_version=str(args.model_version),
            hardware_acceleration=args.hardware_acceleration,
            mixed_precision=args.mixed_precision,
            enable_lifting=args.enable_lifting,
            camera_field_of_view=args.camera_field_of_view,
            estimate_metric_scale=args.estimate_metric_scale,
            estimate_camera_pose=args.estimate_camera_pose,
            estimate_ground_plane=args.estimate_ground_plane,
            primary_subject_height=args.primary_subject_height,
            warmup_frames=args.warmup_frames,
            show_lifting_panel=args.show_lifting_panel,
        )
    elif _is_video(args.input_path) or args.input_path.lower() == "webcam":
        image_mode = False
        results = infer_video(
            args.input_path,
            args.mode,
            detector=args.detector,
            max_detections=args.max_detections,
            spine_only=args.spine_only,
            use_smoothing=args.nosmooth,
            vis_path=args.vis_path,
            model_version=str(args.model_version),
            hardware_acceleration=args.hardware_acceleration,
            mixed_precision=args.mixed_precision,
            enable_lifting=args.enable_lifting,
            camera_field_of_view=args.camera_field_of_view,
            estimate_metric_scale=args.estimate_metric_scale,
            estimate_camera_pose=args.estimate_camera_pose,
            estimate_ground_plane=args.estimate_ground_plane,
            primary_subject_height=args.primary_subject_height,
            warmup_frames=args.warmup_frames,
            show_lifting_panel=args.show_lifting_panel,
        )
    else:
        raise ValueError("Input path must be a valid image or video file.")

    # Save the results if a save path is provided
    if args.save_path is not None:
        save_path = Path(args.save_path)
        if image_mode and save_path.suffix.lower() != ".json":
            raise ValueError("Save path must be a JSON file.")

        if image_mode:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            _write_result_frame(results, save_path)
            print(f"Results saved to {save_path}")

        else:
            save_path.mkdir(parents=True, exist_ok=True)
            for idx, frame_results in enumerate(tqdm(results, desc="Saving results")):
                _write_result_frame(frame_results, save_path / f"frame_{idx:05d}.json")
            print(f"Results saved to {save_path} ({len(results)} frames)")


if __name__ == "__main__":
    main()
