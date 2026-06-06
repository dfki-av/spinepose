import argparse
import json
import os
import warnings
from pathlib import Path
from typing import List

import cv2
import numpy as np
from tqdm import tqdm

from spinepose.pose_estimator import SpinePoseEstimator
from spinepose.pose_tracker import PoseTracker
from spinepose._version import __version__


def infer_image(
    input_path,
    mode="medium",
    spine_only=False,
    vis_path=None,
    model_version="latest",
    detector="rfdetr",
    hardware_acceleration: bool = True,
    mixed_precision: bool = False,
) -> np.ndarray:
    """Runs pose estimation on a single image.

    Args:
        input_path: Path to the input image file.
        mode: Model size to use. One of: 'xlarge', 'large', 'medium', 'small'.
        detector: Detector to use. One of: 'rfdetr', 'yolox'.
        spine_only: Whether to include only spine keypoints.
        vis_path: Optional path to save the output visualization.
        model_version: Model version to use. One of: 'latest', 'v2', 'v1'.
        hardware_acceleration: Whether to use non-CPU execution providers when available.
        mixed_precision: Whether to enable lower-precision execution when supported.

    Returns:
        np.ndarray: Keypoints and scores in ``(N, K, 4)`` format, or an empty array.
    """
    model = SpinePoseEstimator(
        mode,
        detector=detector,
        model_version=model_version,
        hardware_acceleration=hardware_acceleration,
        mixed_precision=mixed_precision,
    )

    img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    keypoints, scores = model(img)

    if len(keypoints) == 0:
        return np.array([])

    if spine_only:
        spine_ids = model.SPINE_IDS
        non_spine_ids = list(set(range(len(scores[0]))) - set(spine_ids))
        scores[:, non_spine_ids] = 0
        keypoints[:, non_spine_ids, :] = 0

    # Create a visualization
    vis = model.visualize(img, keypoints, scores)
    if vis_path is None:
        _imshow(vis, "SpinePose Image Inference")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        cv2.imwrite(vis_path, vis)

    # Stack keypoints and scores for return
    results = np.concatenate([keypoints, scores[..., np.newaxis]], axis=-1)
    if spine_only:
        results = results[:, spine_ids, :]

    return results


def infer_video(
    input_path,
    mode="medium",
    spine_only=False,
    use_smoothing=True,
    vis_path=None,
    model_version="latest",
    detector="rfdetr",
    hardware_acceleration: bool = True,
    mixed_precision: bool = False,
) -> List[np.ndarray]:
    """Runs pose estimation on a video file.

    Args:
        input_path: Path to the input video file or 'webcam' for live video.
        mode: Model size to use. One of: 'xlarge', 'large', 'medium', 'small'.
        detector: Detector to use. One of: 'rfdetr', 'yolox'.
        spine_only: Whether to include only spine keypoints.
        use_smoothing: Whether to apply smoothing to keypoints over time.
        vis_path: Optional path to save the output video.
        model_version: Model version to use. One of: 'latest', 'v2', 'v1'.
        hardware_acceleration: Whether to use non-CPU execution providers when available.
        mixed_precision: Whether to enable lower-precision execution when supported.

    Returns:
        List[np.ndarray]: Per-frame keypoints and scores, including empty frames.
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
        smoothing=use_smoothing,
        smoothing_freq=fps,
        model_version=model_version,
        hardware_acceleration=hardware_acceleration,
        mixed_precision=mixed_precision,
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

            keypoints, scores = pose_tracker(img)

            if spine_only and len(scores) > 0:
                spine_ids = pose_tracker.solution.SPINE_IDS
                non_spine_ids = list(set(range(len(scores[0]))) - set(spine_ids))
                scores[:, non_spine_ids] = 0

            vis = pose_tracker.visualize(img, keypoints, scores)

            # Display the result
            _imshow(vis, "SpinePose Video Inference")
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            # Save the frame if requested
            if writer is not None:
                vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
                writer.append_data(vis_rgb)

            if len(keypoints) == 0:
                all_results.append(np.array([]))
                continue

            # Append frame results
            frame_results = np.concatenate(
                [keypoints, scores[..., np.newaxis]], axis=-1
            )
            if spine_only:
                frame_results = frame_results[:, spine_ids, :]
            all_results.append(frame_results)
        except KeyboardInterrupt:
            print("Inference interrupted by user.")
            break

    cap.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.close()

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


def _write_frame(keypoints_array, save_path):
    """Writes one frame of keypoints in OpenPose JSON format.

    Args:
        keypoints_array: Keypoints in ``(num_people, num_keypoints, 3)`` format.
        save_path: Output JSON path.
    """
    people = []

    if keypoints_array.size > 0:
        for person in keypoints_array:
            keypoints_list = person.reshape(-1).tolist()
            people.append({"pose_keypoints_2d": keypoints_list})

    output_data = {"version": 1.0, "people": people}

    with open(save_path, "w") as f:
        json.dump(output_data, f)


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
        help="Save predictions in OpenPose format (.json for image or folder for video).",
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
        )
    elif _is_video(args.input_path) or args.input_path.lower() == "webcam":
        image_mode = False
        results = infer_video(
            args.input_path,
            args.mode,
            detector=args.detector,
            spine_only=args.spine_only,
            use_smoothing=args.nosmooth,
            vis_path=args.vis_path,
            model_version=str(args.model_version),
            hardware_acceleration=args.hardware_acceleration,
            mixed_precision=args.mixed_precision,
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
            _write_frame(results, save_path)
            print(f"Results saved to {save_path}")

        else:
            save_path.mkdir(parents=True, exist_ok=True)
            for idx, frame_results in enumerate(tqdm(results, desc="Saving results")):
                _write_frame(frame_results, save_path / f"frame_{idx:05d}.json")
            print(f"Results saved to {save_path} ({len(results)} frames)")


if __name__ == "__main__":
    main()
