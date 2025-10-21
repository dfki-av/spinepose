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
) -> np.ndarray:
    """Perform pose estimation on a single image.

    Args:
        input_path: Path to the input image file.
        mode: Model size to use. One of: 'xlarge', 'large', 'medium', 'small'.
        spine_only: Whether to include only spine keypoints.
        vis_path: Optional path to save the output visualization.

    Returns:
        A NumPy array of shape (1, N, 4) containing keypoints and scores,
        or an empty array if no keypoints are detected.
    """
    model = SpinePoseEstimator(mode)

    img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    keypoints, scores = model(img)

    if len(keypoints) == 0:
        return np.array([])

    if spine_only:
        spine_ids = [36, 35, 18, 30, 29, 28, 27, 26, 19]
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
) -> List[np.ndarray]:
    """Perform pose estimation on a video file.

    Args:
        input_path: Path to the input video file.
        mode: Model size to use. One of: 'xlarge', 'large', 'medium', 'small'.
        spine_only: Whether to include only spine keypoints.
        use_smoothing: Whether to apply smoothing to keypoints over time.
        vis_path: Optional path to save the output video.

    Returns:
        A list of NumPy arrays with keypoints and scores for each frame.
        Empty arrays are included for frames with no detections.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)

    pose_tracker = PoseTracker(
        SpinePoseEstimator,
        mode=mode,
        smoothing=use_smoothing,
        smoothing_freq=fps,
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
        ret, img = cap.read()
        if not ret:
            break

        keypoints, scores = pose_tracker(img)

        if spine_only and len(scores) > 0:
            spine_ids = [36, 35, 18, 30, 29, 28, 27, 26, 19]
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
        frame_results = np.concatenate([keypoints, scores[..., np.newaxis]], axis=-1)
        if spine_only:
            frame_results = frame_results[:, spine_ids, :]
        all_results.append(frame_results)

    cap.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.close()

    return all_results


def _imshow(img, title="Image"):
    """Display an image with a maximum dimension of 1024 pixels."""
    h, w = img.shape[:2]
    scale = 1024 / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    img = cv2.resize(img, (new_w, new_h))
    cv2.imshow(title, img)


def _write_frame(keypoints_array, save_path):
    """
    Write a single frame's keypoints in OpenPose-compatible JSON format.
    Expected input shape: [num_people, num_keypoints, 3] (x, y, score)
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
    """Check if the file exists."""
    return os.path.isfile(filepath)


def _is_valid(filepath, formats):
    """Check if the file has a valid format."""
    _, ext = os.path.splitext(filepath)
    return ext.lower() in formats


def _is_image(filename):
    """Check if the file is an image."""
    img_exts = [".jpg", ".jpeg", ".png", ".bmp"]
    return _exists(filename) and _is_valid(filename, img_exts)


def _is_video(filename):
    """Check if the file is a video."""
    video_exts = [".mp4", ".avi", ".mov", ".mkv"]
    return _exists(filename) and _is_valid(filename, video_exts)


def main():
    parser = argparse.ArgumentParser(description="SpinePose Inference")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--version",
        "-V",
        action="store_true",
        help="Print the version and exit."
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
        "--nosmooth",
        action="store_false",
        help="Disable keypoint smoothing for video inference (default: enabled)",
    )
    parser.add_argument(
        "--spine-only",
        action="store_true",
        help="Only use 9 spine keypoints (default: use all 37 keypoints)",
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
            spine_only=args.spine_only,
            vis_path=args.vis_path,
        )
    elif _is_video(args.input_path):
        image_mode = False
        results = infer_video(
            args.input_path,
            args.mode,
            spine_only=args.spine_only,
            use_smoothing=args.nosmooth,
            vis_path=args.vis_path,
        )
    else:
        raise ValueError(
            "Input path must be a valid image or video file."
        )

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
