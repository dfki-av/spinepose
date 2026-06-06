import numpy as np
from OneEuroFilter import OneEuroFilter


class KeypointSmoothing:
    """Smooths keypoint coordinates with One Euro filters."""

    def __init__(
        self,
        num_keypoints,
        freq: float,
        mincutoff: float = 1.0,
        beta: float = 0.0,
        dcutoff: float = 1.0,
    ):
        """Initializes the keypoint smoother.

        Args:
            num_keypoints: Number of keypoints to smooth.
            freq: Estimated signal frequency in hertz.
            mincutoff: Minimum cutoff frequency.
            beta: Speed coefficient for adaptive smoothing.
            dcutoff: Cutoff frequency for the derivative filter.
        """
        kwargs = dict(
            freq=freq,
            mincutoff=mincutoff,
            beta=beta,
            dcutoff=dcutoff,
        )
        self.filters = [
            {
                "filter_x": OneEuroFilter(**kwargs),
                "filter_y": OneEuroFilter(**kwargs),
            }
            for _ in range(num_keypoints)
        ]

    def filter_keypoint(self, point, filters):
        """Filters a single keypoint.

        Args:
            point: Keypoint coordinates.
            filters: Per-axis filters for the keypoint.

        Returns:
            np.ndarray: Smoothed keypoint coordinates.
        """
        filtered_x = filters["filter_x"](point[0])
        filtered_y = filters["filter_y"](point[1])
        return np.array([filtered_x, filtered_y])

    def __call__(self, keypoints):
        """Filters all keypoints in a pose.

        Args:
            keypoints: Keypoint coordinates for one pose.

        Returns:
            np.ndarray: Smoothed keypoint coordinates.
        """
        return np.array(
            [
                self.filter_keypoint(kpt, filters)
                for kpt, filters in zip(keypoints, self.filters)
            ]
        )
