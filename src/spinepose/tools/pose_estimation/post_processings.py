from typing import Tuple

import numpy as np


def get_simcc_maximum(
    simcc_x: np.ndarray, simcc_y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Extracts peak locations and scores from SimCC outputs.

    Args:
        simcc_x: X-axis SimCC tensor in ``(N, K, Wx)`` format.
        simcc_y: Y-axis SimCC tensor in ``(N, K, Wy)`` format.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Peak coordinates and confidence values.
    """
    N, K, Wx = simcc_x.shape
    simcc_x = simcc_x.reshape(N * K, -1)
    simcc_y = simcc_y.reshape(N * K, -1)

    # get maximum value locations
    x_locs = np.argmax(simcc_x, axis=1)
    y_locs = np.argmax(simcc_y, axis=1)
    locs = np.stack((x_locs, y_locs), axis=-1).astype(np.float32)
    max_val_x = np.amax(simcc_x, axis=1)
    max_val_y = np.amax(simcc_y, axis=1)

    # get maximum value across x and y axis
    # mask = max_val_x > max_val_y
    # max_val_x[mask] = max_val_y[mask]
    vals = 0.5 * (max_val_x + max_val_y)
    locs[vals <= 0.0] = -1

    # reshape
    locs = locs.reshape(N, K, 2)
    vals = vals.reshape(N, K)

    return locs, vals
