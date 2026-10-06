from pathlib import Path

import numpy as np
import pytest
import scipy.ndimage

DATA_DIR = Path(__file__).parent / 'data'


def blurred_stripes(sigma_left, sigma_right, height=40, width=192, stripe=24):
    """RGB image of vertical stripes; the left half is blurred by ``sigma_left``,
    the right half by ``sigma_right``. Stripes are wide enough that neighbouring
    edges do not interfere with the derivative-of-Gaussian kernels."""
    x = np.arange(width)
    stripes = np.tile((x // stripe) % 2, (height, 1)).astype(np.float64)
    left = scipy.ndimage.gaussian_filter(stripes, sigma_left)
    right = scipy.ndimage.gaussian_filter(stripes, sigma_right)
    gray = np.where(x < width // 2, left, right)
    return np.repeat((0.2 + 0.6 * gray)[..., None] * 255, 3, axis=-1)


@pytest.fixture
def data_dir():
    return DATA_DIR
