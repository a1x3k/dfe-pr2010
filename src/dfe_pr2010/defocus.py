"""Defocus map estimation from a single image.

Implementation of S. Zhuo, T. Sim, "Defocus map estimation from a single
image", Pattern Recognition, 2011.

The image is re-blurred with two Gaussian kernels; the ratio of the gradient
magnitudes at edge locations gives a sparse blur estimate, which is then
propagated to the whole image with the matting Laplacian of Levin et al.,
"A closed-form solution to natural image matting".

Pixels are flattened in row-major order throughout.
"""

import numpy as np
import scipy.ndimage
import scipy.sparse
import scipy.sparse.linalg
from skimage import feature

# ITU-R BT.601 luma weights, the same ones OpenCV uses for RGB -> gray
LUMA_WEIGHTS = np.array([0.299, 0.587, 0.114])


def to_gray(img):
    """Convert an RGB image in [0, 255] to grayscale in [0, 1].

    The luma is rounded to 8-bit before scaling, matching OpenCV's
    ``cvtColor``. Canny edges are sensitive to this rounding.
    """
    return np.round(img[..., :3] @ LUMA_WEIGHTS) / 255.0


def gaussian_gradient_kernels(sigma, half_window=11):
    """Return the x and y derivative-of-Gaussian kernels.

    :param sigma: Standard deviation of the Gaussian
    :param half_window: Kernel radius; the kernel is (2 * half_window + 1) square
    :return: (kx, ky), each of shape (2 * half_window + 1, 2 * half_window + 1)
    """
    y, x = np.mgrid[-half_window:half_window + 1, -half_window:half_window + 1]
    g = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2)) / (2 * np.pi * sigma ** 4)
    return -x * g, -y * g


def gradient_magnitude(gimg, sigma, half_window=11):
    """Gradient magnitude of ``gimg`` after blurring with a Gaussian of ``sigma``."""
    kx, ky = gaussian_gradient_kernels(sigma, half_window)
    gx = scipy.ndimage.convolve(gimg, kx, mode='nearest')
    gy = scipy.ndimage.convolve(gimg, ky, mode='nearest')
    return np.hypot(gx, gy)


def estimate_sparse_blur(gimg, edge_map, std1, std2, half_window=11, max_blur=5.0):
    """Estimate the blur amount at edge locations.

    :param gimg: Grayscale image in [0, 1]
    :param edge_map: Boolean edge map of the image
    :param std1: Standard deviation of the first re-blurring
    :param std2: Standard deviation of the second re-blurring
    :param half_window: Radius of the derivative-of-Gaussian kernels
    :param max_blur: Upper bound on the estimated blur
    :return: Blur sigma at edge pixels, 0 elsewhere
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = gradient_magnitude(gimg, std1, half_window) / gradient_magnitude(gimg, std2, half_window)
        ratio = ratio * (edge_map > 0)
        variance = (ratio ** 2 * std1 ** 2 - std2 ** 2) / (1 - ratio ** 2)

    sigma = np.sqrt(np.clip(variance, 0, None))
    sigma = np.nan_to_num(sigma, nan=0.0, posinf=max_blur)
    return np.minimum(sigma, max_blur)


def matting_laplacian(img, r=1, eps=1e-7):
    """Build the matting Laplacian of Levin et al.

    :param img: An RGB image in [0, 1], shape (h, w, 3)
    :param r: Window radius
    :param eps: Regularisation added to each window's colour covariance
    :return: Sparse (h * w, h * w) Laplacian matrix
    """
    h, w, c = img.shape
    win_size = (2 * r + 1) ** 2
    pixel_idx = np.arange(h * w).reshape(h, w)

    n_windows = (h - 2 * r) * (w - 2 * r)
    rows = np.empty(n_windows * win_size ** 2, dtype=np.int64)
    cols = np.empty_like(rows)
    vals = np.empty(rows.shape, dtype=np.float64)

    pos = 0
    for j in range(r, h - r):
        for i in range(r, w - r):
            win_idx = pixel_idx[j - r:j + r + 1, i - r:i + r + 1].ravel()
            win = img[j - r:j + r + 1, i - r:i + r + 1].reshape(win_size, c)

            mean = win.mean(axis=0)
            cov = win.T @ win / win_size - np.outer(mean, mean) + eps * np.eye(c)
            centered = win - mean
            affinity = (1 + centered @ np.linalg.inv(cov) @ centered.T) / win_size

            n = win_size ** 2
            rows[pos:pos + n] = np.repeat(win_idx, win_size)
            cols[pos:pos + n] = np.tile(win_idx, win_size)
            vals[pos:pos + n] = affinity.ravel()
            pos += n

    # duplicate (row, col) entries from overlapping windows are summed
    affinity_matrix = scipy.sparse.coo_matrix((vals, (rows, cols)), shape=(h * w, h * w)).tocsr()
    degree = scipy.sparse.diags(np.asarray(affinity_matrix.sum(axis=1)).ravel())
    return (degree - affinity_matrix).tocsr()


def make_system(laplacian, sparse_map, constraint_factor=0.001):
    """Build the linear system that propagates the sparse blur estimates.

    Solves (L + lambda * D) x = lambda * D * d, where d holds the sparse
    estimates and D = diag(d), so pixels with a confident (larger) estimate
    are constrained more strongly.

    :param laplacian: Matting Laplacian
    :param sparse_map: Sparse blur estimates, shape (h, w)
    :param constraint_factor: Weight of the data term (lambda)
    :return: (A, b)
    """
    d = sparse_map.ravel()
    D = scipy.sparse.diags(d)
    A = laplacian + constraint_factor * D
    b = constraint_factor * (D @ d)
    return A.tocsc(), b


def estimate_defocus_map(img, sigma_c=1.0, std1=1.0, std2=1.5):
    """Estimate the full defocus blur map of an image.

    :param img: An RGB image in [0, 255], shape (h, w, 3)
    :param sigma_c: Sigma of the Canny edge detector
    :param std1: Standard deviation of the first re-blurring
    :param std2: Standard deviation of the second re-blurring
    :return: Defocus blur map, shape (h, w)
    """
    img = np.asarray(img, dtype=np.float64)
    h, w = img.shape[:2]

    gimg = to_gray(img)
    edge_map = feature.canny(gimg, sigma_c)
    sparse_bmap = estimate_sparse_blur(gimg, edge_map, std1, std2)

    laplacian = matting_laplacian(img / 255.0)
    A, b = make_system(laplacian, sparse_bmap)
    return scipy.sparse.linalg.spsolve(A, b).reshape(h, w)
