"""Reading input images and writing defocus maps."""

import numpy as np
from skimage import io


def load_rgb(path):
    """Load an image as a float64 RGB array in [0, 255], shape (h, w, 3).

    Grayscale images are replicated to three channels and alpha is dropped.
    """
    img = io.imread(path)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    return img[..., :3].astype(np.float64)


def save_defocus_map(bmap, prefix):
    """Save a defocus map as ``<prefix>_bmap.npy`` and a normalised ``<prefix>_bmap.png``.

    :return: (npy_path, png_path)
    """
    npy_path = f'{prefix}_bmap.npy'
    png_path = f'{prefix}_bmap.png'
    np.save(npy_path, bmap)
    io.imsave(png_path, np.uint8(bmap / bmap.max() * 255), check_contrast=False)
    return npy_path, png_path
