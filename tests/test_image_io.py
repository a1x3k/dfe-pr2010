import numpy as np
from skimage import io

from dfe_pr2010.image_io import load_rgb, save_defocus_map


def test_load_rgb_keeps_rgb(tmp_path):
    rgb = np.random.default_rng(0).integers(0, 256, (5, 7, 3), dtype=np.uint8)
    io.imsave(tmp_path / 'rgb.png', rgb, check_contrast=False)

    img = load_rgb(tmp_path / 'rgb.png')
    assert img.dtype == np.float64
    np.testing.assert_array_equal(img, rgb)


def test_load_rgb_drops_alpha(tmp_path):
    rgba = np.random.default_rng(0).integers(0, 256, (5, 7, 4), dtype=np.uint8)
    io.imsave(tmp_path / 'rgba.png', rgba, check_contrast=False)

    np.testing.assert_array_equal(load_rgb(tmp_path / 'rgba.png'), rgba[..., :3])


def test_load_rgb_expands_grayscale(tmp_path):
    gray = np.random.default_rng(0).integers(0, 256, (5, 7), dtype=np.uint8)
    io.imsave(tmp_path / 'gray.png', gray, check_contrast=False)

    img = load_rgb(tmp_path / 'gray.png')
    assert img.shape == (5, 7, 3)
    for c in range(3):
        np.testing.assert_array_equal(img[..., c], gray)


def test_save_defocus_map(tmp_path):
    bmap = np.array([[0.0, 1.0], [2.0, 4.0]])
    npy_path, png_path = save_defocus_map(bmap, tmp_path / 'out')

    assert npy_path == f'{tmp_path / "out"}_bmap.npy'
    np.testing.assert_array_equal(np.load(npy_path), bmap)
    np.testing.assert_array_equal(io.imread(png_path), [[0, 63], [127, 255]])
