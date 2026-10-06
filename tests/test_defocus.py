import numpy as np
import pytest
import scipy.ndimage
from skimage import io

from conftest import blurred_stripes
from dfe_pr2010.defocus import (
    estimate_defocus_map,
    estimate_sparse_blur,
    gaussian_gradient_kernels,
    make_system,
    matting_laplacian,
    to_gray,
)


class TestToGray:
    def test_primaries_use_bt601_weights(self):
        img = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]]], dtype=np.float64)
        np.testing.assert_allclose(to_gray(img), [[76 / 255, 150 / 255, 29 / 255]])

    def test_white_and_black(self):
        img = np.array([[[255, 255, 255], [0, 0, 0]]], dtype=np.float64)
        np.testing.assert_allclose(to_gray(img), [[1.0, 0.0]])

    def test_ignores_alpha(self):
        rgb = np.full((2, 2, 3), 100.0)
        rgba = np.concatenate([rgb, np.full((2, 2, 1), 7.0)], axis=-1)
        np.testing.assert_array_equal(to_gray(rgba), to_gray(rgb))


class TestGaussianGradientKernels:
    def test_shape(self):
        kx, ky = gaussian_gradient_kernels(1.0, half_window=5)
        assert kx.shape == ky.shape == (11, 11)

    def test_y_kernel_is_transposed_x_kernel(self):
        kx, ky = gaussian_gradient_kernels(1.5)
        np.testing.assert_allclose(ky, kx.T)

    def test_antisymmetric_and_zero_sum(self):
        kx, _ = gaussian_gradient_kernels(1.0)
        np.testing.assert_allclose(kx, -kx[:, ::-1])
        assert kx[:, kx.shape[1] // 2].max() == 0
        assert abs(kx.sum()) < 1e-12

    def test_unit_ramp_has_unit_gradient(self):
        # convolving x with the derivative of a normalised Gaussian gives d/dx = 1
        kx, _ = gaussian_gradient_kernels(1.0)
        ramp = np.tile(np.arange(64, dtype=np.float64), (64, 1))
        gx = scipy.ndimage.convolve(ramp, kx, mode='nearest')
        np.testing.assert_allclose(gx[32, 32], 1.0, rtol=1e-6)


class TestEstimateSparseBlur:
    @staticmethod
    def blurred_step(sigma):
        step = np.zeros((40, 40))
        step[:, 20:] = 1
        edges = np.zeros_like(step, dtype=bool)
        edges[:, 19] = True
        return scipy.ndimage.gaussian_filter(step, sigma), edges

    @pytest.mark.parametrize('sigma', [1.0, 2.0, 3.0])
    def test_recovers_blur_of_step_edge(self, sigma):
        gimg, edges = self.blurred_step(sigma)
        bmap = estimate_sparse_blur(gimg, edges, std1=1, std2=1.5)
        assert bmap[20, 19] == pytest.approx(sigma, abs=0.1)

    def test_zero_away_from_edges(self):
        gimg, edges = self.blurred_step(2.0)
        bmap = estimate_sparse_blur(gimg, edges, std1=1, std2=1.5)
        assert np.all(bmap[~edges] == 0)

    def test_flat_image_gives_zero_without_warnings(self):
        gimg = np.full((20, 20), 0.5)
        with np.errstate(all='raise'):
            bmap = estimate_sparse_blur(gimg, np.ones_like(gimg, dtype=bool), 1, 1.5)
        np.testing.assert_array_equal(bmap, 0)

    def test_clipped_to_max_blur(self):
        gimg, edges = self.blurred_step(8.0)
        bmap = estimate_sparse_blur(gimg, edges, std1=1, std2=1.5, max_blur=2.5)
        assert bmap.max() == 2.5


class TestMattingLaplacian:
    @pytest.fixture
    def laplacian(self):
        img = np.random.default_rng(0).random((8, 10, 3))
        return matting_laplacian(img)

    def test_shape(self, laplacian):
        assert laplacian.shape == (80, 80)

    def test_symmetric(self, laplacian):
        assert abs(laplacian - laplacian.T).max() < 1e-8

    def test_rows_sum_to_zero(self, laplacian):
        np.testing.assert_allclose(laplacian.sum(axis=1), 0, atol=1e-8)

    def test_positive_semidefinite(self, laplacian):
        assert np.linalg.eigvalsh(laplacian.toarray()).min() > -1e-6

    def test_window_neighbours_only(self):
        # pixels further apart than one window (2r + 1) have no affinity
        laplacian = matting_laplacian(np.random.default_rng(1).random((6, 6, 3))).toarray()
        idx = lambda j, i: j * 6 + i
        assert laplacian[idx(0, 0), idx(0, 2)] != 0
        assert laplacian[idx(0, 0), idx(0, 3)] == 0
        assert laplacian[idx(0, 0), idx(3, 0)] == 0


class TestMakeSystem:
    def test_values(self):
        laplacian = matting_laplacian(np.random.default_rng(0).random((4, 5, 3)))
        sparse_map = np.random.default_rng(1).random((4, 5))
        A, b = make_system(laplacian, sparse_map, constraint_factor=0.1)

        d = sparse_map.ravel()
        np.testing.assert_allclose(A.toarray(), laplacian.toarray() + 0.1 * np.diag(d))
        np.testing.assert_allclose(b, 0.1 * d ** 2)


class TestEstimateDefocusMap:
    def test_separates_sharp_and_blurred_regions(self):
        img = blurred_stripes(sigma_left=1.0, sigma_right=3.0)
        bmap = estimate_defocus_map(img)

        assert bmap.shape == img.shape[:2]
        assert np.median(bmap[:, 5:90]) == pytest.approx(1.0, abs=0.2)
        assert np.median(bmap[:, 102:187]) == pytest.approx(3.0, abs=0.2)

    def test_accepts_uint8(self):
        img = blurred_stripes(1.0, 3.0)
        np.testing.assert_allclose(
            estimate_defocus_map(img.round().astype(np.uint8)),
            estimate_defocus_map(img.round()),
        )

    def test_matches_reference_output(self, data_dir):
        # reference produced by the implementation before the refactor
        img = io.imread(data_dir / 'input0_crop.png')
        expected = np.load(data_dir / 'input0_crop_bmap.npy')
        np.testing.assert_allclose(estimate_defocus_map(img), expected, rtol=1e-6, atol=1e-8)
