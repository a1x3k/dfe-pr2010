import shutil
import subprocess
import sys

import numpy as np

from dfe_pr2010.cli import main, parse_args


def test_parse_args_defaults():
    args = parse_args(['-i', 'img.png'])
    assert args.image == 'img.png'
    assert args.output is None
    assert (args.sigma_c, args.std1, args.std2) == (1.0, 1.0, 1.5)


def test_main_writes_outputs_next_to_image(tmp_path, data_dir, capsys):
    image = tmp_path / 'crop.png'
    shutil.copy(data_dir / 'input0_crop.png', image)

    main(['-i', str(image)])

    bmap = np.load(f'{image}_bmap.npy')
    np.testing.assert_allclose(bmap, np.load(data_dir / 'input0_crop_bmap.npy'), rtol=1e-6, atol=1e-8)
    assert (tmp_path / 'crop.png_bmap.png').exists()
    assert 'Saved' in capsys.readouterr().out


def test_main_custom_output_prefix(tmp_path, data_dir):
    main(['-i', str(data_dir / 'input0_crop.png'), '-o', str(tmp_path / 'result')])

    assert (tmp_path / 'result_bmap.npy').exists()
    assert (tmp_path / 'result_bmap.png').exists()


def test_runs_as_module(tmp_path, data_dir):
    result = subprocess.run(
        [sys.executable, '-m', 'dfe_pr2010', '-i', str(data_dir / 'input0_crop.png'), '-o', str(tmp_path / 'm')],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / 'm_bmap.npy').exists()
