# defocusblurPR2011
Implementation of "Defocus map estimation from a single image", S. Zhuo, T. Sim - Pattern Recognition, 2011 - Elsevier


## Usage

```
uv sync
uv run dfe-pr2010 -i input0.png
```

This writes `input0.png_bmap.npy` (the raw defocus map) and `input0.png_bmap.png` (normalised for viewing). Use `-o PREFIX` to choose another output location, and `--sigma-c`, `--std1`, `--std2` to change the Canny and re-blurring parameters.

From Python:

```python
from dfe_pr2010 import estimate_defocus_map, load_rgb

bmap = estimate_defocus_map(load_rgb("input0.png"))
```

## Tests

```
uv run pytest
```
