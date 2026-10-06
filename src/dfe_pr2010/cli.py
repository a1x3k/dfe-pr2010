"""Command-line interface."""

import argparse

from dfe_pr2010.defocus import estimate_defocus_map
from dfe_pr2010.image_io import load_rgb, save_defocus_map


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Defocus map estimation from a single image, '
                    'S. Zhuo, T. Sim - Pattern Recognition, 2011 - Elsevier')
    parser.add_argument('-i', '--image', required=True, help='Defocused image')
    parser.add_argument('-o', '--output', help='Output prefix (default: the image path)')
    parser.add_argument('--sigma-c', type=float, default=1.0, help='Sigma of the Canny edge detector')
    parser.add_argument('--std1', type=float, default=1.0, help='Standard deviation of the first re-blurring')
    parser.add_argument('--std2', type=float, default=1.5, help='Standard deviation of the second re-blurring')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    img = load_rgb(args.image)
    bmap = estimate_defocus_map(img, sigma_c=args.sigma_c, std1=args.std1, std2=args.std2)
    for path in save_defocus_map(bmap, args.output or args.image):
        print(f'Saved {path}')


if __name__ == '__main__':
    main()
