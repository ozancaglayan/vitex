#!/usr/bin/env python

import sys
import argparse
from pathlib import Path

from tqdm import tqdm

import numpy as np

from src.backends import get_renderer

FONT_PATH = "/usr/share/fonts/truetype"


def main():
    parser = argparse.ArgumentParser(
        description='Prepare a dataset with visual-text source representations.',
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-b', '--backend', default='freetype',
                        choices=['freetype', 'pygame', 'pillow'],
                        help='The backend to use for text rendering.')
    parser.add_argument('-mh', '--max-height', default=12, type=int,
                        help='Maximum height of rendered sentences.')
    parser.add_argument('-fs', '--font-size', default=12, type=int,
                        help='Font size.')
    parser.add_argument('-ff', '--font-face', default='ubuntu/Ubuntu-R.ttf', type=str,
                        help='The truetype font face.')
    parser.add_argument('-lp', '--language-pair', required=True, type=str,
                        help='`<src>-<trg>` language pair string.')
    parser.add_argument('--train', type=str, required=True,
                        help='The prefix for training corpus.')
    parser.add_argument('--val', type=str, required=True,
                        help='The prefix for validation corpus.')
    parser.add_argument('--test', type=str, required=True,
                        help='The prefix for test corpus.')

    args = parser.parse_args()
    src_lang, trg_lang = args.language_pair.split('-')

    font = Path(f'{FONT_PATH}/{args.font_face}')
    if not font.exists():
        print(f'Font {font!r} not found.')
        sys.exit(1)

    print(f'Loading font: {str(font)}')

    renderer = get_renderer(args.backend)(
        str(font), args.font_size, args.max_height)

    names = ['train', 'val', 'test']

    for name, prefix in zip(names, (args.train, args.val, args.test)):
        fname = f'{name}-fs{args.font_size}-mh{args.max_height}.npz'

        bitmaps = []
        lens = []
        slices = []

        print(f'Processing {name} split')
        src_file = prefix + f'.{src_lang}'
        with open(src_file) as f:
            for idx, line in enumerate(tqdm(f)):
                line = line.strip()
                width, height, pixels = renderer(line)
                if height != args.max_height:
                    raise RuntimeError(f'height differed for line: {idx + 1}')

                # process bitmap
                bitmaps.append(pixels)
                lens.append(height * width)

        s = 0
        data = np.empty(sum(lens), dtype='bool')

        print(f'Filling the tensor of size: {data.size}')
        for bmp, leng in zip(bitmaps, lens):
            slices.append((s, s + leng))
            data[s: s + leng] = bmp
            s += leng

        print(f'Saving into {fname}')
        np.savez_compressed(fname, data=data, lens=slices)


if __name__ == '__main__':
    sys.exit(main())
