
import numpy as np


def np2ascii(x):
    for row in x:
        print(''.join(['#' if c else ' ' for c in row.tolist()]))
