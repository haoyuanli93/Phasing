import numpy as np
from numba import jit
import scipy
import scipy.ndimage
import math


@jit
def ellipsoid(x, y, z, a, b, c):
    """
    Generate an ellipsoid in the space。

    :param x: half length of the ellipsoid along dimension 0
    :param y: half length of the ellipsoid along dimension 1
    :param z: half length of the ellipsoid along dimension 2
    :param a: Length of the space along dimension 0
    :param b: Length of the space along dimension 1
    :param c: Length of the space along dimension 2
    :return:
    """
    # x,y,z are the dimensions of the sphere
    # a,b,c are the dimensnions of the space.
    space = np.zeros((a, b, c))
    center = [a / 2, b / 2, c / 2]

    fx = float(x)
    fy = float(y)
    fz = float(z)

    x = int(math.floor(x))
    y = int(math.floor(y))
    z = int(math.floor(z))
    for l in range(center[0] - x - 1, center[0] + x + 1):
        for m in range(center[1] - y - 1, center[1] + y + 1):
            for n in range(center[2] - z - 1, center[2] + z + 1):
                if ((l - a / 2) ** 2 / fx ** 2 + (m - b / 2) ** 2 / fy ** 2 + (
                        n - c / 2) ** 2 / fz ** 2 <= 1):
                    space[l, m, n] = 1
    return space


@jit
def torus(r1, r2, a=128, b=128, c=128):
    """
    Generate a torus in the space.

    :param r1:
    :param r2:
    :param a:
    :param b:
    :param c:
    :return:
    """
    space = np.zeros((a, b, c))

    for l in range(a / 2 - r1 - r2 - 1, a / 2 + r1 + r2 + 1):
        for m in range(b / 2 - r1 - r2 - 1, b / 2 + r1 + r2 + 1):
            for n in range(c / 2 - r2 - 1, c / 2 + r2 + 1):
                x = l - a / 2
                y = m - b / 2
                z = n - c / 2
                condition = ((r1 - np.sqrt(x ** 2 + y ** 2)) ** 2 + z ** 2 <= r2 ** 2)
                if condition:
                    space[l, m, n] = 1
    return space


def rotate(data, a, b, c):
    space = data[:]
    # first rotation
    space = scipy.ndimage.rotate(space, a, axes=(1, 0),
                                 reshape=False, output=None, order=3, mode='constant',
                                 cval=0.0, prefilter=True)
    space = scipy.ndimage.rotate(space, b, axes=(2, 1),
                                 reshape=False, output=None, order=3, mode='constant',
                                 cval=0.0, prefilter=True)
    space = scipy.ndimage.rotate(space, c, axes=(1, 0),
                                 reshape=False, output=None, order=3, mode='constant',
                                 cval=0.0, prefilter=True)
    return space


def shift(data, a, b, c):
    space = data[:]
    space = scipy.ndimage.shift(space, [a, b, c], output=None, order=3,
                                mode='constant', cval=0.0, prefilter=True)
    return space
