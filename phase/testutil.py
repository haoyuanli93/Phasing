import numpy as np
from numba import jit
import scipy
import scipy.ndimage
import math
from scipy.spatial import ConvexHull


@jit
def get_ellipsoid(x, y, z, a, b, c):
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
def get_torus(r1, r2, a=128, b=128, c=128):
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
    """
    Rotate the volume along the center of the space with the specified eular angle

    :param data:
    :param a:
    :param b:
    :param c:
    :return:
    """
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


@jit
def produce_volume(condition, a=128, b=128, c=128):
    """
    Generate a volume with the given conditions.

    :param condition:
    :param a:
    :param b:
    :param c:
    :return:
    """
    space = np.zeros((a, b, c))
    center = [a / 2, b / 2, c / 2]

    initial_sign = np.less_equal(condition.dot([center[0], center[1], center[2], 1]),
                                 np.zeros(condition.shape[0]))
    initial_sign = initial_sign[np.newaxis, :]

    for l in range(a):
        for m in range(b):
            for n in range(c):
                sign = np.less_equal(condition.dot([l, m, n, 1]), np.zeros(condition.shape[0]))
                if np.allclose(sign, initial_sign):
                    space[l, m, n] = 1

    return space


def get_equations_for_icosahedron(r, a, b, c):
    # r define the radius of the shape
    # a,b,c are the dimension of the space.
    phi = (np.sqrt(5) + 1) / 2

    vertexes = []
    for y in [-r, r]:
        for z in [-phi * r, phi * r]:
            vertex = [a / 2, b / 2 + int(y), c / 2 + int(z)]
            vertexes += [[vertex[i - j] for i in range(3)] for j in range(3)]

    convex_hull = ConvexHull(vertexes)
    equations = np.ascontiguousarray(convex_hull.equations)

    return equations


def get_equations_for_dodecahedron(r, a, b, c):
    # r define the radius of the shape
    # a,b,c are the dimension of the space.
    phi = int((np.sqrt(5) + 1) / 2 * r)
    phi_invers = int((2 / (np.sqrt(5) + 1) * r))

    vertexes = []

    for u in [-r, r]:
        for v in [-r, r]:
            for w in [-r, r]:
                vertex = [a / 2 + u, b / 2 + v, c / 2 + w]
                vertexes += [vertex]

    for u in [-phi_invers, phi_invers]:
        for v in [-phi, phi]:
            vertex = [a / 2, b / 2 + u, c / 2 + v]
            vertexes += [vertex]

    for u in [-phi_invers, phi_invers]:
        for v in [-phi, phi]:
            vertex = [a / 2 + u, b / 2 + v, c / 2]
            vertexes += [vertex]

    for u in [-phi_invers, phi_invers]:
        for v in [-phi, phi]:
            vertex = [a / 2 + v, b / 2, c / 2 + u]
            vertexes += [vertex]

    convex_hull = ConvexHull(vertexes)
    equations = np.ascontiguousarray(convex_hull.equations)

    return equations


def get_icosahedron(r, a, b, c):
    """
    Generate a numpy array with an icosahedron in the center
    :param r:
    :param a:
    :param b:
    :param c:
    :return:
    """
    equaiton = get_equations_for_icosahedron(r=r, a=a, b=b, c=c)
    return produce_volume(condition=equaiton, a=a, b=b, c=c)


def get_dodecahedron(r, a, b, c):
    """
    Generate a numpy array with a dodecahedron in the center.
    :param r:
    :param a:
    :param b:
    :param c:
    :return:
    """
    equaiton = get_equations_for_dodecahedron(r=r, a=a, b=b, c=c)
    return produce_volume(condition=equaiton, a=a, b=b, c=c)
