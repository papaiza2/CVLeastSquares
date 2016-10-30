import numpy as np
import cv2


def three_point_center(data):
    """Picks three random points from a data set and returns the center"""

    if len(data) < 3:
        print('At least 3 points are needed for calculations.')
        return

    else:
        shuffled = np.random.permutation(data)
        a = shuffled[0]
        b = shuffled[1]
        c = shuffled[2]
        x = int(a[0]) + int(a[1]) * 1j
        y = int(b[0]) + int(b[1]) * 1j
        z = int(c[0]) + int(c[1]) * 1j

        w = z - x
        w /= y - x
        c = (x - y) * (w - abs(w) ** 2) / 2j / w.imag - x
        xcen = -int(c.real)
        ycen = -int(c.imag)
        radius = int(abs(c + x))

    return xcen, ycen, radius


def points_in_range(xcen, ycen, radius, data, W):
    """Returns the number of points within radius + or - W pixels of the center."""

    center = np.array([xcen,ycen])
    count = 0
    for point in data:
        if abs(np.linalg.norm(center-point)-radius)/W < 1:
            count += 1

    return count
