import numpy as np


def overlap_area_touching(a, r1, r2):
    """Find the overlap area for two circles where `r1 - r2 < a < r1 + r2`

    Parameters
    ----------
    a : `float/array`
        Separations
    r1 : `float`
        Radius of circle 1
    r2 : `float`
        Radius of circle 2

    Returns
    -------
    areas : `float/array`
        Areas of overlap at different separations
    """
    # find the angles using the cosine rule
    theta1 = np.arccos((a**2 + r1**2 - r2**2) / (2 * r1 * a))
    theta2 = np.arccos((a**2 + r2**2 - r1**2) / (2 * r2 * a))

    # use the angles to find the sector areas
    sec1 = 0.5 * theta1 * r1**2
    sec2 = 0.5 * theta2 * r2**2

    # use Heron's formula to get the area of the triangle
    s = 0.5 * (a + r1 + r2)
    triangle = np.sqrt(s * (s - r1) * (s - r2) * (s - a))

    return 2 * (sec1 + sec2 - triangle)


def overlap_area(a, r1, r2):
    """Find the overlap area for two circles of radii `r1` and `r2` at different separations `a`

    Parameters
    ----------
    a : `float/array`
        Separations
    r1 : `float`
        Radius of circle 1
    r2 : `float`
        Radius of circle 2

    Returns
    -------
    areas : `float/array`
        Areas of overlap at different separations
    """
    # set up area array and work out which radius is larger
    areas = np.zeros_like(a)
    r_min = min(r1, r2)
    r_max = max(r1, r2)

    # define lower and upper bounds
    lower = r_max - r_min
    upper = r_max + r_min

    # below the lower bound the area is just the area of the smaller circle
    areas[a < lower] = np.pi * r_min**2

    # above the upper bound the area there is no overlap
    areas[a >= upper] = 0

    # for the area in the range in-between we can use the formula we created
    touching = np.logical_and(a >= lower, a < upper)
    areas[touching] = overlap_area_touching(a[touching], r_max, r_min)

    return areas
