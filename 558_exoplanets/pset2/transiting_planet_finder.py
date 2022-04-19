import numpy as np

# need to think more about what happens once more than half of the smaller one is in the bigger one


def overlap_area_touching(a, r1, r2):
    print(a, r1, r2)
    s = 0.5 * (a + r1 + r2)
    theta1 = np.arccos((a**2 + r1**2 - r2**2) / (2 * r1 * r2))
    theta2 = np.arccos((a**2 + r2**2 - r1**2) / (2 * r1 * r2))

    print(theta1, theta2)

    area = theta1 * r1 + theta2 * r2 - 2 * np.sqrt(s * (s - r1) * (s - r2) * (s - a))

    return area

def overlap_area(a, r1, r2):
    overlaps = np.zeros_like(a)

    lower = np.maximum(r1, r2) - np.minimum(r1, r2)
    upper = r1 + r2

    overlaps[a <= lower] = np.pi * lower[a <= lower]**2
    overlaps[a >= upper] = 0

    touching = np.logical_and(a > lower, a < upper)
    overlaps[touching] = overlap_area_touching(a[touching], r1[touching], r2[touching])

    return overlaps