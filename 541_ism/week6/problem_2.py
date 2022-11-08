from scipy.optimize import fsolve
import numpy as np

def func(T):
    return T * (0.86 + 0.54 * (T / 1e4)**(0.37)) - 3/2 * 32000

print(fsolve(func, 32000))