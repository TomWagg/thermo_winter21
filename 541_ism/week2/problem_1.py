import astropy.units as u
import astropy.constants as const
import numpy as np
from scipy.integrate import quad


def maxwellian(v, m, T):
    exp_term = np.exp(- m * v**2 / (2 * const.k_B.si.value * T))
    return (4 * np.pi * (m / (2 * np.pi * const.k_B.si.value * T))**(3/2) * v**2 * exp_term)

E_ionise = 13.6 * u.eV
v_ionise = np.sqrt(2 * E_ionise / const.m_e).to(u.km / u.s)
print(v_ionise)

for T in [1e4, 1e6]:
    prob = 1 - quad(maxwellian, 0, v_ionise.si.value, args=(const.m_e.si.value, T))[0]
    print(f"At T={T * u.K:1.1e}, fraction {prob:1.1e} of electrons are able to ionise")
