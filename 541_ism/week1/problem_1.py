import astropy.units as u
import astropy.constants as const
import numpy as np

Mdisc = 4e9 * u.Msun
Rdisc = 15 * u.kpc
H = 200 * u.pc

density = (Mdisc / (np.pi * Rdisc**2 * H)).to(u.g / u.cm**3)
print(density)

n_H = (density / (1.4 * const.m_p)).to(u.cm**(-3))
print(n_H)