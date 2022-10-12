import astropy.units as u
import astropy.constants as const
import numpy as np

n_H = 0.22 * u.cm**(-3)
rho = 1.4 * const.m_p * n_H
rho_dust = 0.5 / 100 * rho
m_dust_particle = 4/3 * np.pi * (0.1 * u.micrometer)**3 * (2 * u.g * u.cm**(-3))
n_dust = rho_dust / m_dust_particle

print(f"Number density of dust {n_dust.to(u.cm**(-3)): 1.1e}")

v = 26 * u.km / u.s

collecting_area = ((1 / u.hr) / (n_dust * v)).to(u.cm**2)
print(f"Collecting area required: {collecting_area:1.1f}")
