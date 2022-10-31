import astropy.units as u
import astropy.constants as const
import numpy as np


omega_B = 0.0455
rho_crit = 9e-27 * u.kg * u.m**(-3)
rho_B = omega_B * rho_crit

print(f"{rho_B.to(u.Msun * u.Mpc**(-3)):1.1e}")

mean_lum = 2e8 * u.Lsun * u.Mpc**(-3) * 3.4 * (u.Msun / u.Lsun)

print((mean_lum / rho_B).decompose() * 100)
