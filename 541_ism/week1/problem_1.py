import astropy.units as u
import astropy.constants as const
import numpy as np

Mdisc = 4e9 * u.Msun
Rdisc = 15 * u.kpc
H = 200 * u.pc
Vdisc = (np.pi * Rdisc**2 * H).to(u.cm**3)

disc_density = (Mdisc / Vdisc).to(u.g / u.cm**3)
print(f"Disc density {disc_density:1.1e}")

n_H = (disc_density / (1.4 * const.m_p)).to(u.cm**(-3))
print(f"n_H {n_H:1.1e}")
print()

v_dg = 4/3 * np.pi * (1000 * u.Angstrom)**3
rho_dg = 2 * u.g / u.cm**3
m_dg = (v_dg * rho_dg).to(u.g)
print(f"Mass of a dust grain: {m_dg:1.1e}")

m_dust = 0.7 / 100 * Mdisc
N_dg = m_dust / m_dg
n_dg = (N_dg / Vdisc).to(u.cm**(-3))
print(f"Number density of dust grains {n_dg: 1.1e}")
print()


mc_vol = 4/3 * np.pi * (15 * u.pc)**3
h2_mass = 2 * const.m_p
mc_density = h2_mass * 100 * u.cm**(-3)
mc_mass_single = (mc_vol * mc_density).to(u.Msun)
print(f"Mass of a MC: {mc_mass_single:1.1e}")

mc_count = 0.3 * Mdisc / (mc_mass_single)
print(f"Number of MCs: {mc_count:1.1e}")
print()

# print(f"Fraction of volume occupied {(mc_vol / Vdisc).decompose():1.1e}")

v_los = (np.pi * (15 * u.pc)**2 * (8.5 * u.kpc)).to(u.pc**3)
print(f"Volume along line of sight: {v_los:1.1e}")

print((v_los / Vdisc * mc_count).decompose())
