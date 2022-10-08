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


mc_vol = 4/3 * np.pi * (15 * u.pc)**3
h2_mass = 2 * const.m_p
mc_density = h2_mass * 100 * u.cm**(-3)
mc_mass_single = (mc_vol * mc_density).to(u.Msun)
print(f"Mass of a MC: {mc_mass_single:1.1e}")

mc_count = 0.3 * Mdisc / (mc_mass_single)
print(f"Number of MCs: {mc_count:1.1e}")
print()

# print(f"Fraction of volume occupied {(mc_vol / Vdisc).decompose():1.1e}")

v_los = (np.pi * (15 * u.pc)**2 * (8.5 * u.kpc)).to(u.cm**3)
print("Volume along line of sight")

print((v_los / Vdisc * mc_count).decompose())