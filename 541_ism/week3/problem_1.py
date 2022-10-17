import astropy.units as u
import numpy as np


def stromgren_radius(Q, n_e, n_ion, alpha):
    return (3 / (4 * np.pi) * Q / (n_e * n_ion * alpha))**(1/3)


# Q_H and ratio from Draine Table 15.1
Q_H = 10**(48.75) * u.s**(-1)
Q_He = 0.135 * Q_H

# test for an O3V star
# Q_H = 10**(49.64) * u.s**(-1)
# Q_He = 0.251 * Q_H

# values given in the question
n_H = 5000 * u.cm**(-3)
alpha_H = 4e-13 * u.cm**3 * u.s**(-1)
n_He = 0.1 * n_H
alpha_He = 2 * alpha_H

# calculate radii
r_H = stromgren_radius(Q_H, n_H, n_H, alpha_H).to(u.pc)
r_He = stromgren_radius(Q_He, n_H + n_He, n_He, alpha_He).to(u.pc)

print(f"Radius for H is {r_H:1.2f}")
print(f"Radius for He is {r_He:1.2f}")

print(f"Timescale for H {(1 / (alpha_H * n_H)).to(u.yr):1.1f}")
print(f"Timescale for He {(1 / (alpha_He * n_He)).to(u.yr):1.1f}")
