import numpy as np
import astropy.units as u

k_ra = 1.9e-16 * u.cm**3 / u.s
k_ad = 1.3e-9 * u.cm**3 / u.s
zeta_pd = 2.4e-7 * u.s**(-1)

n_H = 30 * u.cm**(-3)
n_e = 0.02 * u.cm**(-3)

ratio = (k_ra * n_e) / (k_ad * n_H + zeta_pd)

print(f"n_H-/ n_H: {ratio:1.2e}")

fraction_of_ad = k_ad * n_H / (k_ad * n_H + zeta_pd)
print(f"Fraction of associated detachments {fraction_of_ad:1.2f}")

R_hminus = k_ad * ratio
print(R_hminus)

R_ratio = ((3e-17 * u.cm**3 / u.cm**(-1)) / R_hminus).value
print(f"Required temperature {R_ratio**(1/0.67) * 100 * u.K:1.2e}")