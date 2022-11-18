import numpy as np
import astropy.units as u
import astropy.constants as const

w1 = 10 * u.km / u.s
w2 = 5 * u.km / u.s
M1 = 1 * const.m_p
M2 = 23 * const.m_p

T = (((w1**2 - w2**2) / (8 * np.log(2))) * (const.k_B / M1 - const.k_B / M2)**(-1)).decompose()

sigma_turb = np.sqrt(w1**2 / (8 * np.log(2)) - const.k_B * T / M1)

print(f"Temperature: {T.to(u.K):1.0f}")
print(f"Turblence: {sigma_turb.to(u.km / u.s):1.2f}")
