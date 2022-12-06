import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants as const

ratio = 24 / 14.5


f1393 = 0.5280
f1402 = 0.2620

l1393 = 1393.755 * u.Angstrom
l1402 = 1402.770 * u.Angstrom

W1393 = 50 * u.mAngstrom
W1402 = W1393 / ratio

factor = np.log(f1393 * l1393 / (f1402 * l1402))
tau_0 = np.e**(factor / (ratio**2 - 1)) * np.log(2)


def column_density(ew, f, l):
    return 1.13e17 * ew.to(u.mAngstrom).value /\
        (f * l.to(u.Angstrom).value**2) * u.cm**(-2)


print(f"tau_0 = {tau_0:1.2f}")

N_1393 = column_density(W1393, f1393, l1393)
N_1402 = column_density(W1402, f1402, l1402)
print(f"N_1393 = {N_1393:1.2e}")
print(f"N_1402 = {N_1402:1.2e}")

abundance = 12 + np.log10(N_1402.value * 10 / 1e19)
print(f"abundance = {abundance:1.2f}")

FWHM = 40 * u.km / u.s
T = 10000 * u.K
M = 1 * const.m_p

sigma_vturb = np.sqrt(FWHM**2 / (8 * np.log(2)) - const.k_B * T / M)
print(sigma_vturb)

print((sigma_vturb**2 + const.k_B * T / (M * 28))**(0.5) * np.sqrt(8 * np.log(2)))