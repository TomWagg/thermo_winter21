import astropy.units as u
import astropy.constants as const
import numpy as np

import sys
sys.path.append("../final_project")

import terms
import configurations as con

print("Part a")
print(f"Electronic configuration of SII is: {con.electronic_configuration(15, formatted=True)}")

print("\nPart b")
print("Spectroscopic terms of SII in order are:", *terms.get_spectroscopic_terms(3, 1, 3, formatted=True))

print("\nPart c")
ns = np.array([[0.999, 0.00008, 0.0004],
               [0.964, 0.0088, 0.027],
               [0.787, 0.084, 0.129]])
A_31 = 2.6e-4 / u.s
A_21 = 8.8e-4 / u.s
lambda_31 = 6716.5 * u.Angstrom
lambda_21 = 6730.8 * u.Angstrom
base_rate = None
for i, n_e in enumerate([10, 1e3, 1e5]):
    n_1, n_2, n_3 = ns[i]

    I_31 = n_3 * A_31 * const.h * const.c / lambda_31
    I_21 = n_2 * A_21 * const.h * const.c / lambda_21

    I = I_31 + I_21
    if i == 0:
        base_rate = I
    print(f"Cooling rate for log(n_e)={np.log10(n_e)}: {I.to(u.erg / u.s):1.2e}, {(I / base_rate).decompose():1.1f}")


print("\nPart d")
base_T = None
for i, n_e in enumerate([10, 1e3, 1e5]):
    n_1, n_2, n_3 = ns[i]
    factor = (-const.h * const.c / lambda_31) / const.k_B
    t_ex_21 = factor * np.log(n_2 / n_1)**(-1)
    if i == 0:
        base_T = t_ex_21
    print(f"Excitation temperature: {t_ex_21.to(u.K): 1.1e}, {(t_ex_21 / base_T).decompose():1.2f}")
