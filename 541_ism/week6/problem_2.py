from scipy.optimize import fsolve
import numpy as np
import astropy.constants as const
import astropy.units as u
import matplotlib.pyplot as plt

import sys
sys.path.append("../final_project")
from atom import Atom

# relative solar abundance of oxygen vs. hydrogen
# http://hyperphysics.phy-astr.gsu.edu/hbase/Tables/suncomp.html
fO_H = 0.043 / 91.2

# energy emitted from the [OII 3727] transition
wavelength = 3728.8 * u.Angstrom
hnu = (const.h * const.c / wavelength)

print(f"h nu = {hnu.to(u.eV):1.2f}")


# total recombination coefficient
def alpha(T4):
    return 4e-13 * (u.cm**3 / u.s) * (T4)**(-0.73)


def HII_cooling(T, T_init):
    """Function for fsolve - for HII cooling a pure HII region"""
    T4 = T / 1e4
    return T * (0.86 + 0.54 * (T4)**(0.37)) - 3/2 * T_init


def HII_cooling_with_OII(T, T_init):
    """Function for fsolve - for HII cooling with a particular OII line"""
    T4 = T / 1e4

    oII_term = (T4**(0.753 - 0.008 * np.log(T4)) * (T * u.K)**(-1/2)
                * 1.57e8 * u.cm**(-3) * u.s * (const.h**3 * const.c /
                (wavelength * (2 * np.pi * const.m_e)**(3/2) * (const.k_B)**(3/2))))

    return T * (0.86 + 0.54 * (T4)**(0.37)) + oII_term.to(u.K).value - 3/2 * T_init


print(fsolve(HII_cooling, 32000, args=(32000)))
print(fsolve(HII_cooling_with_OII, 32000, args=(32000)))

fig, ax = Atom("OII").plot_energy_levels([(2, 1, 3276.0 * u.Angstrom),
                                          (3, 1, 3278.8 * u.Angstrom),
                                          (4, 1, 2470.3 * u.Angstrom),
                                          (5, 1, 2470.2 * u.Angstrom)],
                                         show_all_levels=True, show=False, figsize=(15, 6))

plt.savefig("OII_levels.pdf", format="pdf", bbox_inches="tight")
