import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt

def does_stuff(frac):
    tau_nu = -np.log(1 - frac)
    T_spin = 0.552 * 100 * u.K / tau_nu * (3 / 20)
    return tau_nu, T_spin

fracs = [0.01, 0.5]

for frac in fracs:
    print(f"< {frac} fractional change")
    tau_nu, T_spin = does_stuff(frac)
    print(f"  Optical depth < {tau_nu:1.2f}")
    print(f"  Spin temperature > {T_spin:1.0f}")


frac_range = np.logspace(-5, 0, 1000)
plt.loglog(frac_range, does_stuff(frac_range)[0])
plt.show()