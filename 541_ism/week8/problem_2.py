import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u


def column_density(ew_rest, osc_strength, wavelength):
    return 1.13e17 * ew_rest.to(u.mAngstrom).value / (osc_strength * wavelength.to(u.Angstrom).value**2) * u.cm**(-2)


labels = ["SiII 1304", "SiII 1260", "SiII 1193", "SiII 1190"]
wavelengths = [1304.3702, 1260.4221, 1193.2897, 1190.4158] * u.Angstrom
osc_strengths = [0.0863, 1.18, 0.582, 0.292]
ew_rests = [210.945, 504.717, 434.75, 236.652 + 111.967] * u.mAngstrom

log_column_density_1304 = 14.3786
