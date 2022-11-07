from linetools.spectra.xspectrum1d import XSpectrum1D
from linetools.guis import xspecgui as ltxsg
import numpy as np


def run_gui():
    fname = '~/Downloads/J0929+4644_172_157_spec.fits'
    # fname = '~/Downloads/J0943+0531_106_34_redspec.fits'

    sp = XSpectrum1D.from_file(fname)

    ltxsg.main(sp)


# run_gui()

z = [0.01657, 0.22833]
sII_6716 = [583.826, 18.8138]
sII_6731 = [459.247, 14.4956]
oIII_5007 = [14772.4, 16.6357]
H_beta = [2712.44, 38.7836]
nII_6583 = [452.39, 83.1522]
H_alpha = [9469.3, 150.604]

for i in [0, 1]:
    print(f"Galaxy {i + 1}")
    sII_ratio = sII_6716[i] / sII_6731[i]
    O3HB = oIII_5007[i] / H_beta[i]
    N2 = nII_6583[i] / H_alpha[i]
    O3N2 = np.log10(O3HB / N2)

    if O3N2 < 1.9:
        oxy_abund = 8.73 - (0.32 * O3N2)
    else:
        oxy_abund = 8.92 + (0.57 * np.log10(N2))

    print(f"  Redshift = {z[i]:1.3f}")
    print(f"  sII ratio = {sII_ratio:1.2f}")
    print(f"  O3HB = {O3HB:1.2f}")
    print(f"  N2 = {N2:1.2f}")
    print(f"  O3N2 = {O3N2:1.2f}")
    print(f"  Oxygen Abundance = {oxy_abund:1.2f}")
