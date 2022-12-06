import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u

plt.rc('font', family='serif')
plt.rcParams['text.usetex'] = False
fs = 24

# update various fontsizes to match
params = {'figure.figsize': (12, 8),
          'legend.fontsize': 0.7 * fs,
          'axes.labelsize': fs,
          'xtick.labelsize': 0.7 * fs,
          'ytick.labelsize': 0.7 * fs,
          'axes.linewidth': 1.1,
          'xtick.major.size': 7,
          'xtick.minor.size': 4,
          'ytick.major.size': 7,
          'ytick.minor.size': 4}
plt.rcParams.update(params)

def column_density(ew_rest, osc_strength, wavelength):
    return 1.13e17 * ew_rest.to(u.mAngstrom).value / (osc_strength * wavelength.to(u.Angstrom).value**2) * u.cm**(-2)


labels = ["SiII 1304", "SiII 1260", "SiII 1193", "SiII 1190", "HI 1215", "HI 1025", "HI 972", "HI 949"]
wavelengths = [1304.3702, 1260.4221, 1193.2897, 1190.4158, 1215.67, 1025.7222, 972.5367, 949.7430] * u.Angstrom
osc_strengths = [0.0863, 1.18, 0.582, 0.292, 0.4164, 0.07914, 0.02901, 0.01395]
ew_rests = [210.945, 504.717, 434.75, 236.652 + 111.967, 1109.33, 735.994, 641.365, 601.016] * u.mAngstrom

log_column_density_1304 = 14.3786

Ns = np.concatenate((np.repeat(column_density(ew_rests[0], osc_strengths[0], wavelengths[0]), 4),
                     np.repeat(10**(18.4) * u.cm**(-2), 4)))

calc_Ns = column_density(ew_rests, osc_strengths, wavelengths).to(u.cm**(-2)).value
order = np.argsort(np.log10(ew_rests / wavelengths))
plt.plot(np.log10(osc_strengths * wavelengths.value * calc_Ns)[order], np.log10(ew_rests / wavelengths)[order],
         color="grey", linestyle="dashed", label="Linear Regime", zorder=-1)

x = np.log10(osc_strengths * wavelengths.value * Ns.to(u.cm**(-2)).value)
y = np.log10(ew_rests / wavelengths)
plt.scatter(x[:4], y[:4], label="SIII", s=100)
plt.scatter(x[4:], y[4:], label="HI", s=100)


for xx, yy, label in zip(x, y, labels):
    plt.annotate(label, xy=(xx, yy * 1.005), ha="center", va="top", color="grey")
plt.annotate("Linear Fit", xy=(16.75, -3.2), rotation=78, ha="center", va="center",
             fontsize=0.6 * fs, color="grey")

plt.xlabel(r"$\log_{10}(Nf\lambda \,/\, ({\rm cm^{-2} \AA}))$")
plt.ylabel(r"$\log_{10}(W_\lambda / \lambda \,/\, ({\rm m\AA / \AA}))$")
# plt.legend()

plt.title("Curve of Growth", fontsize=1.2 * fs)

plt.savefig("cog.pdf", format="pdf", bbox_inches="tight")

plt.show()