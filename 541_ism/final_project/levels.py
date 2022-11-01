import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u

import terms

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


def plot_energy_levels(spec_terms, transitions, title=None):
    x_vals = np.arange(len(transitions) + 2)

    fig, ax = plt.subplots(figsize=(6, 10))

    previous_L = spec_terms[0][1]
    colour_ind = 0
    height = 1
    heights = [1]
    for n, term in enumerate(spec_terms):
        L = term[1]
        height += 1
        if L != previous_L:
            height += 2
            colour_ind += 1
        ax.plot(x_vals, np.ones_like(x_vals) * height, color=f"C{colour_ind}", lw=2)
        ax.annotate(*terms.format_terms(term, use_latex=True), xy=(-0.1, height),
                    ha="right", va="center", fontsize=fs)
        ax.annotate(f"n = {n + 1}", xy=(x_vals[-1] + 0.1, height),
                    ha="left", va="center", fontsize=0.8*fs)
        previous_L = L
        heights.append(height)

    for x, transition in enumerate(transitions):
        upper_height = heights[transition[0]]
        lower_height = heights[transition[1]]
        midpoint = lower_height + ((upper_height - lower_height) / 2)
        ax.annotate("", xytext=(x + 1, upper_height), xy=(x + 1, lower_height),
                    arrowprops=dict(arrowstyle="-|>", linewidth=2, color="grey"))

        wavelength = transition[2].to(u.Angstrom).value
        t_string = f"{wavelength:1.1f}" if wavelength < 1e5 else f"{wavelength:1.1e}"
        ax.annotate(t_string + r"$\rm \AA$", xy=((x + 1) * 1.01, midpoint), rotation=90,
                    va="center", fontsize=0.6*fs, color="grey")

    ax.set_xlim(-0.5, x_vals[-1] + 0.5)
    if title is not None:
        ax.set_title(title, fontsize=fs, y=-0.05)
    ax.axis("off")

    plt.show()


# plot_energy_levels(terms.get_spectroscopic_terms(3, 1, 3)[:3],
#                    transitions=[(2, 1, 6730.8 * u.Angstrom),
#                                 (3, 1, 6716.5 * u.Angstrom),
#                                 (3, 2, 3.145e6 * u.Angstrom)],
#                    title="SII")
