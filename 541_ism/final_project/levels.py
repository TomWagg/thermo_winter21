import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u

import terms

__all__ = ["plot_energy_levels", "is_forbidden"]

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


def plot_energy_levels(spec_terms, transitions, title=None, show_term_labels=True, show_n_labels=True,
                       fig=None, ax=None, show=True, figsize=(6, 10)):
    """Plot an energy level diagram

    Parameters
    ----------
    spec_terms : `list` of `tuples`
        Spectroscopic terms for each energy level, the first is n = 1
    transitions : `list` of `tuples`
        A list of transitions, where each tuple is (n_u, n_l, lambda), the number of the upper state, the
        number of the lower state and the wavelength of the transition
    title : `str`, optional
        A title for the plot, by default None
    show_term_labels : `bool`, optional
        Whether to label each energy level with its term, by default True
    show_n_labels : `bool`, optional
        Whether to label each energy level with its n, by default True
    fig : `Figure`, optional
        A matplotlib Figure on which to plot, by default it will be automatically create
    ax : Axis, optional
        A matplotlib Axis on which to plot, by default it will be automatically create
    show : `bool`, optional
        Whether to show the matplotlib figure immediately, by default True
    figsize : `tuple`, optional
        Size of the figure, by default (6, 10)

    Returns
    -------
    fig, ax
        The figure and axis with the plot on it
    """
    # create a figure if necessary
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # work out the max n used in the transitions and sort the transitions
    max_n = max(transitions)[0]
    transitions = sorted(transitions, key=lambda x: (x[1], x[0]), reverse=False)

    # create some dummy x values (1 for each transition plus some padding on the sides)
    x_vals = np.arange(len(transitions) + 2)

    # track the previous L, colour indicator, current height and all previous heights
    previous_L = spec_terms[0][1]
    colour_ind = 0
    height = 1
    heights = [1]

    # loop over the terms (ignoring any above transitions)
    for n, term in enumerate(spec_terms[:max_n]):
        # add to the height
        height += 1

        # if the L has changed then add extra to the height and change colour
        L = term[1]
        if L != previous_L:
            height += 2
            colour_ind += 1

        # plot the energy level
        ax.plot(x_vals, np.ones_like(x_vals) * height, color=f"C{colour_ind}", lw=2)

        # add term and n labels as desired
        if show_term_labels:
            ax.annotate(terms.format_terms(term, use_latex=True), xy=(-0.1, height),
                        ha="right", va="center", fontsize=fs)
        if show_n_labels:
            ax.annotate(f"n = {n + 1}", xy=(x_vals[-1] + 0.1, height),
                        ha="left", va="center", fontsize=0.8*fs)

        # update L and append new height
        previous_L = L
        heights.append(height)

    # loop over transitions
    for x, transition in enumerate(transitions):
        t_flag, _ = apply_selection_rules(spec_terms[transition[0] - 1], spec_terms[transition[1] - 1])
        linestyle = "-" if t_flag == 0 else ("dashed" if t_flag == 1 else "dotted")

        # work out the height of the upper and lower levels (plus a midpoint)
        upper_height = heights[transition[0]]
        lower_height = heights[transition[1]]
        midpoint = lower_height + ((upper_height - lower_height) / 2)

        # plot an arrow from the upper to the lower
        ax.annotate("", xytext=(x + 1, upper_height), xy=(x + 1, lower_height),
                    arrowprops=dict(arrowstyle="-", linewidth=2, shrinkB=5, linestyle=linestyle, color="grey"))
        ax.annotate("", xytext=(x + 1, upper_height), xy=(x + 1, lower_height),
                    arrowprops=dict(arrowstyle="-|>", linewidth=0, mutation_scale=20, color="grey"))

        # add an annotation of the wavelength at the midpoint of the arrow
        wavelength = transition[2].to(u.Angstrom).value
        t_string = f"{wavelength:1.1f}" if wavelength < 1e5 else f"{wavelength:1.1e}"
        ax.annotate(t_string + r"$\rm \AA$", xy=((x + 1) + 0.125, midpoint), rotation=90,
                    va="center", fontsize=0.6*fs, color="grey", bbox=dict(boxstyle="round", ec="none",
                                                                          fc="white"))

    # add a title if desired
    if title is not None:
        ax.set_title(title, fontsize=fs, y=-0.05)

    # fix the axis limits and hide the axis
    ax.set_xlim(-0.5, x_vals[-1] + 0.5)
    ax.axis("off")

    # show plot if desired
    if show:
        plt.show()
    return fig, ax


def apply_selection_rules(term_low, term_up):
    """Apply electric dipole transition selection rules to a pair of spectroscopic terms.

    This follows the rules from Draine Section 6.7

    Parameters
    ----------
    term_low : `tuple`
        Spectroscopic term of lower level in form (2S+1, L, J, parity)
    term_up : `tuple`
        Spectroscopic term of lower level in form (2S+1, L, J, parity)

    Returns
    -------
    flag : `int`
        A flag describing the type of transition (0=permitted, 1=semiforbidden, 2=forbidden)
    reason : `str`
        A list of reasons for why a flag was chosen
    """
    mult_low, L_low, J_low, parity_low = term_low
    mult_up, L_up, J_up, parity_up = term_up
    S_low = (mult_low - 1) / 2
    S_up = (mult_up - 1) / 2

    parity_changed = parity_low != parity_up
    delta_J_allowed = (abs(J_low - J_up) in [0, 1]) and not (J_low == 0 and J_up == 0)

    delta_L_allowed = (abs(L_low - L_up) in [0, 1]) and not (L_low == 0 and L_up == 0)
    spin_unchanged = S_low == S_up

    semiforbidden = parity_changed & delta_J_allowed & delta_L_allowed
    permitted = semiforbidden & spin_unchanged

    reason_strings = [
        "The parity changed",
        f"J changed from {J_low} to {J_up} - this is forbidden",
        f"L changed from {L_low} to {L_up} - this is forbidden",
        f"The spin changed (from {S_low} to {S_up})",
    ]
    flags = [parity_changed, delta_J_allowed, delta_L_allowed, spin_unchanged]
    reason = ""
    for flag, string in zip(flags, reason_strings):
        if not flag:
            reason += f"- {string}\n"
    if reason == "":
        reason = "All rules obeyed"

    return 0 if permitted else (1 if semiforbidden else 2), reason
