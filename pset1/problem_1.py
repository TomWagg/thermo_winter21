# import the code for the simulation
from simulator import Simulation
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
import numpy as np

plt.rc('font', family='serif')
plt.rcParams['text.usetex'] = False
fs = 18

# update various fontsizes to match
params = {'figure.figsize': (12, 8),
          'legend.fontsize': fs,
          'axes.labelsize': fs,
          'xtick.labelsize': 0.9 * fs,
          'ytick.labelsize': 0.9 * fs,
          'axes.linewidth': 1.1,
          'xtick.major.size': 7,
          'xtick.minor.size': 4,
          'ytick.major.size': 7,
          'ytick.minor.size': 4}
plt.rcParams.update(params)


def plot_positions(positions, radius, fig, ax):
    ax.scatter(positions[:, 0], positions[:, 1], s=radius)
    ax.set_xlabel(r"$x \ [\rm cm]$")
    ax.set_ylabel(r"$y \ [\rm cm]$")
    return fig, ax

def plot_velocities(velocities, fig, ax):
    if np.allclose(velocities, velocities[0]):
        ax.axvline(velocities[0], lw=3)
    else:
        ax.hist(velocities, bins="fd")
    ax.set_xlabel(r"Velocity, $v \ [\rm cm \ s^{-1}]$")
    ax.set_ylabel(r"$\mathrm{d}N/\mathrm{d}v$")
    return fig, ax


def main():
    fig, axes = plt.subplots(2, 2, figsize=(13, 13))
    # fig.subplots_adjust(wspace=0.3)

    sim = Simulation(N=100, E=0.1, size=1000, radius=20, masses=1, delay=20, visualise=False)

    fig, axes[0, 0] = plot_positions(positions=sim.pos, radius=sim.radius**2, fig=fig, ax=axes[0, 0])
    fig, axes[0, 1] = plot_velocities(velocities=np.sqrt(np.sum(sim.vel**2, axis=1)), fig=fig, ax=axes[0, 1])

    sim.run_simulation(seconds=10000)

    fig, axes[1, 0] = plot_positions(positions=sim.pos, radius=sim.radius**2, fig=fig, ax=axes[1, 0])
    fig, axes[1, 1] = plot_velocities(velocities=np.sqrt(np.sum(sim.vel**2, axis=1)), fig=fig, ax=axes[1, 1])

    axes[1, 1].set_xlim(left=0)
    axes[0, 1].set_xlim(axes[1, 1].get_xlim())

    axes[0, 1].annotate("Initial", xy=(1.1, 0.5), rotation=270,
                        xycoords="axes fraction", va="center", fontsize=1.2 * fs)

    axes[1, 1].annotate("Final", xy=(1.1, 0.5), rotation=270,
                        xycoords="axes fraction", va="center", fontsize=1.2 * fs)

    for i in [0, 1]:
        axes[i, 0].set_xlim(0, sim.size)
        axes[i, 0].set_ylim(0, sim.size)

    plt.savefig("figures/1b.png", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
