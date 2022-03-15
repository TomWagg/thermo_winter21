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


def main():

    an_val = []
    sim_val = []

    N_range = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    for N in N_range:
        print("N =", N, "starting")

        mass = 1
        sim = Simulation(N=N, E=10, size=1000, radius=5, masses=mass, delay=0, visualise=True)
        steady = sim.run_simulation(run_until_steadystate=True)
        print("  steady state reached after {:1.2f} steps".format(steady))

        sim.wall_momenta = []

        seconds = 1000
        sim.run_simulation(seconds=seconds)

        sim_val.append(np.sum(sim.wall_momenta) / (4 * sim.size) / seconds)

        speeds = np.sqrt(np.sum(sim.vel**2, axis=1))
        v_rms = np.sqrt(np.mean(speeds**2))

        n = sim.N / sim.size**2
        kBT = 0.5 * mass * v_rms**2

        an_val.append(n * kBT)

        print("  done")

    np.save("data/an_val_4.npy", an_val)
    np.save("data/sim_val_4.npy", sim_val)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), gridspec_kw={"height_ratios": [3, 2]})
    fig.subplots_adjust(hspace=0)

    axes[0].plot(N_range, sim_val, label=r"$P$", color=plt.get_cmap("viridis")(0.9),
                 marker="o", markersize=12)
    axes[0].plot(N_range, an_val, label=r"$n k_B T$", color=plt.get_cmap("viridis")(0.3),
                 marker="o", markersize=12)

    axes[1].scatter(N_range, np.divide(sim_val, an_val), label=r"$P / n k_B T$",
                    color=plt.get_cmap("viridis")(0.5), s=100)
    axes[1].set_ylim(0.5, 1.5)
    axes[1].axhspan(0.8, 1.2, color="grey", alpha=0.2, zorder=-1)
    axes[1].axhline(1, color="black", linestyle="dotted", lw=1, zorder=0)

    axes[0].legend()
    axes[0].set_ylabel("Pressure")
    axes[1].set_xlabel(r"Number of particles, $N$")
    axes[1].set_ylabel(r"Ratio, $P / n k_B T$")

    plt.savefig("figures/4.pdf", format="pdf", bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    main()
