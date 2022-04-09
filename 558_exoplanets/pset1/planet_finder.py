import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import astropy.units as u
import astropy.constants as const
import emcee
import corner

# ANSI colour codes
RED = "\033[0;31m"
GREEN = "\033[0;32m"
BOLD = "\033[1m"
UNDERLINE = "\033[4m"
END = "\033[0m"

plt.rc('font', family='serif')
plt.rcParams['text.usetex'] = False
fs = 24

# update various fontsizes to match
params = {'figure.figsize': (12, 8),
          'legend.fontsize': fs,
          'axes.labelsize': fs,
          'xtick.labelsize': 0.6 * fs,
          'ytick.labelsize': 0.6 * fs,
          'axes.linewidth': 1.1,
          'xtick.major.size': 7,
          'xtick.minor.size': 4,
          'ytick.major.size': 7,
          'ytick.minor.size': 4}
plt.rcParams.update(params)


def g(E, e, M):
    """ Kepler's equation in functional form (g(E, e, M) = 0) """
    return E - e * np.sin(E) - M


def g_prime(E, e):
    """ Derivative of g(E, e, M) """
    return 1 - e * np.cos(E)


def solve_kepler_equation(e, M, tol=1e-10, max_steps=10000, verbose=False):
    """Solve Kepler's equation using Newton's method

    Parameters
    ----------
    e : `float`
        Eccentricity
    M : `float`
        Mean Anomaly
    tol : `float`, optional
        Tolerance, meaning the difference between E across steps at which to declare we've converged to a
        solution (too big and the value will be wrong, too small and you may not converge), by default 1e-10
    max_steps : `int`, optional
        Maximum number of steps to take, by default 10000
    verbose : `bool`, optional
        Whether to print out detailed information, by default False

    Returns
    -------
    E : `float`
        Eccentric anomaly
    """
    # make an initial guess of the eccentric anomaly
    E = M + 0.85 * e * np.sign(np.sin(M))
    steps = 0
    tol_to_dp = int(abs(np.log10(tol)))

    if verbose:
        print("Initial guess:", np.round(E, tol_to_dp))

    # loop until you reach the maximum number of steps
    while steps < max_steps:
        # calculate the next value of E
        new_E = E - g(E, e, M) / g_prime(E, e)
        if verbose:
            print(f"Step {steps} guess: {np.round(new_E, tol_to_dp)}")

        # check whether the value of E has change significantly (if not then return)
        if np.abs(new_E - E) < tol:
            return E

        # update E and the steps
        E = new_E
        steps += 1

    raise ValueError("Did not converge")


def test_kepler_solver(n_tests=10000, **kwargs):
    """Test whether the solver for Kepler's equation is working

    Parameters
    ----------
    n_tests : `int`, optional
        Number of random tests to do, by default 10000

    Returns
    -------
    e, M, E : `tuple of array of floats` :p
        Eccentricity and anomalies, returned only when tests fail, otherwise None returned
    """
    print(f"Running {n_tests} tests of the Kepler equation solver")

    # create random e in [0, 1) and random M in [0, 2 pi)
    e = np.random.uniform(0, 1, size=n_tests)
    M = np.random.uniform(0, 2 * np.pi, size=n_tests)

    # calculate eccentric anomaly using the solver
    E = np.array([solve_kepler_equation(e[i], M[i], **kwargs) for i in range(n_tests)])
    # ensure that the values are sensible
    assert np.all(np.logical_and(E > 0, E < 2 * np.pi))

    # apply Kepler's equation to the solutions we found (should give zeros)
    should_be_zero = g(E, e, M)

    # check whether the values are actually zero or not
    if np.allclose(should_be_zero, np.zeros(n_tests)):
        print(f"{GREEN}You passed!{END}")
        return None
    else:
        fail_frac = len(should_be_zero[np.isclose(should_be_zero, np.zeros(n_tests))]) / len(should_be_zero)
        print(f"{RED}Ah sad, you failed, looks like {np.round(fail_frac * 100, 1)}% of tests failed{END}")
        return e, M, E


def radial_velocity(k_star, omega, e, t, t_p, P, gamma):
    """Find the radial velocity of a star

    Parameters
    ----------
    k_star : `float`
        Velocity semi-amplitude
    omega : `float`
        Pericenter angle
    e : `float`
        Eccentricity
    t : `float`
        Time
    t_p : `float`
        Time of Perihelion
    P : `float`
        Period
    gamma : `float`
        CoM radial velocity

    Returns
    -------
    v_rv : `float`
        Radial velocity
    """
    M = 2 * np.pi / P * (t - t_p)
    E = solve_kepler_equation(e, M)
    f = 2 * np.arctan(((1 + e) / (1 - e))**(0.5) * np.tan(E / 2))

    return k_star * (np.cos(omega + f) + e * np.cos(omega)) + gamma


def get_rvs(times, P, k_star, t_p, gamma, omega, e):
    """Get radial velocities over time for a system

    Parameters
    ----------
    times : `list`
        Times at which to evaluate
    k_star : `float`
        Velocity semi-amplitude
    omega : `float`
        Pericenter angle
    e : `float`
        Eccentricity
    t_p : `float`
        Time of Perihelion
    P : `float`
        Period
    gamma : `float`
        CoM radial velocity

    Returns
    -------
    v_rv : `np.array`
        Radial velocities
    """
    return np.array([radial_velocity(P=P, k_star=k_star, t=t, t_p=t_p,
                                     gamma=gamma, omega=omega, e=e) for t in times])


def log_prior(theta, bounds):
    """Find the log of the prior using some bounds. Basically don't let the MCMC choose values out of these
    bounds

    Parameters
    ----------
    theta : `list`
        List of the 6 parameters (P, k_star etc.)
    bounds : `list of tuples`
        Bounds for each of the 6 parameters

    Returns
    -------
    lg_prior: `float`
        Log of the prior probability
    """
    for i in range(len(bounds)):
        if theta[i] < bounds[i][0] or theta[i] >= bounds[i][1]:
            return -np.inf
    return 0.0


def log_likelihood(theta, data_times, data_rvs, data_errors):
    """Find the log of the likelihood given the model parameters and data (following emcee tutorial)

    Parameters
    ----------
    theta : `list`
        List of parameters
    data_times : `list`
        Times from the data
    data_rvs : `list`
        Radial velocities from the data
    data_errors : `list`
        Errors on the RVs from the data

    Returns
    -------
    lg_like : `float`
        Log likelihood of these parameters
    """
    P, k_star, t_p, gamma, omega, e = theta
    model_rvs = get_rvs(data_times, P=P, k_star=k_star, t_p=t_p,
                        gamma=gamma, omega=omega, e=e)

    sigma2 = data_errors**2
    return -0.5 * np.sum((data_rvs - model_rvs)**2 / sigma2 + np.log(sigma2))


def log_prob(theta, data_times, data_rvs, data_errors, bounds):
    """Find the log probability of a set of parameters given data and bounds

    Parameters
    ----------
    theta : `list`
        List of parameters
    data_times : `list`
        Times from the data
    data_rvs : `list`
        Radial velocities from the data
    data_errors : `list`
        Errors on the RVs from the data
    bounds : `list of tuples`
        Bounds for each of the 6 parameters

    Returns
    -------
    lg_prob : `float`
        Log probability of these parameters
    """
    prior = log_prior(theta, bounds)
    if not np.isfinite(prior):
        return prior
    else:
        return log_likelihood(theta, data_times, data_rvs, data_errors) + prior


def planet_mass(P, k_star, e, m_star, i):
    """Find a planet mass given various parameters (NOTE: this assumes m_p << m_s)

    Parameters
    ----------
    P : `float`
        Period
    k_star : `float`
        Velocity semi-amplitude
    e : `float`
        Eccentricity
    m_star : `float`
        Mass of the star
    i : `float`
        Inclination

    Returns
    -------
    m_planet : `float`
        Mass of the planet
    """
    return k_star * (2 * np.pi * const.G / (P * (1 - e**2)))**(-1/3) / np.sin(i) * m_star**(2/3)


class PlanetFinder():
    def __init__(self, file):
        """PlanetFinder class for fitting a planet from RV data. Initial function reads the data in.

        Parameters
        ----------
        file : `string`
            Path to the data
        """
        self.file = file

        planet = pd.read_csv(self.file, sep="\t", names=["time", "rv", "rv_err"])
        self.time = planet["time"] - planet["time"].min()
        self.rv = planet["rv"]
        self.rv_err = planet["rv_err"]
        self.period_lsq = None

    def least_squares_period(self, period_range):
        """Estimate the period of the planet using least squares fitting when folding on the period. I.e. the
        change in radial velocity between subsequent points after folding on the period should be minimised.

        Parameters
        ----------
        period_range : `list`
            List of periods to try

        Returns
        -------
        period_lsq : `float`
            Best fitting period
        """
        least_squares = np.zeros_like(period_range)

        # loop over each period
        for i, period in enumerate(period_range):
            # folder the data on this period
            folded_time = self.time % period

            # sort the data by new folded times
            order = np.argsort(folded_time)
            folded_time = folded_time[order]
            rvs = self.rv[order].values

            # calculate the sum of the squares for this period
            least_squares[i] = np.sum(((rvs[1:] - rvs[:-1]))**2)

        # pick the minimum and return it
        self.period_lsq = period_range[least_squares.argmin()]
        return self.period_lsq

    def run_mcmc(self, initial_guesses, bounds, perturbation=1e-5, n_walkers=16, steps=5000, burn_in=1000):
        """Run an MCMC to find the best fits for [P, k_star, t_p, omega, gamma, e]

        Parameters
        ----------
        initial_guesses : `list`
            List of initial guesses for each parameter
        bounds : `list of tuples`
            Bounds on each parameters
        perturbation : `float`, optional
            How large a perturbation on the initial guesses to use for different walkers, by default 1e-5
        n_walkers : `int`, optional
            How many walkers to use, by default 16
        steps : `int`, optional
            How many steps for each walker to take, by default 5000
        burn_in : `int`, optional
            How many steps to discard to allow for a burn in phase, by default 1000
        """
        # get some initial positions by perturbing the initial guesses
        pos = initial_guesses + perturbation * np.random.randn(n_walkers, 6)

        # instantiate a sampler
        sampler = emcee.EnsembleSampler(nwalkers=n_walkers, ndim=6, log_prob_fn=log_prob,
                                        args=(self.time, self.rv, self.rv_err, bounds))

        # run the MCMC with a progress bar
        _ = sampler.run_mcmc(pos, steps, progress=True)

        # get the samples and the best fit
        self.samples = sampler.get_chain(discard=burn_in, flat=True)
        self.best_fit = np.median(self.samples, axis=0)

    def mcmc_corner_plot(self):
        """Create a corner plot of the parameters that were fit by the MCMC"""
        # make everything smaller to fit the corner plot
        params = {
            'axes.labelsize': 0.7 * fs,
            'xtick.labelsize': 0.4 * fs,
            'ytick.labelsize': 0.4 * fs
        }
        plt.rcParams.update(params)

        corner.corner(
            self.samples, labels=[r"$P\, [{\rm days}]$", r"$k_{*}\, [{\rm m/s}]$", r"$t_p\, [{\rm days}]$",
                                  r"$\gamma\, [{\rm m / s}]$", r"$\omega \, [{\rm rad}]$", r"$e$"]
        )

        plt.savefig("figures/mcmc_corner.pdf", format="pdf", bbox_inches="tight")

        # reset it all afterwards
        params = {
            'axes.labelsize': fs,
            'xtick.labelsize': 0.6 * fs,
            'ytick.labelsize': 0.6 * fs
        }
        plt.rcParams.update(params)

    def plot_best_fit(self):
        """Plot the best fit model against the folded data"""
        fig, ax = plt.subplots()

        # get the period from the best fit
        period = self.best_fit[0]

        # fold and sort the data
        folded_time = self.time % period
        order = np.argsort(folded_time)
        folded_time = folded_time[order]
        folded_rv = self.rv[order]
        folded_rv_err = self.rv_err[order]

        # plot the data with errorbars
        ax.errorbar(x=folded_time, y=folded_rv, yerr=folded_rv_err, fmt=".",
                    color="black", markersize=10, label="Folded data")

        # calculate the model and plot it
        model_times = np.linspace(0, period, 1000)
        model_rvs = get_rvs(model_times, *self.best_fit)
        ax.plot(model_times, model_rvs, color="tab:orange", lw=2, label="Best fit")

        # add a line at zero to help guide the eye
        ax.axhline(0, color="grey", linestyle="dotted")

        ax.set_xlabel("Phase [days]")
        ax.set_ylabel(r"$v_{rv} \, [{\rm m /s}]$")
        ax.legend()

        plt.savefig("figures/best_fit.pdf", format="pdf", bbox_inches="tight")
        plt.show()

    def planet_mass_plot(self):
        """Create a contour plot of the planet mass for different star masses and inclinations"""
        fig, ax = plt.subplots()

        m_star_range = np.logspace(-2, 1, 500) * u.Msun
        i_range = np.linspace(0.01, np.pi/2, 10000) * u.rad

        MS, I = np.meshgrid(m_star_range, i_range)

        m_planets = planet_mass(self.best_fit[0] * u.day,
                                self.best_fit[1] * u.m / u.s,
                                self.best_fit[-1], i=I, m_star=MS).to(u.kg) / const.M_earth

        cont = ax.contourf(i_range, m_star_range, m_planets.T, norm=LogNorm(),
                           levels=np.logspace(np.log10(0.08), 3, 25), cmap="magma")
        cbar = fig.colorbar(cont)

        ax.set_yscale("log")
        ax.set_xlim(0, np.pi/2)
        ax.set_ylim(1e-2, 1e1)

        ax.set_xticks([0, np.pi / 8, np.pi / 4, 3 * np.pi / 8, np.pi / 2])
        ax.set_xticklabels(["0", r"$\pi / 8$", r"$\pi / 4$", r"$3 \pi / 8$", r"$\pi / 2$"])

        ax.set_xlabel(r"Inclination, $i \, [{\rm rad}]$")
        ax.set_ylabel(r"Star Mass, $m_* \, [{\rm M_{\odot}}]$")
        cbar.set_label(r"Planet Mass, $m_p \, [{\rm M_{\oplus}}]$")

        earth = ax.contour(i_range, m_star_range, m_planets.T, norm=LogNorm(),
                           levels=[1, 17], colors="white", linewidths=2)
        ax.clabel(earth, [1, 17], fmt=lambda x: "Earth" if x == 1 else "Neptune",
                  use_clabeltext=True, fontsize=0.7 * fs, manual=[(0.2, 1e-1), (0.1, 3e0)])

        plt.savefig("figures/planet_mass.pdf", format="pdf", bbox_inches="tight")

        plt.show()


def main():
    print()
    print(f"{BOLD}{UNDERLINE}Problem 1 - Create Kepler equation solver and show it works{END}")
    test_kepler_solver(tol=1e-9)

    print()
    print(f"{BOLD}{UNDERLINE}Problem 3 - Find period of planet{END}")
    finder = PlanetFinder(file="data/mystery_planet01.txt")
    finder.least_squares_period(period_range=np.linspace(100, 120, 25000))
    print(f"Looks like the period of this planet is about {GREEN}{BOLD}{finder.period_lsq:1.4f}{END} days")

    print()
    print(f"{BOLD}{UNDERLINE}Problem 4 - Fit the other parameters{END}")

    # last four initial guesses are done by eye
    initial_guesses = [finder.period_lsq, 0.5 * (finder.rv.max() - finder.rv.min()), 10.7, -0.1, 5, 0.92]
    bounds = [(110, 112), (0.2, 0.7), (10.2, 11.2), (-1, 1), (0.01, 2 * np.pi), (0, 1)]
    finder.run_mcmc(initial_guesses=initial_guesses, bounds=bounds)
    print(f"Looks like the best fits are {GREEN}{BOLD}{finder.best_fit}{END}")
    finder.mcmc_corner_plot()
    finder.plot_best_fit()

    print()
    print(f"{BOLD}{UNDERLINE}Problem 4.5 - Tom wondered what the planet mass would be{END}")
    finder.planet_mass_plot()
    print("And here's plot to show it!")

    print()


if __name__ == "__main__":
    main()
