import numpy as np

# ANSI colour codes
RED = "\033[0;31m"
GREEN = "\033[0;32m"
UNDERLINE = "\033[4m"
END = "\033[0m"


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
    print(f"Running {UNDERLINE}{n_tests}{END} tests of the solver")

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


def main():
    test_kepler_solver(tol=1e-9)


if __name__ == "__main__":
    main()