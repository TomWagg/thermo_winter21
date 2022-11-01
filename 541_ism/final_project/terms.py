import numpy as np
import itertools
import collections
from constants import L_lookup, level_sizes

__all__ = ["format_terms", "get_spectroscopic_terms"]


def _underline_print(string):
    """Same as a regular print call but underlined with hyphens

    Parameters
    ----------
    string : `str`
        Some string to print out
    """
    print(string)
    print("".join(["-" for _ in range(len(string))]))


def format_terms(*terms, use_latex=False):
    """Format a list of spectroscopic terms

    Parameters
    ----------
    *terms : `tuple`
        Each `term` should be a tuple of `2S+1, L, J`
    use_latex : `bool`, optional
        Whether to use LaTeX in the formatted string, by default False

    Returns
    -------
    term_strings : `list` of `str`s
        A list of strings representing each term
    """
    strings = []
    for S, L, J in terms:
        term_string = None
        if not J.is_integer():
            if use_latex:
                term_string = rf"$^{{{S}}} {L_lookup[L]}_{{\frac{{{int(J * 2)}}}{{2}}}}$"
            else:
                term_string = f"{S}{L_lookup[L]}({int(J * 2)}/2)"
        else:
            if use_latex:
                term_string = rf"$^{{{S}}} {L_lookup[L]}_{{{int(J)}}}$"
            else:
                term_string = f"{S}{L_lookup[L]}{int(J)}"
        strings.append(term_string)
    return strings


def get_spectroscopic_terms(n, l, n_electron, formatted=False, use_latex=False, stepbystep=False):
    """Compute the spectroscopic terms for a given level (n, l) that contains n_electron electrons

    Parameters
    ----------
    n : `int`
        Quantum number of state
    l : `int`
        Angular momentum of state
    n_electron : `int`
        Number of electrons in the state
    formatted : `bool`, optional
        Whether to format the result as a string, by default False
    use_latex : `bool`, optional
        Whether to use LaTeX for the formatted string, by default False
    stepbystep : `bool`, optional
        Whether to print out the step-by-step algorithmic approach to getting the terms, by default False

    Returns
    -------
    terms : `various`
        The spectroscopic terms
    """
    assert l < n, "`l` must be less than `n`"
    assert n_electron <= level_sizes[l], "Number electrons must be no more than subshell capacity"
    assert n_electron > 0, "Number electrons must be positive"
    m_l_range = list(range(-l, l + 1))
    m_s_range = [-1/2, 1/2]

    # these states are available to the electrons in this shell
    electron_states = [(m_l, m_s) for m_l in m_l_range for m_s in m_s_range]
    if stepbystep:
        _underline_print("Step 1: Possible electron states (m_l, m_s)")
        # print("-------------------------------------------")
        print(electron_states)
        print()

    # find all combination of these states
    combs = itertools.combinations(electron_states, n_electron)
    if stepbystep:
        _underline_print("Step 2: All possible combinations of states")
        print(list(combs))
        print()
        combs = itertools.combinations(electron_states, n_electron)

    # create a dict to keep track of the L, S combos (for the matrix)
    LS_combos = collections.defaultdict(int)

    # two sets for tracking the total L and S values in each combo
    M_l_vals = set()
    M_s_vals = set()

    # get the information about each possible combination
    for comb in combs:
        M_l, M_s = np.asarray(comb).sum(axis=0)
        LS_combos[(M_l, M_s)] += 1
        M_l_vals.add(M_l)
        M_s_vals.add(M_s)

    # turn the sets of values into ranges and start an empty matrix
    M_s_range = np.arange(min(M_s_vals), max(M_s_vals) + 1)
    M_l_range = np.arange(min(M_l_vals), max(M_l_vals) + 1)
    matrix = np.zeros((len(M_l_range), len(M_s_range))).astype(int)

    # populate the matrix
    for i, M_l in enumerate(M_l_range):
        for j, M_s in enumerate(M_s_range):
            matrix[i, j] = LS_combos[(M_l, M_s)]

    if stepbystep:
        _underline_print("Step 3: Initial matrix")
        print(matrix)
        print()

        _underline_print("Step 4: Matrix reduction")

    # keep track of the terms
    terms = []

    # while there is still a combination remaining in the matrix
    while matrix.sum() > 0:
        # go through each row
        for i in range(len(matrix)):
            # check if any of the columns are nonzero
            columns = matrix[i] > 0
            if columns.any():
                # if yes, then build a unitary matrix with the same width and reflected height
                unitary = np.array([np.repeat(False, len(columns)) if abs(j) > abs(M_l_range[i])
                                    else columns for j in M_l_range]).astype(int)

                # subtract this from the overall matrix for the next round
                matrix -= unitary

                # work out the value of L and S for this term(s) based on the matrix
                L, S = int(abs(M_l_range[i])), M_s_range[columns].max()

                # add a term for each possible J value in the correct order based on how filled the level is
                J_range = np.arange(abs(L - S), abs(L + S) + 1)
                if n_electron > level_sizes[l] / 2:
                    J_range = reversed(J_range)
                for J in J_range:
                    terms.append((int(2 * S + 1), L, J))

                if stepbystep:
                    print(unitary, f"{int(2 * S + 1)}{L_lookup[L]}")

                # go start back at the first row now that the matrix has changed
                break

    if stepbystep:
        print()
        _underline_print("Step 5: Expand terms")
        print(*format_terms(*terms))
        print()

        # it's crazy that I can add a cookie to my code
        print("Step 6: Go eat this 🍪")

    # sort based on Hund's 1st and 2nd rules
    terms = sorted(terms, key=lambda x: (x[0], x[1]), reverse=True)

    if formatted:
        return format_terms(*terms, use_latex=use_latex)
    else:
        return terms
