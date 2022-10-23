import numpy as np
import itertools
import collections

L_lookup = ["S", "P", "D", "F", "G", "H", "I", "J", "K"]


def format_terms(*terms):
    strings = []
    for S, L, J in terms:
        term_string = None
        if not J.is_integer():
            term_string = f"{S}{L}({int(J * 2)}/2)"
        else:
            term_string = f"{S}{L}{int(J)}"
        strings.append(term_string)
    return strings


def get_spectroscopic_terms(n, l, n_electron):
    assert l < n
    m_l_range = list(range(-l, l + 1))
    m_s_range = [-1/2, 1/2]

    # these states are available to the electrons in this shell
    electron_states = [(m_l, m_s) for m_l in m_l_range for m_s in m_s_range]

    # find all combination of these states
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

                # add a term for each possible J value
                for J in np.arange(np.abs(L - S), np.abs(L + S) + 1):
                    terms.append((int(2 * S + 1), L_lookup[L], J))

                # go start back at the first row now that the matrix has changed
                break

    return terms


terms = get_spectroscopic_terms(2, 1, 3)

# print out the formatted terms
print(format_terms(*terms))
