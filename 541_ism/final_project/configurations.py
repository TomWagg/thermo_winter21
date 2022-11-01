import numpy as np
from constants import L_lookup, level_sizes

__all__ = ["parse_electrons", "get_configuration", "format_configuration"]


def read_elements():
    """Read the list of elements from our tiny csv file

    Returns
    -------
    element_list : :class:`~np.ndarray`
        List of elements, each has (n_electron, symbol, name)
    """
    element_list = []
    with open("data/elements.csv") as elements:
        element_list = [line.rstrip("\n").split("\t") for line in elements]
    return np.transpose(element_list)


def electrons_from_element(el_string):
    """Convert a string representing an element to a number of electrons

    Parameters
    ----------
    el_string : `str`
        String that gives an element

    Returns
    -------
    n_electron : `int`
        Number of electrons corresponding to the string

    Raises
    ------
    ValueError
        If the string supplied doesn't match any elements
    """
    n_elec, symbols, names = read_elements()
    electrons = []

    # if they give a short string then match to a symbol, otherwise name to name
    if len(el_string) <= 2:
        electrons = n_elec[np.char.lower(symbols) == el_string.lower()]
    else:
        electrons = n_elec[np.char.lower(names) == el_string.lower()]

    if len(electrons) == 0:
        raise ValueError(f"Cannot recognise this element: `{el_string}`")

    return int(electrons[0])


def parse_electrons(source):
    # TODO: handle ionisation better
    split_source = source.split(" ")
    if len(split_source) > 1:
        element, ionisation = split_source
        n_I = sum([int(ionisation[i] == "I") for i in range(len(ionisation))])
    else:
        element = split_source[0]
        n_I = 1
    return electrons_from_element(element) - n_I + 1


def get_configuration(n_electron, formatted=False, use_latex=False):
    """Get the electronic configuration of an atom/ion given a number of electrons

    Parameters
    ----------
    n_electron : `int`
        Number of electrons
    formatted : `bool`, optional
        Whether to format the configuration into a string, by default False
    use_latex : `bool`, optional
        Whether to use LaTeX in the formatted string, by default False

    Returns
    -------
    configuration : `various`
        The electronic configuration
    """
    # the order in which to fill shells following the Aufbau principle
    # each tuple is (n, l)
    order = [(1, 0), (2, 0), (2, 1), (3, 0), (3, 1),
             (4, 0), (3, 2), (4, 1), (4, 2), (5, 0), (5, 1)]

    # some exceptions to the principle which we handle manually
    exceptions = {
        24: [(1, 0, 2), (2, 0, 2), (2, 1, 6), (3, 0, 2), (3, 1, 6), (4, 0, 1), (3, 2, 5)],
        29: [(1, 0, 2), (2, 0, 2), (2, 1, 6), (3, 0, 2), (3, 1, 6), (4, 0, 1), (3, 2, 10)]
    }
    if n_electron in exceptions:
        configuration = exceptions[n_electron]
    else:
        # prep the configuration list
        configuration = []

        # loop over the order of subshells
        for n, l in order:
            # fill subshells until you run out of electrons
            n_filled = level_sizes[l] if level_sizes[l] <= n_electron else n_electron
            n_electron -= n_filled
            configuration.append((n, l, n_filled))

            if n_electron <= 0:
                break

    # format the result if the user wants to
    if formatted:
        return format_configuration(configuration, use_latex=use_latex)
    else:
        return configuration


def format_configuration(configuration, use_latex=False):
    """Format an electronic configuration into a string

    Parameters
    ----------
    configuration : `list` of `tuples`
        Electronic configuration, each tuple is (n, l, n_filled)
    use_latex : `bool`, optional
        Whether to use LaTeX in the formatted string, by default False

    Returns
    -------
    config_string : `str`
        The electronic configuration as a string
    """
    config_string = ""
    for n, l, n_e in configuration:
        if use_latex:
            config_string += rf"${{{n}}}{{{L_lookup[l].lower()}}}^{{{n_e}}}$ "
        else:
            config_string += f"{n}{L_lookup[l].lower()}{n_e} "
    return config_string.rstrip()
