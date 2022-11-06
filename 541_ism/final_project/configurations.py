from copy import copy
import numpy as np
from constants import L_lookup, level_sizes, aufbau_order

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


def parse_electrons(input_string):
    """Parse the number of electrons and ionisation from an input string.

    Parameters
    ----------
    input_string : `str`
        An input string, something like "Carbon", "HII", "O3+"

    Returns
    -------
    n_elec : `int`
        Number of electrons
    n_ion : `int`
        Number of times to ionise
    """
    final_char = input_string[-1]
    if final_char in ["+", "-"]:
        if input_string[-2].isdigit():
            n_ion = int(input_string[-2]) * (-1 if final_char == "-" else 1)
            n_elec = electrons_from_element(input_string[:-2])
        else:
            n_ion = -1 if final_char == "-" else 1
            n_elec = electrons_from_element(input_string[:-1])
    elif final_char == "I":
        n_ion = 0
        i = -2
        while input_string[i] == "I":
            n_ion += 1
            i -= 1
        n_elec = electrons_from_element(input_string[:i + 1])
    else:
        n_elec = electrons_from_element(input_string)
        n_ion = 0
    return n_elec, n_ion


def electrons_to_string(n_electron, n_ion):
    if n_ion < 0:
        ion_string = f"{-n_ion}-"
    else:
        # add 1 since n_ion=1 corresponds to II not I
        n_ion += 1

        # convert n_ion to a roman numeral
        roman_num = [1, 4, 5, 9, 10, 40, 50, 90, 100, 400, 500, 900, 1000]
        roman_sym = ["I", "IV", "V", "IX", "X", "XL", "L", "XC", "C", "CD", "D", "CM", "M"]
        i = 12

        ion_string = ""
        while n_ion:
            div = n_ion // roman_num[i]
            n_ion %= roman_num[i]

            while div:
                ion_string += roman_sym[i]
                div -= 1
            i -= 1
    n_elec, symbols, _ = read_elements()

    matching_symbol = symbols[n_elec.astype(int) == n_electron]
    if len(matching_symbol) == 0:
        raise ValueError(f"Can't find an element to match your `n_electron` value - `{n_electron}`")

    # print(symbols, n_electron)
    return f"{matching_symbol[0]}{ion_string}"


def get_configuration(n_electron, n_ion=0, formatted=False, use_latex=False):
    """Get the electronic configuration of an atom/ion given a number of electrons

    Parameters
    ----------
    n_electron : `int`
        Number of electrons
    n_ion : `int`
        Number of times to ionise
    formatted : `bool`, optional
        Whether to format the configuration into a string, by default False
    use_latex : `bool`, optional
        Whether to use LaTeX in the formatted string, by default False

    Returns
    -------
    configuration : `various`
        The electronic configuration
    """
    # if we are gaining electrons then just add them directly
    if n_ion < 0:
        n_electron += (-n_ion)
        n_ion = 0

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
        for n, l in aufbau_order:
            # fill subshells until you run out of electrons
            n_filled = level_sizes[l] if level_sizes[l] <= n_electron else n_electron
            n_electron -= n_filled
            configuration.append((n, l, n_filled))

            if n_electron <= 0:
                break

    # ionise the configuration if necessary
    if n_ion > 0:
        configuration = ionise_configuration(configuration=configuration, n_ion=n_ion)

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


def ionise_configuration(configuration, n_ion):
    """Ionise a configuration a given number of times

    Parameters
    ----------
    configuration : `list` of `tuples`
        Electronic configuration
    n_ion : `int`
        Number of times to ionise, must be positive

    Returns
    -------
    ionised_configuration : `list` of `tuples`
        Configuration after ionisation
    """
    assert n_ion > 0, "Number of ionisations must be positive"

    # sort in descending n, l
    ionised_configuration = sorted(copy(configuration), reverse=True)

    # step through each shell
    for i, conf in enumerate(ionised_configuration):
        n, l, n_filled = conf

        # remove electrons from this shell
        n_left = n_filled - n_ion
        ionised_configuration[i] = (n, l, n_left)

        # if there are still electrons left to remove then change n_ion
        if n_left < 0:
            n_ion -= n_filled
        else:
            break

    # add the configuration back in order for any shells that have a positive n_filled value
    configuration = [(n, l, n_filled) for n, l, n_filled in sorted(ionised_configuration) if n_filled > 0]
    return configuration


def has_many_half_filled_shells(configuration):
    """Check whether a configuration has more than 1 half-filled subshell

    Parameters
    ----------
    configuration : `list` of `tuples`
        Electronic configuration

    Returns
    -------
    flag : `bool`
        Returns true when more than 1 are half-filled
    """
    n_half_filled = 0
    # check each subshell and whether it has reached the level size
    for _, l, n_filled in configuration:
        if n_filled < level_sizes[l]:
            n_half_filled += 1
    return n_half_filled > 1
