import numpy as np
from constants import L_lookup, level_sizes


def read_elements():
    element_list = []
    with open("elements.csv") as elements:
        element_list = [line.rstrip("\n").split("\t") for line in elements]
    return np.transpose(element_list)


def electrons_from_element(el_string):
    n_elec, symbols, names = read_elements()
    electrons = []
    if len(el_string) <= 2:
        electrons = n_elec[np.char.lower(symbols) == el_string.lower()]
    else:
        electrons = n_elec[np.char.lower(names) == el_string.lower()]

    if len(electrons) == 0:
        raise ValueError(f"Cannot recognise this element: `{el_string}`")

    return int(electrons[0])


def parse_electrons(source):
    split_source = source.split(" ")
    if len(split_source) > 1:
        element, ionisation = split_source
        n_I = sum([int(ionisation[i] == "I") for i in range(len(ionisation))])
    else:
        element = split_source[0]
        n_I = 1
    return electrons_from_element(element) - n_I + 1


def electronic_configuration(n_electron, formatted=False):
    order = [(1, 0), (2, 0), (2, 1), (3, 0), (3, 1),
             (4, 0), (3, 2), (4, 1), (4, 2), (5, 0), (5, 1)]

    exceptions = {
        24: [(1, 0, 2), (2, 0, 2), (2, 1, 6), (3, 0, 2), (3, 1, 6), (4, 0, 1), (3, 2, 5)],
        29: [(1, 0, 2), (2, 0, 2), (2, 1, 6), (3, 0, 2), (3, 1, 6), (4, 0, 1), (3, 2, 10)]
    }

    if n_electron in exceptions:
        return exceptions[n_electron]

    configuration = []
    for n, l in order:
        n_filled = level_sizes[l] if level_sizes[l] <= n_electron else n_electron
        n_electron -= n_filled
        configuration.append((n, l, n_filled))

        if n_electron <= 0:
            break

    if formatted:
        return format_configuration(configuration)
    else:
        return configuration


def format_configuration(configuration):
    config_string = ""
    for n, l, n_e in configuration:
        config_string += f"{n}{L_lookup[l].lower()}{n_e} "
    return config_string.rstrip()


# for i in range(1, 37):
#     print(i, ":", format_configuration(electronic_configuration(i)))
