__all__ = ["L_lookup", "level_sizes"]

# convert a value of L to the letter representation
L_lookup = ["S", "P", "D", "F", "G", "H", "I", "J", "K"]

# get the size of a level for each value of l
level_sizes = [2, 6, 10, 14]

# the order in which to fill shells following the Aufbau principle each tuple is (n, l)
aufbau_order = [(1, 0), (2, 0), (2, 1), (3, 0), (3, 1),
                (4, 0), (3, 2), (4, 1), (4, 2), (5, 0), (5, 1)]
