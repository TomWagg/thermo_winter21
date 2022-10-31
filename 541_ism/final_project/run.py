from linetools.lists.linelist import LineList
import terms
import configurations as con

ism = LineList("ISM")

sulfur_II = ism._data[(ism._data["ion"] == 6) & (ism._data["Z"] == 8)]# & (ism._data["A"] > 0.0)]

for line in sulfur_II:
    print(line["name"], line["wrest"], line["Ek"])
    # print(line["nj"], line["Jj"], line["gj"], "-", line["nk"], line["Jk"], line["gk"])

print(con.format_configuration(con.electronic_configuration(8)))
print(con.format_configuration(con.electronic_configuration(3)))
print(terms.format_terms(*terms.get_spectroscopic_terms(2, 0, 1)))
