import configurations as con
import terms
import levels


class Atom():
    def __init__(self, name=None, n_electron=None, formatted=True, use_latex=False, no_cache=False):
        if n_electron is None and name is None:
            raise ValueError("One of `n_electron` or `name` must be supplied")
        elif n_electron is None:
            n_electron = con.parse_electrons(name)

        self.name = name if name is not None else f"{n_electron} electrons"
        self._n_electron = n_electron
        self.formatted = formatted
        self.use_latex = use_latex
        self.no_cache = no_cache

    def __repr__(self):
        return f"<Atom: {self.name}, {self.configuration}, {self.terms[-1]}>"

    @property
    def _configuration(self):
        return con.get_configuration(self._n_electron)

    @property
    def configuration(self):
        if self.formatted:
            return con.format_configuration(self._configuration, use_latex=self.use_latex)
        else:
            return self._configuration

    @property
    def _terms(self):
        return terms.get_spectroscopic_terms(*self._configuration[-1])

    @property
    def terms(self):
        if self.formatted:
            return terms.format_terms(*self._terms, use_latex=self.use_latex)
        else:
            return self._terms

    def plot_energy_levels(self, transitions, **kwargs):
        return levels.plot_energy_levels(spec_terms=self._terms, transitions=transitions,
                                         title=self.name, **kwargs)
