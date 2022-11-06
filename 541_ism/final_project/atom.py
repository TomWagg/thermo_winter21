import configurations as con
import terms
import levels
import warnings


class Atom():
    def __init__(self, name=None, n_electron=None, n_ion=0,
                 formatted=True, use_latex=False, quiet=False):
        """A class for accessing various electronic configuration/spectroscopic term/energy level diagram
        functions.

        Parameters
        ----------
        name : `str`, optional
            The name of an atom or ion, by default None
        n_electron : `int`, optional
            A number of electrons, by default None
        n_ion : `int`, optional
            Number of times it has been ionised, by default None
        formatted : `bool`, optional
            Whether to format variable into strings, by default True
        use_latex : `bool`, optional
            Whether to turn string variables into LaTeX, by default False
        quiet : `bool`, optional
            Whether to silence warnings, by default False

        Raises
        ------
        ValueError
            If neither `name` or `n_electron` are supplied
        """
        if n_electron is None and name is None:
            raise ValueError("One of `n_electron` or `name` must be supplied")
        elif n_electron is None:
            n_electron, n_ion = con.parse_electrons(name)

        self.name = name if name is not None else f"({n_electron} - {n_ion}) electrons"
        self._n_electron = n_electron
        self._n_ion = n_ion
        self.formatted = formatted
        self.use_latex = use_latex
        self.quiet = quiet

    def __repr__(self):
        return f"<Atom: {self.name}, {self.configuration}, {self.terms[-1]}>"

    @property
    def _configuration(self):
        return con.get_configuration(n_electron=self._n_electron, n_ion=self._n_ion)

    @property
    def configuration(self):
        if self.formatted:
            return con.format_configuration(self._configuration, use_latex=self.use_latex)
        else:
            return self._configuration

    @property
    def _terms(self):
        if not self.quiet and con.has_many_half_filled_shells(self._configuration):
            warnings.warn(("This atom/ion has more than one half-filled subshell! The terms returned by this "
                           "function only correspond to the *outermost* half-filled subshell."))
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
