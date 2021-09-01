from pandana.core.tables import Tables


class Loader:
    """A class for accessing data in h5py files."""

    def __init__(self, files, idcol, main_table_name, indices):
        self._files = files
        self._idcol = idcol
        self._main_table_name = main_table_name
        self._indices = indices

        self._specdefs = []

    def add_spectrum(self, spec):
        if not spec in self._specdefs:
            self._specdefs.append(spec)

    def Go(self):
        """
        Iterate through the associated spectra and compute the cuts and vars for each
        :return: None
        """
        for f in self._files:
            # Construct the tables for this file
            tables = Tables(
                f, self._idcol, self._main_table_name, indices=self._indices
            )

            # FILL ALL SPECTRA for this file
            for spec in self._specdefs:
                spec.fill(tables)

            tables.closeFile()

        self.Finish()

    def Finish(self):
        # Combine together result for each file
        for spec in self._specdefs:
            spec.finish()
