# pylint: disable=too-many-lines,redefined-builtin
"""
This module contains methods for statistical analysis of stable isotope data,
especially Principal Component Analysis (PCA) using the `sklearn` and `scipy`
libraries, and Factor Analysis of Mixed Data (FAMD) using the `prince` library.
"""
from click import group
from isotopes.plot import plot
from isotopes.describe import describe

@group()
def isotopes():
    """
    Command line interface for the scallop stable isotope analysis module.
    """

isotopes.add_command(describe)
isotopes.add_command(plot)
