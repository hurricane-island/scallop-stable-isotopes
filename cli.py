"""
Command line interface for the scallop stable isotope analysis module.
"""
from click import group
from isotopes.plot import plot
from isotopes.describe import describe

@group()
def cli():
    """
    Command line interface for the scallop stable isotope analysis module.
    """

cli.add_command(describe)
cli.add_command(plot)

if __name__ == "__main__":
    cli()
