"""
Command line interface for the scallop stable isotope analysis module.
"""
from isotopes import plot, summarize
from click import group

@group()
def cli():
    """
    Command line interface for the scallop stable isotope analysis module.
    """

if __name__ == "__main__":
    cli.add_command(summarize)
    cli.add_command(plot)
    cli()