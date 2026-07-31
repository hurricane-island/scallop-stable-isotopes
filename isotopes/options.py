"""
Enums and Click options for the isotopes package.
"""

from enum import Enum
from pathlib import Path
from click import option, Choice

bad_run_dates = {"9/6/23"}
figures = Path(__file__).parent / "../figures"
data_dir = Path(__file__).parent / "../data"
raw_data = data_dir / "gonadosomatic-index.csv"
isotopes_no_outliers = data_dir / "stable-isotopes-no-outliers.csv"
isotopes_raw = data_dir / "stable-isotopes.csv"
env_data = data_dir / "temperature-and-light.csv"
custom_colors = ("black", "blue", "red")


class Command(Enum):
    """
    Let the linter help with making sure we use consistent command names.
    Only needed for commands that are used many times, where a typo is more likely
    from copy-pasting.
    """

    GEAR_TYPE = "gear"
    TEMPERATURE = "temperature"


class Dimension(Enum):
    """
    Let the linter help with making sure we use consistent column names.
    Only needed for columns that are used many times, where a typo is more likely
    from copy-pasting.
    """

    NITROGEN_PERCENTAGE = "% N"
    CARBON_FRACTIONATION = "d13C"
    CARBON_PERCENTAGE = "%C"
    NITROGEN_FRACTIONATION = "d15N"
    MOLAR_RATIO = "C/N (Molar)"
    GEAR = "Gear Type"
    COLLECTION_DATE = "Collection Date"
    TISSUE = "Tissue Type"
    SEX = "Sex"
    DATE_RUN = "Date Run"


class GSIDimension(Enum):
    """
    Let the linter help with making sure we use consistent column names.
    Only needed for columns that are used many times, where a typo is more likely
    from copy-pasting.
    """

    COLLECTION_DATE = "Month"
    GEAR = "Gear Type"
    SEX = "Sex"
    SHELL_HEIGHT = "Shell_Height"
    TOTAL_VISCERA_WEIGHT = "Total_Viscera_Weight"
    MUSCLE_WEIGHT = "Meat_Weight"
    GONAD_WEIGHT = "Gonad_Weight"
    GSI = "GSI"


class EnvDimension(Enum):
    """
    Enum for the different dimensions in the temperature dataset.
    """

    DATE = "Date-Time (EDT)"
    NET_TOP_TEMP = "Net Top, Temperature (°F)"
    CAGE_TEMP = "Cage, Temperature (°F)"
    CAGE_LUM = "Cage, Light (lum)"
    NET_BOTTOM_TEMP = "Net Bottom, Temperature (°F)"
    NET_BOTTOM_LUM = "Net Bottom, Light (lum)"
    WILD_TEMP = "Wild, Temperature (°F)"


class CultureMethod(Enum):
    """
    Enum for the different culture methods.
    """

    CAGE = "C"
    NET = "N"
    WILD = "W"


class TissueType(Enum):
    """
    Enum for the different tissue types.
    """

    MUSCLE = "M"
    GONAD = "G"


def figure_size(default_size: tuple[float, float]):
    """
    Decorator to add a --figsize option to a Click command.
    """

    def decorator(cmd):
        return option(
            "--figsize",
            nargs=2,
            default=default_size,
            help="Size of the output figure in inches (width, height).",
        )(cmd)

    return decorator


def tissue_type_option(default: TissueType):
    """
    Decorator to add a --tissue-type option to a Click command.
    """

    def decorator(cmd):
        return option(
            "--tissue-type",
            default=default,
            help="Tissue type to analyze.",
            type=Choice(TissueType, case_sensitive=False),
        )(cmd)

    return decorator
