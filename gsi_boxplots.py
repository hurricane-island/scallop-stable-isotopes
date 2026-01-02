"""
This looks at the GSI data to determine the time of year of spawning for gear types
and how that aligns with isotopic signatures.
"""

from pathlib import Path
from enum import Enum
from pandas import read_csv
from matplotlib.axes import Axes
from matplotlib.pyplot import xticks, ylabel, legend, subplots
from matplotlib import patches as mpatches


figures = Path(__file__).parent / "figures"
raw_data = Path(__file__).parent / "data" / "2023_StableIsotope_GSI_data.csv"


class Dimension(Enum):
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


def load_and_subset_source_data():
    """
    Data needed for GSI boxplots. Converts from data from, to a dictionary grouped by
    tuples of gear and month that can be combined as needed for plotting.
    """
    df = read_csv(
        raw_data,
        header=0,
        usecols=[
            Dimension.COLLECTION_DATE.value,
            Dimension.GEAR.value,
            Dimension.GSI.value,
        ],
    )
    gb = df.dropna().groupby([Dimension.GEAR.value, Dimension.COLLECTION_DATE.value])
    return gb[Dimension.GSI.value].agg(list).to_dict()


def plot_partition(
    axis: Axes,
    data: list[list[float]],
    positions: list[float],
    widths: float,
    color: str = "black",
):
    """
    Convenience function to plot a data partition as a boxplot on the given axis.
    """
    axis.boxplot(
        data,
        positions=positions,
        widths=widths,
        patch_artist=True,
        boxprops={
            "facecolor": "white",
            "color": color,
        },
        medianprops={
            "color": color,
        },
        whiskerprops={
            "color": color,
        },
        capprops={
            "color": color,
        },
        flierprops={
            "marker": "o",
            "markersize": 2,
            "color": color,
            "markerfacecolor": color,
        },
    )


def plot_farm_vs_wild_gsi(widths: float = 0.3):
    """
    Render boxplot figure for farmed and wild gonadosomatic index over time.

    Farm means net or cage, but not wild.
    """
    groups = load_and_subset_source_data()
    fig, ax = subplots(figsize=(6, 4))
    positions = [1.0, 2.0, 3.0, 4.0]
    plot_partition(
        axis=ax,
        data=[
            groups.get(("C", 7), []) + groups.get(("N", 7), []),
            groups.get(("C", 8), []) + groups.get(("N", 8), []),
            groups.get(("C", 9), []) + groups.get(("N", 9), []),
            groups.get(("C", 10), []) + groups.get(("N", 10), []),
        ],
        positions=positions,
        widths=widths,
        color="black",
    )
    plot_partition(
        axis=ax,
        data=[
            groups.get(("W", 7), []),
            groups.get(("W", 8), []),
            groups.get(("W", 9), []),
            groups.get(("W", 10), []),
        ],
        positions=[x + widths for x in positions],
        widths=widths,
        color="red",
    )
    xticks(
        [x + widths / 2 for x in positions], ["July", "August", "September", "October"]
    )
    ylabel("gonadosomatic index")
    legend(
        handles=[
            mpatches.Patch(color="black", label="Farm"),
            mpatches.Patch(color="red", label="Wild"),
        ]
    )
    fig.savefig(figures / "GSI_farm_vs_wild_boxplot.png")


def plot_gonadosomatic_index(widths: float = 0.2):
    """
    Render a boxplot figure of GSI by gear type and month.
    """
    groups = load_and_subset_source_data()
    fig, ax = subplots(figsize=(6, 4))
    positions = [1.0, 2.0, 3.0, 4.0]
    plot_partition(
        axis=ax,
        data=[
            groups.get(("N", 7), []),
            groups.get(("N", 8), []),
            groups.get(("N", 9), []),
            groups.get(("N", 10), []),
        ],
        positions=positions,
        widths=widths,
        color="black",
    )
    plot_partition(
        axis=ax,
        data=[
            groups.get(("C", 7), []),
            groups.get(("C", 8), []),
            groups.get(("C", 9), []),
            groups.get(("C", 10), []),
        ],
        positions=[x + widths for x in positions],
        widths=widths,
        color="blue",
    )
    plot_partition(
        axis=ax,
        data=[
            groups.get(("W", 7), []),
            groups.get(("W", 8), []),
            groups.get(("W", 9), []),
            groups.get(("W", 10), []),
        ],
        positions=[x + 2 * widths for x in positions],
        widths=widths,
        color="red",
    )

    xticks([x + widths for x in positions], ["July", "August", "September", "October"])
    ylabel("gonadosomatic index")
    legend(
        handles=[
            mpatches.Patch(color="black", label="Net"),
            mpatches.Patch(color="blue", label="Cage"),
            mpatches.Patch(color="red", label="Wild"),
        ]
    )
    fig.savefig(figures / "GSI_gear_boxplot.png")


if __name__ == "__main__":
    plot_farm_vs_wild_gsi()
    plot_gonadosomatic_index()
