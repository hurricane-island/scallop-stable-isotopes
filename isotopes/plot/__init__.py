"""
Plotting commands. Isolating these from basic stats, and from
commands that focus on output tables or summaries.
"""
from enum import Enum
from itertools import cycle, count
from calendar import month_name
from typing import Union
from pandas import read_csv, Series
from matplotlib.pyplot import subplots
from matplotlib.axes import Axes
from matplotlib.patches import Patch
from seaborn import scatterplot, pairplot
from click import option, group, Choice, argument
from numpy import column_stack, vstack
from scipy.spatial import ConvexHull
from isotopes.options import (
    Dimension,
    TissueType,
    CultureMethod,
    tissue_type_option,
    figure_size,
    isotopes_no_outliers,
    bad_run_dates,
    figures,
    custom_colors,
    raw_data,
    GSIDimension
)
from isotopes.statistics import calculate_famd, partition_data_by_tissue

class PlotCommand(Enum):
    """
    Valid plot commands for the isotopes package.
    """
    LIPID_EXTRACTION = "lipids"
    PAIRS = "pairs"
    SCATTER_MONTHLY_GEAR = "monthly"
    SCATTER_GEAR_MONTHLY = "scatter-gear-monthly"
    TISSUE_BY_MONTH = "tissue"
    MOLAR_RATIO_BY_TISSUE = "ratio"

@group()
def plot():
    """
    Generate plots for the stable isotope data.
    """

@group()
def tissue():
    """
    Generate plots for the stable isotope data, specifically for tissue samples.
    """

@group()
def box():
    """
    Generate boxplots for the stable isotope data.
    """

@group()
def scatter():
    """
    Generate scatter plots for the stable isotope data.
    """

plot.add_command(tissue)
tissue.add_command(box)
tissue.add_command(scatter)


@scatter.command(PlotCommand.LIPID_EXTRACTION.value)
@tissue_type_option(TissueType.MUSCLE)
@option(
    "--culture-method",
    type=Choice(CultureMethod, case_sensitive=False),
    default=CultureMethod.WILD.name,
    help="Culture method to filter the data by (default: wild).",
)
@option(
    "--compare",
    type=Choice([Dimension.CARBON_FRACTIONATION, Dimension.NITROGEN_FRACTIONATION], case_sensitive=False),
    default=Dimension.CARBON_FRACTIONATION.name,
    help="Dimension to compare against molar ratio (default: carbon_fractionation).",
)
@figure_size((10, 8))
def plot_scatter_lipid_extraction(
    tissue_type: TissueType,
    culture_method: CultureMethod,
    compare: Dimension,
    figsize: tuple[float, float]
):
    """
    Determine if lipid extraction has an effect on the carbon fractionation
    of wild scallops.
    """
    filter_dims = [
        Dimension.TISSUE.value,
        Dimension.GEAR.value,
        Dimension.DATE_RUN.value,
    ]
    df = read_csv(
        isotopes_no_outliers,
        header=0,
        usecols=[
            *filter_dims,
            compare.value,
            Dimension.MOLAR_RATIO.value,
        ],
    )
    mask = (
        (~df[Dimension.DATE_RUN.value].isin(bad_run_dates))
        & (df[Dimension.TISSUE.value] == tissue_type.value)
        & (df[Dimension.GEAR.value] == culture_method.value)
    )
    df = (
        df[mask]
        .drop(columns=filter_dims)
        .dropna()
    )
    fig, ax = subplots(figsize=figsize)
    ax.scatter(
        x=df[Dimension.MOLAR_RATIO.value],
        y=df[compare.value],
        s=30,
    )
    fig.savefig(f"{figures}/lipid_extraction_{culture_method.name.lower()}_{tissue_type.name.lower()}_{compare.name.lower()}.png")


@tissue.command(PlotCommand.PAIRS.value)
def plot_pairs_seaborn():
    """
    Pairplot of nitrogen fractionation, carbon fractionation, and molar ratio, colored by gear type.
    """
    df = read_csv(
        isotopes_no_outliers,
        header=0,
        usecols=[
            Dimension.TISSUE.value,
            Dimension.DATE_RUN.value,
            Dimension.NITROGEN_FRACTIONATION.value,
            Dimension.CARBON_FRACTIONATION.value,
            Dimension.MOLAR_RATIO.value,
            Dimension.GEAR.value,
        ],
    )
    # Remove known bad samples, and select only one tissue type for analysis
    mask = (~df[Dimension.DATE_RUN.value].isin(bad_run_dates)) & (
        df[Dimension.TISSUE.value].isin({"M", "G"})
    )
    df[Dimension.GEAR.value] = df[Dimension.GEAR.value].map(
        {"C": "Cage", "N": "Net", "W": "Wild"}
    )
    # Tissue type is already filtered, so only need to check gear and collection date
    df = (
        df[mask]
        .drop(columns=[Dimension.DATE_RUN.value, Dimension.TISSUE.value])
        .dropna()
    )
    pairplot(
        df[
            [
                Dimension.NITROGEN_FRACTIONATION.value,
                Dimension.CARBON_FRACTIONATION.value,
                Dimension.MOLAR_RATIO.value,
                Dimension.GEAR.value,
            ]
        ],
        markers="x",
        hue=Dimension.GEAR.value,
        palette=custom_colors,
    ).savefig(figures / "new-pairplot-all-tissue-gear.png")


@tissue.command(PlotCommand.SCATTER_MONTHLY_GEAR.value)
@tissue_type_option(TissueType.MUSCLE)
@figure_size((7.5, 3))
def plot_scatter_monthly_gear(
    tissue_type: TissueType, 
    figsize: tuple[float, float]
):
    """
    Scatter plot of molar ratio vs carbon fractionation, separated
    into subplots for each month.
    """
    df = partition_data_by_tissue(
        isotopes_no_outliers,
        [
            Dimension.CARBON_FRACTIONATION,
            Dimension.MOLAR_RATIO,
            Dimension.GEAR,
            Dimension.COLLECTION_DATE,
        ],
        tissue_type,
    )
    df[Dimension.GEAR.value] = df[Dimension.GEAR.value].map(
        {"C": "Cage", "N": "Net", "W": "Wild"}
    )
    color_map = {"Cage": "tab:cyan", "Net": "tab:blue", "Wild": "tab:pink"}
    # preserve groups as numerical so that they are sorted
    groups = (
        df.dropna()
        .groupby(Dimension.COLLECTION_DATE.value)
    )

    context = subplots(1, len(groups), figsize=figsize, sharex=True, sharey=False)
    fig = context[0]
    ax: list[Axes] = context[1]  # force type hinting
    for ind, (month, data) in enumerate(groups):
        colors = data[Dimension.GEAR.value].map(color_map)
        ax[ind].scatter(
            data[Dimension.MOLAR_RATIO.value],
            data[Dimension.CARBON_FRACTIONATION.value],
            c=colors,
            marker="x",
        )
        ax[ind].set_title(month_name[int(month)])  # type: ignore
        ax[ind].set_xlim(3, 6)
        ax[ind].set_ylim(-19, -16)
        ax[ind].set_xlabel(Dimension.MOLAR_RATIO.value)
        if ind > 0:
            ax[ind].set_yticks([])
        else:
            ax[ind].set_ylabel(Dimension.CARBON_FRACTIONATION.value)

    handles = [Patch(color=color, label=label) for label, color in color_map.items()]
    fig.legend(handles=handles, loc="upper right")
    fig.savefig(figures / "rawdata_scatter_monthly_gear.png")


@tissue.command(PlotCommand.SCATTER_GEAR_MONTHLY.value)
@figure_size((7.5, 5))
@tissue_type_option(TissueType.MUSCLE)
def plot_scatter_gear_monthly(figsize: tuple[float, float], tissue_type: TissueType):
    """
    Carbon fractionation vs molar ratio for a tissue type, segemented
    by culture method and collection date (month). Plotting uses
    seaborn. For greater control, modify the code to use matplotlib directly.
    """
    df = partition_data_by_tissue(
        isotopes_no_outliers,
        [
            Dimension.CARBON_FRACTIONATION,
            Dimension.MOLAR_RATIO,
            Dimension.GEAR,
            Dimension.COLLECTION_DATE,
        ],
        tissue_type,
    )

    # Tissue type is already filtered, so only need to check gear and collection date
    df = df.dropna()
    fig, ax = subplots(figsize=figsize)
    scatterplot(
        x=df[Dimension.MOLAR_RATIO.value],
        y=df[Dimension.CARBON_FRACTIONATION.value],
        hue=df[Dimension.GEAR.value],
        palette="tab10",
        style=df[Dimension.COLLECTION_DATE.value],
        legend="auto",
        s=150,
        ax=ax,
    )
    ax.set_xlabel(Dimension.MOLAR_RATIO.value)
    ax.set_ylabel(Dimension.CARBON_FRACTIONATION.value)
    fig.savefig(figures / "rawdata_scatter_gear_monthly.png")


def plot_partition(
    axis: Axes,
    data: list[Union[list[float], Series]],
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


# pylint: disable=too-many-locals
@box.command("var")
@tissue_type_option(TissueType.MUSCLE)
@figure_size((4,3))
@argument(
    "dim",
    type=Choice(
        [
            Dimension.NITROGEN_FRACTIONATION,
            Dimension.CARBON_FRACTIONATION,
            Dimension.MOLAR_RATIO,
            Dimension.NITROGEN_PERCENTAGE
        ],
        case_sensitive=False,
    ),
)
def isotopes_plot_box_var(
    tissue_type: TissueType,
    figsize: tuple[float, float],
    dim: Dimension
):
    """
    Plot single isotope variable by month and culture method.
    """
    df = partition_data_by_tissue(
        isotopes_no_outliers,
        [
            # Grouping and filtering options
            Dimension.COLLECTION_DATE,
            Dimension.GEAR,
            # Analysis options
            Dimension.NITROGEN_PERCENTAGE,
            Dimension.NITROGEN_FRACTIONATION,
            Dimension.CARBON_FRACTIONATION,
            Dimension.MOLAR_RATIO,
        ],
        tissue_type,
    ).dropna()
    lookup = {
        "N": ("Net", "black"),
        "C": ("Cage", "blue"),
        "W": ("Wild", "red")
    }
    df[Dimension.GEAR.value] = df[Dimension.GEAR.value].map({
        key: name for key, (name, _) in lookup.items()
    })
    groups: dict[tuple[float, str], list[float]] = (
        df.groupby(
            [
                Dimension.COLLECTION_DATE.value,
                Dimension.GEAR.value,
            ]
        )
        .agg(list)
        .to_dict()[dim.value]
    )
    fig, ax = subplots(figsize=figsize)
    month_ind = sorted(month for month, _ in groups.keys())
    positions = [(month - min(month_ind) + 1) for month in month_ind]
    labels = [month_name[int(ii)] for ii in month_ind]
    widths = 0.2
    for ii, (gear, color) in enumerate(lookup.values()):
        ax.boxplot(
            [groups[(month, gear)] for month in month_ind],
            positions=[x + widths * ii for x in positions],
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
            label=gear
        )
    ax.set_xticks([x + widths for x in positions], labels=labels)
    ax.set_ylabel(dim.value)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / f"isotopes_plot_box_var_{dim.name.lower()}.png")

@box.command("gsi")
@figure_size((6, 7))
def isotopes_plot_box_gsi(figsize: tuple[float, float]):
    """
    Render multiple plots that have boxplots of d13C, d15N, and C/N (molar) by month
    separated by gear type (net, cage, wild).
    """
    fig, ax = subplots(figsize=figsize, squeeze=True)
    positions = [1.0, 2.0, 3.0, 4.0, 5.0]
    widths = 0.2
    groups: dict[tuple[str, int], list[float]] = (
        read_csv(
            raw_data,
            header=0,
            usecols=[
                GSIDimension.COLLECTION_DATE.value,
                GSIDimension.GEAR.value,
                GSIDimension.GSI.value,
            ],
        )
        .dropna()
        .groupby([GSIDimension.GEAR.value, GSIDimension.COLLECTION_DATE.value])
        .agg(list)
        .to_dict()[GSIDimension.GSI.value]
    )
    gear = ["N", "C", "W"]

    for ii, (gg, color) in enumerate(zip(gear, custom_colors)):
        data = []
        for month in range(6, 11):
            gr = groups.get((gg, month), [])
            data.append(gr)
        plot_partition(
            axis=ax,
            data=data,
            positions=[x + 0.2 * ii for x in positions],
            widths=widths,
            color=color,
        )
    ax.set_xticks([x + widths for x in positions], [])
    ax.set_ylabel("GSI")
    ax.set_xticklabels(["June", "July", "August", "September", "October"])
    fig.legend(
        handles=[
            Patch(color="black", label="Net"),
            Patch(color="blue", label="Cage"),
            Patch(color="red", label="Wild"),
        ]
    )
    fig.savefig(figures / "isotopes_plot_box_gsi.png")


@tissue.command("histograms")
@tissue_type_option(TissueType.MUSCLE)
@figure_size((8, 6))
def plot_tissue_histograms(figsize: tuple[float, float], tissue_type: TissueType):
    """
    Visually determine whether data appears normally distributed for a given tissue type.
    This is intended as a simple diagnostic, and should be followed up with a more rigorous statistical test for normality.
    """
    dims = [
        Dimension.NITROGEN_FRACTIONATION,
        Dimension.CARBON_FRACTIONATION,
        Dimension.MOLAR_RATIO,]
    df = partition_data_by_tissue(
        isotopes_no_outliers,
        dims,
        tissue_type,
    ).dropna()
    fig, ax = subplots(figsize=figsize)
    for dim, color in zip(
        (each.value for each in dims),
        ("black", "blue", "red"),
    ):
        series: Series = df[dim]
        series.hist(
            ax=ax,
            label=dim,
            color=color,
        )
    ax.legend()
    ax.grid(False)
    fig.savefig(f"{figures}/tissue_histograms_{tissue_type.name.lower()}.png")


@scatter.command("famd")
@figure_size((10, 8))
@tissue_type_option(TissueType.MUSCLE)
@option(
    "--convex-hulls",
    is_flag=True,
    help="Whether to draw convex hulls around the data points for each month.",
    default=False,
)
def plot_famd_analysis_2d(
    tissue_type: TissueType,
    figsize: tuple[float, float],
    convex_hulls: bool,
):
    """
    Perform Factor Analysis of Mixed Data (FAMD) on the muscle tissue data and generate plots.
    """
    df = partition_data_by_tissue(
        isotopes_no_outliers,
        [
            Dimension.NITROGEN_FRACTIONATION,
            Dimension.CARBON_FRACTIONATION,
            Dimension.MOLAR_RATIO,
            Dimension.COLLECTION_DATE,
            Dimension.GEAR,
        ],
        tissue_type,
    ).dropna()
    df[Dimension.COLLECTION_DATE.value] = df[Dimension.COLLECTION_DATE.value].map(dict((i, month_name[i]) for i in range(1, 13)))
    factors = calculate_famd(df, 2)
    inner_join = df.join(factors, how="inner")
    gear_groups = inner_join.groupby(Dimension.GEAR.value)
    colors = (("red", "o"), ("blue", "D"), ("black", "s"))
    fig, ax = subplots(figsize=figsize)
    for (gear_type, group_df), (color, marker) in zip(gear_groups, colors):
        print("group", group_df.columns)
        ax.scatter(
            group_df[0],
            group_df[1],
            label=gear_type,
            color=color,
            marker=marker,
            s=30,
            zorder=2,
        )

    if convex_hulls:
        month_groups = inner_join.groupby(Dimension.COLLECTION_DATE.value)
        linestyles = ["--", "-", "-."]
        style_count = len(linestyles)
        for (month, group_df), linestyle, hull_count in zip(
            month_groups, cycle(linestyles), count()
        ):
            cycle_num = hull_count // style_count
            points = column_stack((group_df[0], group_df[1]))
            if len(points) >= 3:
                hull = ConvexHull(points)
                hull_points = points[hull.vertices]
                hull_points = vstack([hull_points, hull_points[0]])  # close polygon
                ax.plot(
                    hull_points[:, 0],
                    hull_points[:, 1],
                    lw=cycle_num + 1,  # increase line width for each cycle
                    linestyle=linestyle,
                    color="black",
                    zorder=1,
                    label=month,
                )
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_xlabel("Factor 1")
    ax.set_ylabel("Factor 2")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(f"{figures}/factor_analysis_of_mixed_data.png")
