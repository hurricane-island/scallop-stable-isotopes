# pylint: disable=too-many-lines,redefined-builtin
"""
This module contains methods for statistical analysis of stable isotope data,
especially Principal Component Analysis (PCA) using the `sklearn` and `scipy`
libraries, and Factor Analysis of Mixed Data (FAMD) using the `prince` library.

scipy.spatial is in C, and needs to be dynamically loaded by pylint in order
to avoid a false positive error
"""

from pathlib import Path
from calendar import month_name
from typing import Dict, Sequence, Union
from itertools import cycle, count
from enum import Enum
from pandas import DataFrame, read_csv, to_datetime, set_option, Series
from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols
from numpy import arange, sqrt, column_stack, vstack, abs
from matplotlib.pyplot import subplots, savefig
import matplotlib.collections
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Patch
from scipy.stats import levene, zscore
from scipy.spatial import ConvexHull
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
from prince import FAMD
from seaborn import scatterplot, pairplot
from click import option, group, Choice


bad_run_dates = {"9/6/23"}  # use Set() as more generic lookup than single value
figures = Path(__file__).parent / "../figures"
data_dir = Path(__file__).parent / "../data"
raw_data = data_dir / "gonadosomatic-index.csv"
isotopes_no_outliers = data_dir / "stable-isotopes-no-outliers.csv"
isotopes_raw = data_dir / "stable-isotopes.csv"
env_data = data_dir / "temperature-and-light.csv"
custom_colors = ("black", "blue", "red")


@group()
def summarize():
    """
    Summarize data and perform statistical analysis.
    """

@group()
def plot():
    """
    Generate plots for the stable isotope data.
    """

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

def file_output_options(cmd):
    """
    Decorator to add common file output options to a Click command.
    """
    cmd = option(
        "--encoding",
        default="png",
        help="Encoding format for the output table (e.g., 'png', 'pdf').",
    )(cmd)
    cmd = figure_size((10, 7))(cmd)
    cmd = option(
        "--fontsize",
        default=10,
        help="Font size for the table text.",
    )(cmd)
    return cmd

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


def load_temperature_data(filepath: Path) -> DataFrame:
    """
    Load environmental data from the CSV file.
    """
    df = read_csv(
        filepath,
        header=0,
        usecols=[
            EnvDimension.DATE.value,
            EnvDimension.CAGE_TEMP.value,
            EnvDimension.NET_BOTTOM_TEMP.value,
            EnvDimension.WILD_TEMP.value,
        ],
    )
    times = to_datetime(df[EnvDimension.DATE.value], format="%m/%d/%y %H:%M")
    df[EnvDimension.DATE.value] = times
    df["Month"] = times.dt.month.map(dict((i, month_name[i]) for i in range(1, 13)))
    return df


def load_and_subset_source_data(filepath: Path) -> dict[tuple[str, int], list[float]]:
    """
    Data needed for GSI boxplots. Converts from dataframe, to a dictionary grouped by
    tuples of gear and month that can be combined as needed for plotting.
    """
    df = read_csv(
        filepath,
        header=0,
        usecols=[
            GSIDimension.COLLECTION_DATE.value,
            GSIDimension.GEAR.value,
            GSIDimension.SEX.value,
            GSIDimension.GSI.value,
        ],
    ).dropna(
        subset=[
            GSIDimension.GSI.value,
            GSIDimension.COLLECTION_DATE.value,
            GSIDimension.GEAR.value,
        ]
    )
    groups: dict[tuple[str, int], list[float]] = (
        df.groupby([GSIDimension.GEAR.value, GSIDimension.COLLECTION_DATE.value])
        .agg(list)
        .to_dict()[GSIDimension.GSI.value]
    )
    return groups


def load_tissue_data(filepath: Path) -> DataFrame:
    """Same as scatter data loader"""
    df = read_csv(
        filepath,
        header=0,
        usecols=[
            Dimension.COLLECTION_DATE.value,
            Dimension.GEAR.value,
            Dimension.SEX.value,
            Dimension.TISSUE.value,
            "Mass (mg)",  # possible PCA input
            Dimension.NITROGEN_PERCENTAGE.value,
            "N (umoles)",  # possible PCA input if N limited?
            Dimension.NITROGEN_FRACTIONATION.value,
            Dimension.CARBON_PERCENTAGE.value,
            "C (umoles)",  # possible PCA input if N limited?
            Dimension.CARBON_FRACTIONATION.value,
            Dimension.MOLAR_RATIO.value,
            Dimension.DATE_RUN.value,
        ],
    ).dropna(
        subset=[
            Dimension.CARBON_FRACTIONATION.value,
            Dimension.NITROGEN_FRACTIONATION.value,
            Dimension.MOLAR_RATIO.value,
            Dimension.COLLECTION_DATE.value,
            Dimension.GEAR.value,
            Dimension.TISSUE.value,
        ]
    )
    bad_values_mask = df[Dimension.DATE_RUN.value].isin(bad_run_dates)
    df = df.drop(df[bad_values_mask].index).drop(columns=[Dimension.DATE_RUN.value])
    df["Farmed or Wild"] = df[Dimension.GEAR.value].map({"C": "F", "N": "F", "W": "W"})
    df = df.dropna(subset=[Dimension.TISSUE.value])
    return df[
        df[Dimension.TISSUE.value].isin(["M", "G"])
    ]  # Only muscle and gonad tissue


def load_scatter_data(filepath: Path) -> DataFrame:
    """Load scatter plot data, filtering out unwanted tissues and contaminated samples."""
    df = read_csv(
        filepath,
        header=0,
        usecols=[
            Dimension.COLLECTION_DATE.value,
            Dimension.GEAR.value,
            Dimension.SEX.value,
            Dimension.TISSUE.value,
            "Mass (mg)",  # possible PCA input
            Dimension.NITROGEN_PERCENTAGE.value,
            "N (umoles)",  # possible PCA input if N limited?
            Dimension.NITROGEN_FRACTIONATION.value,
            Dimension.CARBON_PERCENTAGE.value,
            "C (umoles)",  # possible PCA input if N limited?
            Dimension.CARBON_FRACTIONATION.value,
            Dimension.MOLAR_RATIO.value,
            Dimension.DATE_RUN.value,
        ],
    )

    # Remove known contaminated samples
    mask = df[Dimension.DATE_RUN.value].isin(bad_run_dates)
    df = df.drop(df[mask].index)
    # Don't need the date run for analysis, only pre-filtering
    df = df.drop(columns=[Dimension.DATE_RUN.value]).dropna(
        subset=[Dimension.GEAR.value, Dimension.TISSUE.value]
    )
    return df[
        df[Dimension.TISSUE.value].isin(["M", "G"])
    ]  # Only muscle and gonad tissue


def quantize_categorical_column(
    df: DataFrame, column_name: str, categories: Dict[str, int]
) -> DataFrame:
    """
    Replace strings with integers in a categorical column of a DataFrame,
    using a provided mapping.

    Rows with values not in the mapping are replaced with 0 and then
    removed from the DataFrame.

    This process changes the original dataframe.
    """
    set_option("future.no_silent_downcasting", True)  # suppress runtime warning
    for value in df[column_name]:
        replace = categories.get(value, 0)
        df[column_name] = df[column_name].replace(value, replace)
    non_zero_mask = df[column_name] != 0
    return df[non_zero_mask]


def return_column_to_categorical(
    df: DataFrame, column_name: str, categories: Dict[int, str]
) -> DataFrame:
    """
    Replace integers with strings to return a categorical column of a DataFrame.

    This is for the purpose of making plots more readable,
    and should be used AFTER statistical analysis.

    This process changes the original dataframe.
    """
    for value in df[column_name]:
        replace = categories.get(value, 0)
        df[column_name] = df[column_name].replace(value, replace)
    return df

@plot.command("tissue-histograms")
@tissue_type_option(TissueType.MUSCLE)
@figure_size((8, 6))
def plot_tissue_histograms(
    figsize: tuple[float, float],
    tissue_type: TissueType
):
    """
    Visually determine whether data appears normally distributed for a given tissue type.
    This is intended as a simple diagnostic, and should be followed up with a more rigorous statistical test for normality.
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
        ],
    )
    # Remove known bad samples, and select only one tissue type for analysis
    mask = (~df[Dimension.DATE_RUN.value].isin(bad_run_dates)) & \
        (df[Dimension.TISSUE.value] == tissue_type.value)

    # Tissue type is already filtered, so only need to check gear and collection date
    df = df[mask] \
        .drop(columns=[Dimension.DATE_RUN.value, Dimension.TISSUE.value]) \
        .dropna()
    
    fig, ax = subplots(figsize=figsize)
    for dim, color in zip((
        Dimension.NITROGEN_FRACTIONATION.value,
        Dimension.CARBON_FRACTIONATION.value,
        Dimension.MOLAR_RATIO.value,
    ), ("black", "blue", "red")):
        series: Series = df[dim]
        series.hist(
            ax=ax,
            label=dim,
            color=color,
        )
    ax.legend()
    ax.grid(False)
    fig.savefig(f"{figures}/tissue_histograms_{tissue_type.name.lower()}.png")

@summarize.command("anova")
@tissue_type_option(TissueType.MUSCLE)
def summarize_analysis_of_variance(
    tissue_type: TissueType
):
    """
    Summarize Analysis of Variance (ANOVA) for a given tissue type. This is only valid
    if the assumptions of ANOVA are met, which can be checked using Levene's test for 
    homogeneity of variances.

    Can only be run for a single variable at a time.

    Because tissue types have an effect, it only makes sense to run this on a single
    tissue partition. Generally this will be muscle.
    """
    analysis = Dimension.NITROGEN_PERCENTAGE
    df = read_csv(
            isotopes_no_outliers,
            header=0,
            usecols=[
                Dimension.COLLECTION_DATE.value,
                Dimension.GEAR.value,
                Dimension.TISSUE.value,
                Dimension.DATE_RUN.value,
                analysis.value
            ],
        )
    
    # Remove known bad samples, and select only one tissue type for analysis
    mask = (~df[Dimension.DATE_RUN.value].isin(bad_run_dates)) & \
        (df[Dimension.TISSUE.value] == tissue_type.value)
    # Tissue type is already filtered, so only need to check gear and collection date
    df = df[mask].drop(columns=[Dimension.DATE_RUN.value, Dimension.TISSUE.value]).dropna()

    # Patsy syntax: C() = categorical, Q() = wrap special characters
    formula = (
        f"Q('{analysis.value}') ~ "
        f"C(Q('{Dimension.GEAR.value}')) + "
        f"C(Q('{Dimension.COLLECTION_DATE.value}')) + "
        f"C(Q('{Dimension.GEAR.value}')):C(Q('{Dimension.COLLECTION_DATE.value}'))"
    )
    model = ols(formula, data=df).fit()
    result = anova_lm(model, type=2)  # Type II sum of squares

    print((
        f"Analysis of Variance\n"
        f"Tissue: {tissue_type.name.lower()}\n"
        f"Variable: {analysis.value}\n"
    ))
    print(result)


@summarize.command("levenes")
@tissue_type_option(TissueType.MUSCLE)
@option(
    "--group-by",
    required=True,
    help="Dimension to group by.",
    type=Choice([Dimension.GEAR, Dimension.COLLECTION_DATE], case_sensitive=False),
)
@option(
    "--variable",
    required=True,
    help="Dimension to test for homogeneity of variances.",
    type=Choice([
        Dimension.NITROGEN_FRACTIONATION,
        Dimension.CARBON_FRACTIONATION,
        Dimension.MOLAR_RATIO
    ], case_sensitive=False)
)
def summarize_levenes_test(
    tissue_type: TissueType,
    group_by: Dimension,
    variable: Dimension
):
    """
    Summarize Levene's test for homogeneity of variances. This is used to determine whether the assumptions of ANOVA are met for a given tissue type. If the p-value is greater than 0.05, we can assume homogeneity of variances and proceed with ANOVA.
    """
    df = read_csv(
        isotopes_no_outliers,
        header=0,
        usecols=[
            group_by.value,
            Dimension.TISSUE.value,
            Dimension.DATE_RUN.value,
            variable.value
        ],
    )
    # Remove known bad samples, and select only one tissue type for analysis
    mask = (~df[Dimension.DATE_RUN.value].isin(bad_run_dates)) & \
        (df[Dimension.TISSUE.value] == tissue_type.value)

    # Tissue type is already filtered, so only need to check gear and collection date
    groupby_result = df[mask] \
        .drop(columns=[Dimension.DATE_RUN.value, Dimension.TISSUE.value]) \
        .dropna() \
        .groupby(group_by.value)

    # Test expects bins, not necessarily equal sized
    groups = [each for _, each in groupby_result[variable.value]]
    result = levene(*groups)

    print((
        f"\nLevene's Test of Homogeneity of Variance"
        f"\nTissue: {tissue_type.name.lower()}"
        f"\nDimension: {variable.name.lower()} ({variable.value})"
        f"\nGroup by: {group_by.name}"
        f"\nResult: {result.statistic}"
        f"\nP-value: {result.pvalue}"
        f"\nHomogenous: {result.pvalue > 0.05}"
    ))


@plot.command("pca")
def plot_pca_analysis():
    """
    When ANOVA assumptions are not met, we can still use Principle Components Analysis. Perform PCA analysis on the muscle tissue data and generate plots.
    """
    # only scallops and filters are being plotted
    df = load_scatter_data(isotopes_no_outliers)
    data_muscle = df[df[Dimension.TISSUE.value] == "M"]

    # Since ANOVA assumptions are not met, try PCA
    # Ensure data are quantized properly
    data_muscle = quantize_categorical_column(
        data_muscle, Dimension.GEAR.value, {"C": 1, "N": 2, "W": 3}
    )
    data_muscle = quantize_categorical_column(
        data_muscle, Dimension.SEX.value, {"F": 1, "M": 2}
    )
    data_muscle = quantize_categorical_column(
        data_muscle, Dimension.TISSUE.value, {"G": 1, "M": 2}
    )

    pca_df = data_muscle[
        [
            Dimension.NITROGEN_FRACTIONATION.value,
            Dimension.CARBON_FRACTIONATION.value,
            Dimension.MOLAR_RATIO.value,
        ]
    ]

    std_scaler = StandardScaler()
    scaled_df = std_scaler.fit_transform(pca_df)
    pca = PCA(n_components=2)

    components = pca.fit_transform(scaled_df)
    explained_variance = pca.explained_variance_ratio_

    # Make a scree plot to visualize the proportion of variance
    # explained by each principal component
    fig, ax = subplots()
    pc_numbers = arange(len(explained_variance)) + 1
    ax.plot(pc_numbers, explained_variance, marker="o", linestyle="-")
    ax.set_title("Scree Plot")
    ax.set_xlabel("Principal Component Number")
    ax.set_ylabel("Proportion of Explained Variance")
    ax.grid(True)
    fig.savefig(f"{figures}/pca_scree_plot.png")

    fig, ax = subplots(figsize=(10, 3))
    ax.axis("off")

def calculate_famd(
    partition: DataFrame,
    components: int
):
    """
    Perform Factor Analysis of Mixed Data (FAMD)
    """
    famd_data = partition[
        [
            Dimension.CARBON_FRACTIONATION.value,
            Dimension.NITROGEN_FRACTIONATION.value,
            Dimension.MOLAR_RATIO.value,
            Dimension.COLLECTION_DATE.value,
        ]
    ]
    famd = FAMD(
        n_components=components,
        n_iter=3,
        copy=True,
        check_input=True,
        random_state=None,
        engine="sklearn",
        handle_unknown="error",
    ).fit(famd_data)
    print(famd.eigenvalues_summary)
    print(famd.column_contributions_)
    return famd.row_coordinates(famd_data)

@plot.command("famd")
@figure_size((10, 8))
@option(
    "--data-file",
    default="stable-isotopes-no-outliers.csv",
    help="CSV file containing the stable isotope data.",
)
@option(
    "--convex-hulls",
    is_flag=True,
    help="Whether to draw convex hulls around the data points for each month.",
    default=False,
)
def plot_famd_analysis_2d(
    figsize: tuple[float, float],
    data_file: str,
    convex_hulls: bool,
):
    """
    Perform Factor Analysis of Mixed Data (FAMD) on the muscle tissue data and generate plots.
    """
    df = return_column_to_categorical(
        load_scatter_data(data_dir / data_file),
        Dimension.COLLECTION_DATE.value,
        dict((i, month_name[i]) for i in range(1, 13)),
    )
    partition = df[df[Dimension.TISSUE.value] == "M"]
    factors = calculate_famd(partition, 2)
    inner_join = partition.join(factors, how="inner")
    gear_groups = inner_join.groupby(Dimension.GEAR.value)
    custom_colors = (("red", "o"), ("blue", "D"), ("black", "s"))
    fig, ax = subplots(figsize=figsize)
    for (gear_type, group_df), (color, marker) in zip(gear_groups, custom_colors):
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
        for (month, group_df), linestyle, hull_count in zip(month_groups, cycle(linestyles), count()):
            cycle_num = hull_count // style_count
            points = column_stack(
                (group_df[0], group_df[1])
            )
            if len(points) < 3:
                continue
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


@plot.command("clustering")
def clustering_and_pca_analysis():
    """
    Perform clustering and PCA analysis on the muscle tissue data.
    """
    # only scallops and filters are being plotted
    df = load_scatter_data(isotopes_no_outliers)
    data_muscle = df[df[Dimension.TISSUE.value] == "M"]


    # Since ANOVA assumptions are not met, try PCA
    # Ensure data are quantized properly

    df = quantize_categorical_column(df, Dimension.GEAR.value, {"C": 1, "N": 2, "W": 3})
    df = quantize_categorical_column(df, Dimension.SEX.value, {"F": 1, "M": 2})
    df = quantize_categorical_column(df, Dimension.TISSUE.value, {"G": 1, "M": 2})
    data_muscle = quantize_categorical_column(
        data_muscle, Dimension.GEAR.value, {"C": 1, "N": 2, "W": 3}
    )
    data_muscle = quantize_categorical_column(
        data_muscle, Dimension.SEX.value, {"F": 1, "M": 2}
    )
    data_muscle = quantize_categorical_column(
        data_muscle, Dimension.TISSUE.value, {"G": 1, "M": 2}
    )

    pca_df = data_muscle[
        [
            Dimension.NITROGEN_FRACTIONATION.value,
            Dimension.CARBON_FRACTIONATION.value,
            Dimension.MOLAR_RATIO.value,
        ]
    ]

    std_scaler = StandardScaler()
    scaled_df = std_scaler.fit_transform(pca_df)
    pca = PCA(n_components=2)

    components = pca.fit_transform(scaled_df)
    explained_variance = pca.explained_variance_ratio_


    # Factor Analysis of Mixed Data
    # Variables used: CARBON_FRACTIONATION, NITROGEN_FRACTIONATION, MOLAR_RATIO, COLLECTION_DATE
    data_muscle = return_column_to_categorical(
        data_muscle,
        Dimension.COLLECTION_DATE.value,
        {6: "June", 7: "July", 8: "August", 9: "September", 10: "October"},
    )
    data_muscle = return_column_to_categorical(
        data_muscle, Dimension.GEAR.value, {1: "Farm", 2: "Farm", 3: "Wild"}
    )
    df = return_column_to_categorical(
        df,
        Dimension.COLLECTION_DATE.value,
        {6: "June", 7: "July", 8: "August", 9: "September", 10: "October"},
    )
    df = return_column_to_categorical(
        df, Dimension.GEAR.value, {1: "Farm", 2: "Farm", 3: "Wild"}
    )

    

    df = quantize_categorical_column(
        df,
        Dimension.COLLECTION_DATE.value,
        {"June": 6, "July": 7, "August": 8, "September": 9, "October": 10},
    )
    data_muscle = quantize_categorical_column(
        data_muscle,
        Dimension.COLLECTION_DATE.value,
        {"June": 6, "July": 7, "August": 8, "September": 9, "October": 10},
    )



    # Make a table to summarize the PCA results
    summary = [
        pca.explained_variance_.round(2),
        pca.explained_variance_ratio_.round(2),
        pca.explained_variance_ratio_.cumsum().round(2),
    ]
    fig, ax = subplots()
    table = ax.table(
        cellText=summary,
        colLabels=["PC1", "PC2"],
        rowLabels=["Explained Var", "Explained Var Ratio", "Cum Explained Var Ratio"],
        cellLoc="center",
        rowLoc="center",
        loc="center",
    )
    fig.savefig(f"{figures}/new_pca_summary_table.png")

    # Look at score plots to visualize how samples relate to each
    # other in the space defined by the principal components

    # Alternative PCA score plot using seaborn to use different markers
    custom_colors = ("black", "red")
    fig, ax = subplots(figsize=(10, 8))
    sns.scatterplot(
        x=components[:, 0],
        y=components[:, 1],
        hue=data_muscle[Dimension.GEAR.value],
        palette=custom_colors,
        legend="full",  # depending on how you want the legend to look, use this or replace with False and plt.legend below
        s=100,
    )
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    # plt.legend(handles=[
    #     Patch(color='black', label='Wild'),
    #     Patch(color='red', label='Farmed')],
    #     loc = 'upper right')
    fig.savefig(f"{figures}/pca_score_plot_gear.png")

    loadings = pca.components_.T * sqrt(explained_variance)
    # print(loadings)
    fig, ax = subplots(figsize=(10, 8))
    sns.scatterplot(
        x=loadings[:, 0],
        y=loadings[:, 1],
        hue=loadings[:, 1],
        palette="tab10",
        legend=False,  # depending on how you want the legend to look, use this or replace with False and plt.legend below
        s=150,
    )
    plt.xlim(-0.6, 0.6)
    plt.ylim(-0.6, 0.6)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    labels = ["d15N", "d13C", "C/N"]
    for i, txt in enumerate(labels):
        plt.text(loadings[:, 0][i], loadings[:, 1][i] + 0.02, txt, fontsize=12)
    plt.grid(True, "major")
    fig.savefig(f"{figures}/pca_loadings.png")

@plot.command("wild-lipid-extraction")
@option(
    "--tissue-type",
    default="M",
    help="Tissue type to plot (M for muscle, G for gonad).",
)
@figure_size((10, 8))
def plot_scatter_wild_lipid_extraction(
    tissue_type: str,
    figsize: tuple[float, float]
):
    """
    Determine if lipid extraction has an effect on the carbon fractionation
    of wild scallops.
    """
    tissue = load_scatter_data(isotopes_no_outliers)
    muscle = tissue.groupby(Dimension.TISSUE.value).get_group(tissue_type)
    fig, ax = subplots(figsize=figsize)
    wild_muscle = muscle[muscle[Dimension.GEAR.value] == "W"]
    scatterplot(
        x=wild_muscle[Dimension.MOLAR_RATIO.value],
        y=wild_muscle[Dimension.CARBON_FRACTIONATION.value],
        # hue = muscle[Dimension.GEAR.value],
        # palette = custom_colors,
        # style = muscle[Dimension.GEAR.value],
        # edgecolor = 'black',
        # facecolor = 'black',
        legend="auto",
        s=30,
        ax=ax,
    )
    fig.savefig(f"{figures}/wild_lipid_extraction.png")

@plot.command("pairs")
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

@plot.command("scatter-monthly-gear")
@tissue_type_option(TissueType.MUSCLE)
@figure_size((10, 3))
def plot_scatter_monthly_gear(
    tissue_type: TissueType,
    figsize: tuple[float, float]
):
    """
    Scatter plot of molar ratio vs carbon fractionation, separated 
    into subplots for each month.
    """
    df = read_csv(
            isotopes_no_outliers,
            header=0,
            usecols=[
                Dimension.TISSUE.value,
                Dimension.DATE_RUN.value,
                Dimension.CARBON_FRACTIONATION.value,
                Dimension.MOLAR_RATIO.value,
                Dimension.GEAR.value,
                Dimension.COLLECTION_DATE.value
            ],
        )
    # Remove known bad samples, and select only one tissue type for analysis
    mask = (~df[Dimension.DATE_RUN.value].isin(bad_run_dates)) & (
        df[Dimension.TISSUE.value] == tissue_type.value
    )
    df[Dimension.GEAR.value] = df[Dimension.GEAR.value].map(
        {"C": "Cage", "N": "Net", "W": "Wild"}
    )
    color_map = {"Cage": "tab:cyan", "Net": "tab:blue", "Wild": "tab:pink"}
    groups = (
        df[mask]
        .drop(columns=[Dimension.DATE_RUN.value, Dimension.TISSUE.value])
        .dropna()
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
        ax[ind].set_title(month_name[int(month)])
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

@plot.command("scatter-gear-monthly")
@figure_size((10, 8))
@tissue_type_option(TissueType.MUSCLE)
def plot_scatter_gear_monthly(
    figsize: tuple[float, float],
    tissue_type: TissueType
):
    """
    Carbon fractionation vs molar ratio for a tissue type, segemented
    by culture method and collection date (month). Plotting uses
    seaborn. For greater control, modify the code to use matplotlib directly.
    """
    df = read_csv(
            isotopes_no_outliers,
            header=0,
            usecols=[
                Dimension.TISSUE.value,
                Dimension.DATE_RUN.value,
                Dimension.CARBON_FRACTIONATION.value,
                Dimension.MOLAR_RATIO.value,
                Dimension.GEAR.value,
                Dimension.COLLECTION_DATE.value
            ],
        )
    # Remove known bad samples, and select only one tissue type for analysis
    mask = (~df[Dimension.DATE_RUN.value].isin(bad_run_dates)) & \
        (df[Dimension.TISSUE.value] == tissue_type.value)

    # Tissue type is already filtered, so only need to check gear and collection date
    df = df[mask] \
        .drop(columns=[Dimension.DATE_RUN.value, Dimension.TISSUE.value]) \
        .dropna()

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

@summarize.command("outliers")
def outliers():
    """
    CHECKING FOR OUTLIERS in DATASET WITH A Z-SCORE GREATER THAN 3
    """
    tissue = load_scatter_data(isotopes_no_outliers)
    muscle = tissue.groupby(Dimension.TISSUE.value).get_group("M")
    z = abs(zscore(muscle[Dimension.CARBON_FRACTIONATION.value]))
    a = abs(zscore(muscle[Dimension.NITROGEN_FRACTIONATION.value]))
    b = abs(zscore(muscle[Dimension.MOLAR_RATIO.value]))
    muscle["d13C z score"] = z
    muscle["d15N z score"] = a
    muscle["C/N z score"] = b
    d13C_outliers = muscle.loc[muscle["d13C z score"] > 3]
    d15N_outliers = muscle.loc[muscle["d15N z score"] > 3]
    CN_outliers = muscle.loc[muscle["C/N z score"] > 3]
    print(d13C_outliers)
    print(d15N_outliers)
    print(CN_outliers)


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

@plot.command("tissue")
@option(
    "--tissue-type",
    default="M",
    help="Tissue type to plot (M for muscle, G for gonad).",
)
@figure_size((6, 7))
def plot_tissue_by_month(
    tissue_type: str,
    figsize: tuple[float, float]
):
    """
    Render multiple plots that have boxplots of d13C, d15N, and C/N (molar) by month
    separated by gear type (net, cage, wild).
    """
    dim = 5
    context = subplots(dim, 1, figsize=figsize, squeeze=True)
    fig = context[0]
    ax: Sequence[Axes] = context[1]  # force type conversion for type hinting
    positions = [1.0, 2.0, 3.0, 4.0, 5.0]
    widths = 0.2
    groups = load_and_subset_source_data(raw_data)
    group_by = [
        Dimension.COLLECTION_DATE.value,
        Dimension.GEAR.value,
        Dimension.TISSUE.value,
    ]
    tissue = load_tissue_data(isotopes_no_outliers).groupby(group_by)
    gear = ["N", "C", "W"]
    for jj, [key, label] in enumerate([["d13C", r"$\delta$$^1$$^3$C"],
        ["d15N", r"$\delta$$^1$$^5$N"],
        ["C/N (Molar)", "C/N"],
        ["% N", "% N"],
    ]):
        for ii, (gg, color) in enumerate(zip(gear, custom_colors)):
            data = []
            for month in range(6, 11):
                gr = tissue.get_group((month, gg, tissue_type))
                data.append(gr[key])
            plot_partition(
                axis=ax[jj],
                data=data,
                positions=[x + 0.2 * ii for x in positions],
                widths=widths,
                color=color,
            )
        ax[jj].set_xticks([x + 0.2 for x in positions], labels=[])
        ax[jj].set_ylabel(label)

    for ii, (gg, color) in enumerate(zip(gear, custom_colors)):
        data = []
        for month in range(6, 11):
            gr = groups.get((gg, month), [])
            data.append(gr)
        plot_partition(
            axis=ax[4],
            data=data,
            positions=[x + 0.2 * ii for x in positions],
            widths=widths,
            color=color,
        )
    ax[4].set_xticks([x + widths for x in positions], [])
    ax[4].set_ylabel("GSI")
    ax[dim - 1].set_xticklabels(["June", "July", "August", "September", "October"])

    fig.legend(
        handles=[
            Patch(color="black", label="Net"),
            Patch(color="blue", label="Cage"),
            Patch(color="red", label="Wild"),
        ]
    )
    fig.savefig(figures / "tissue_by_month_boxplot.png")

@plot.command("ratio")
@figure_size((5, 2))
def plot_molar_ratio_by_tissue(
    figsize: tuple[float, float]
):
    """
    Boxplot figure comparing pre and post spawn for muscle and gonad
    """
    group_by = [
        Dimension.COLLECTION_DATE.value,
        Dimension.GEAR.value,
        Dimension.TISSUE.value,
    ]
    df = load_tissue_data(isotopes_no_outliers)
    summary = df[[Dimension.MOLAR_RATIO.value, *group_by]].groupby(group_by)
    context = subplots(2, 1, figsize=figsize, squeeze=True)
    fig = context[0]
    ax: Sequence[Axes] = context[1]
    positions = [1.0, 2.0]
    months = [8, 10]
    gear = ["N", "C", "W"]
    print(gear)
    for jj, tissue in enumerate(["M", "G"]):
        for ii, (gg, color) in enumerate(zip(gear, custom_colors)):
            data = [summary.get_group((mm, gg, tissue))[Dimension.MOLAR_RATIO.value] for mm in months]
            plot_partition(
                axis=ax[jj],
                data=data,
                positions=[x + 0.2 * ii for x in positions],
                widths=0.2,
                color=color,
            )
        ax[jj].set_xticks([x + 0.2 for x in positions], labels=[])
        ax[jj].set_ylabel(f"C/N of {tissue}")
    ax[1].set_xticklabels(["August", "October"])
    handles = [
        Patch(color=color, label=label) for label, color in zip(gear, custom_colors)
    ]
    fig.legend(handles=handles)
    fig.savefig(figures / "molar_ratio_by_tissue.png")


@summarize.command(Command.GEAR_TYPE.value)
@option(
    "--filename",
    default="tissue_by_gear_type_table",
    help="Filename for the output table.",
)
@file_output_options
def summarize_tissue_by_gear_type(
    filename: str,
    encoding: str,
    size: tuple[float, float],
    fontsize: int,
):
    """
    Table of averages and standard deviations for d13C, d15N,
    and C/N (molar) by gear type and month.
    """
    group_by = [
        Dimension.COLLECTION_DATE.value,
        Dimension.GEAR.value,
        Dimension.TISSUE.value,
    ]
    analyze = [
        Dimension.CARBON_FRACTIONATION.value,
        Dimension.NITROGEN_FRACTIONATION.value,
        Dimension.MOLAR_RATIO.value,
        Dimension.NITROGEN_PERCENTAGE.value,
    ]
    fcns = ["mean", "std"]
    agg_map = {key: fcns for key in analyze}
    summary = load_tissue_data(isotopes_no_outliers).groupby(group_by).agg(agg_map).round(2)
    text = DataFrame(index=summary.index)
    for key in analyze:
        text[key] = (
            summary[(key, "mean")].astype(str)
            + " ± "
            + summary[(key, "std")].astype(str)
        )
    fig, ax = subplots(figsize=size)
    ax.axis("off")
    table = ax.table(
        cellText=text,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    fig.tight_layout()
    fig.savefig(figures / f"{filename}.{encoding}")

@summarize.command(Command.TEMPERATURE.value)
@option(
    "--threshold",
    default=55.4,
    help="Temperature threshold for calculating degree hours.",
)
@option(
    "--filename",
    default="temperature_by_month_table",
    help="Filename for the output table.",
)
@file_output_options
def summarize_temperature_by_month(
    threshold: float,
    filename: str,
    encoding: str,
    size: tuple[float, float],
    fontsize: int
):
    """
    Calculate mean temperature and total hours above a set threshold, and save as a pivot table.
    The rows are the the culture method and statisitc, and the columns are the months.
    """

    def degree_hours(x: Series) -> float:
        """Aggregate function for degree hours"""
        return (x > threshold).sum()

    groups = load_temperature_data(env_data) \
        .drop(columns=[EnvDimension.DATE.value]) \
        .groupby("Month") \
        .aggregate(["mean", degree_hours]).T.round(2)

    fig, ax = subplots(figsize=size)
    ax.axis("off")
    table = ax.table(
        cellText=groups,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    fig.tight_layout()
    fig.savefig(figures / f"{filename}.{encoding}")
