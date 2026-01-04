"""
This module contains methods for statistical analysis of stable isotope data,
especially Principal Component Analysis (PCA) using the `sklearn` and `scipy`
libraries.
"""

from pathlib import Path
from typing import Dict, Sequence, Union
from enum import Enum
from pandas import DataFrame, read_csv, to_datetime, set_option, Series
from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols
from numpy import arange, sqrt, column_stack, vstack, abs
from matplotlib.pyplot import subplots, savefig
from matplotlib.axes import Axes
from matplotlib.patches import Patch
from scipy.stats import levene, zscore
from scipy.spatial import ConvexHull
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
from prince import FAMD
from seaborn import scatterplot, pairplot


bad_run_dates = {"9/6/23"}  # use Set() as more generic lookup than single value
figures = Path(__file__).parent / "figures"
data_dir = Path(__file__).parent / "data"
raw_data = data_dir / "gonadosomatic-index.csv"
scatter_data = data_dir / "stable-isotopes-no-outliers.csv"
env_data = data_dir / "temperature-and-light.csv"
custom_colors = ("black", "blue", "red")


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


def load_temperature_data() -> DataFrame:
    """
    Load the temperature data from the CSV file.

    Returns:
        DataFrame: The loaded temperature data.
    """
    df = read_csv(
        env_data,
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
    df["Month"] = times.dt.month
    return df


def load_and_subset_source_data():
    """
    Data needed for GSI boxplots. Converts from data from, to a dictionary grouped by
    tuples of gear and month that can be combined as needed for plotting.
    """
    df = read_csv(
        raw_data,
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


def load_tissue_data():
    """Same as scatter data loader"""
    df = read_csv(
        scatter_data,
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
            "d13C",
            "d15N",
            "C/N (Molar)",
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


def load_scatter_data() -> DataFrame:
    """Load scatter plot data, filtering out unwanted tissues and contaminated samples."""
    df = read_csv(
        scatter_data,
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


# pylint: disable=redefined-outer-name
def analysis_of_variance(df: DataFrame) -> DataFrame:
    """
    Perform ANOVA on the selected tissue data using
    Ordinary Least Squares (OLS) model.
    """
    subset = df[
        [
            Dimension.NITROGEN_PERCENTAGE.value,
            Dimension.GEAR.value,
            Dimension.COLLECTION_DATE.value,
        ]
    ].dropna()

    # Make columns categorical
    subset[Dimension.GEAR.value] = subset[Dimension.GEAR.value].astype("category")
    subset[Dimension.COLLECTION_DATE.value] = subset[
        Dimension.COLLECTION_DATE.value
    ].astype("category")

    # Rename columns
    subset = subset.rename(
        columns={
            Dimension.NITROGEN_PERCENTAGE.value: "N",
            Dimension.GEAR.value: "Gear",
            Dimension.COLLECTION_DATE.value: "Month",
        }
    )
    model = ols("N ~ C(Gear) + C(Month) + C(Gear):C(Month)", data=subset).fit()
    return anova_lm(model, type=2)  # Type II sum of squares


# pylint: disable=redefined-outer-name
def levenes_test_month(df: DataFrame, column: str):
    """
    Levene's test for homogeneity of variances for a given column.

    If p > 0.05, we can assume homogeneity of variances
    """
    monthly = df.groupby(Dimension.COLLECTION_DATE.value)
    june = monthly.get_group(6)
    july = monthly.get_group(7)
    august = monthly.get_group(8)
    september = monthly.get_group(9)
    october = monthly.get_group(10)
    return levene(
        june[column],
        july[column],
        august[column],
        september[column],
        october[column],
    )


# pylint: disable=redefined-outer-name
def levenes_test_gear(df: DataFrame, column: str):
    """
    Levene's test for homogeneity of variances for a given column.

    If p > 0.05, we can assume homogeneity of variances
    """

    gear_types = df.groupby(Dimension.GEAR.value)
    cages = gear_types.get_group("C")
    nets = gear_types.get_group("N")
    wild = gear_types.get_group("W")
    return levene(cages[column], nets[column], wild[column])


# pylint: disable=redefined-outer-name
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


# When file is run directly, this block will execute.
# The reason to do this is so that as the methods are wrapped as functions
# and imported into other scripts, this block will not execute, when that
# happens.
def clustering_and_pca_analysis():
    df = read_csv(
        scatter_data,
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
    for date in range(len(df[Dimension.DATE_RUN.value])):
        if df[Dimension.DATE_RUN.value][date] in bad_run_dates:
            df.drop(date, inplace=True)
        else:
            to_datetime(df[Dimension.DATE_RUN.value][date], format="%m/%d/%y")

    # Don't need the date run for analysis, only pre-filtering
    df = df.drop(columns=[Dimension.DATE_RUN.value])

    # only scallops and filters are being plotted
    df.dropna(subset=[Dimension.GEAR.value], inplace=True)

    tissue = df.dropna(subset=[Dimension.TISSUE.value])
    mask = tissue[Dimension.TISSUE.value] == "G"
    data_muscle = tissue.drop(tissue[mask].index)

    custom_colors = ("black", "blue", "red")
    fig, ax = subplots(figsize=(8, 6))
    # Confirm that the data is normally distributed
    for dim in [
        Dimension.NITROGEN_FRACTIONATION.value,
        Dimension.CARBON_FRACTIONATION.value,
        Dimension.MOLAR_RATIO.value,
    ]:
        data_muscle[dim].hist(
            ax=ax,
            label=dim,
            color=custom_colors[["d15N", "d13C", "C/N (Molar)"].index(dim)],
        )
    ax.legend()
    ax.grid(False)
    savefig(f"{figures}/muscle_tissue_histograms.png")

    mask = tissue[Dimension.TISSUE.value] == "M"
    data_gonad = tissue.drop(tissue[mask].index)

    print("\nAnalysis of Variance")
    print("\nTissue: Gonad\n")
    print(analysis_of_variance(data_gonad))

    print("\nTissue: Muscle\n")
    print(analysis_of_variance(data_muscle))

    print("\nLevene's Test of Homogeneity of Variance")
    for name, tissue in {"Gonad": data_gonad, "Muscle": data_muscle}.items():
        print(f"\nTissue: {name}")
        for dim in [
            Dimension.CARBON_FRACTIONATION.value,
            Dimension.NITROGEN_FRACTIONATION.value,
            Dimension.MOLAR_RATIO.value,
        ]:
            print(f"\nDimension: {dim}")
            a = levenes_test_month(data_muscle, dim)
            b = levenes_test_gear(data_muscle, dim)
            print(
                "Month:",
                a.statistic,
                "P-value:",
                a.pvalue,
                "(Passed)" if a.pvalue > 0.05 else "(Failed)",
            )
            print(
                "Gear:",
                b.statistic,
                "P-value:",
                b.pvalue,
                "(Passed)" if b.pvalue > 0.05 else "(Failed)",
            )

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

    famd = FAMD(
        n_components=2,
        n_iter=3,
        copy=True,
        check_input=True,
        random_state=None,
        engine="sklearn",
        handle_unknown="error",  # same parameter as sklearn.preprocessing.OneHotEncoder
    )
    famd_data = data_muscle[
        [
            Dimension.CARBON_FRACTIONATION.value,
            Dimension.NITROGEN_FRACTIONATION.value,
            Dimension.MOLAR_RATIO.value,
            Dimension.COLLECTION_DATE.value,
        ]
    ]
    famd = famd.fit(famd_data)

    print(famd.eigenvalues_summary)
    print(famd.column_contributions_)

    factors_famd = DataFrame(famd.row_coordinates(famd_data))

    # FAMD plot using seaborn to use different markers
    custom_colors = ("black", "red")
    fig, ax = subplots(figsize=(10, 8))
    sns.scatterplot(
        x=factors_famd[0],
        y=factors_famd[1],
        hue=data_muscle[Dimension.GEAR.value],
        style=data_muscle[Dimension.GEAR.value],
        markers=("o", "D"),
        palette=custom_colors,
        legend="brief",
        s=30,
        ax=ax,
    )
    for date in data_muscle[Dimension.COLLECTION_DATE.value].unique():
        if date == "October":
            subset_mask = data_muscle[Dimension.COLLECTION_DATE.value] == date
            subset_points = column_stack(
                (factors_famd.loc[subset_mask, 0], factors_famd.loc[subset_mask, 1])
            )

            if len(subset_points) >= 3:
                hull = ConvexHull(subset_points)
                hull_points = subset_points[hull.vertices]
                hull_points = vstack([hull_points, hull_points[0]])  # close polygon

                ax.fill(
                    hull_points[:, 0],
                    hull_points[:, 1],
                    alpha=0,
                    label=f"{date}",
                    zorder=2,
                )
                ax.plot(
                    hull_points[:, 0],
                    hull_points[:, 1],
                    lw=1,
                    linestyle="--",
                    color="black",
                    zorder=3,
                )
        if date == "June":
            subset_mask = data_muscle[Dimension.COLLECTION_DATE.value] == date
            subset_points = column_stack(
                (factors_famd.loc[subset_mask, 0], factors_famd.loc[subset_mask, 1])
            )

            if len(subset_points) >= 3:
                hull = ConvexHull(subset_points)
                hull_points = subset_points[hull.vertices]
                hull_points = vstack([hull_points, hull_points[0]])  # close polygon

                ax.fill(
                    hull_points[:, 0],
                    hull_points[:, 1],
                    alpha=0,
                    label=f"{date}",
                    zorder=2,
                )
                ax.plot(
                    hull_points[:, 0],
                    hull_points[:, 1],
                    lw=1,
                    linestyle="-",
                    color="black",
                    zorder=3,
                )
    ax.set_xlabel("Factor 1")
    ax.set_ylabel("Factor 2")
    fig.savefig(f"{figures}/convex_hull_famd_date_gear_plot.png")

    custom_colors = ("black", "red")
    fig, ax = subplots(figsize=(10, 8))
    sns.scatterplot(
        x=factors_famd[0],
        y=factors_famd[1],
        hue=data_muscle[Dimension.GEAR.value],
        palette=custom_colors,
        legend="auto",
        s=100,
        zorder=2,
    )
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_xlabel("Factor 1")
    ax.set_ylabel("Factor 2")
    fig.savefig(f"{figures}/famd_gear_plot.png")

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

    # 3D plot showing 3 factors

    # x = factors_famd[0]
    # y = factors_famd[1]
    # z = factors_famd[2]
    # hue_var = data_muscle[Dimension.COLLECTION_DATE.value]
    # fig = plt.figure(figsize=(10, 8))
    # ax = fig.add_subplot(111, projection='3d')
    # scatter = ax.scatter(x, y, z, c=hue_var, cmap='icefire', s=50)
    # plt.show()

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


def plot_scatter_wild_lipid_extraction():
    tissue = load_scatter_data()
    muscle = tissue.groupby(Dimension.TISSUE.value).get_group("M")
    fig, ax = subplots(figsize=(10, 8))
    wild_muscle = muscle[muscle[Dimension.GEAR.value] == "Wild"]
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


def plot_pairs_seaborn():
    tissue = load_scatter_data()
    tissue = quantize_categorical_column(
        tissue, Dimension.GEAR.value, {"C": 1, "N": 2, "W": 3}
    )
    tissue = return_column_to_categorical(
        tissue, Dimension.GEAR.value, {1: "Cage", 2: "Net", 3: "Wild"}
    )
    pairplot(
        tissue[
            [
                Dimension.NITROGEN_FRACTIONATION.value,
                Dimension.CARBON_FRACTIONATION.value,
                Dimension.MOLAR_RATIO.value,
                Dimension.GEAR.value,
            ]
        ],
        hue=Dimension.GEAR.value,
        palette=custom_colors,
    ).savefig(figures / "new-pairplot-all-tissue-gear.png")


def plot_scatter_monthly_gear():
    tissue = load_scatter_data()
    groups = tissue.groupby([Dimension.TISSUE.value, Dimension.COLLECTION_DATE.value])

    june = groups.get_group(("M", 6))
    july = groups.get_group(("M", 7))
    august = groups.get_group(("M", 8))
    september = groups.get_group(("M", 9))
    october = groups.get_group(("M", 10))

    context = subplots(1, 5, figsize=(10, 3), sharex=False, sharey=False)
    fig = context[0]
    ax: list[Axes] = context[1]

    ax[0].scatter(
        june[Dimension.MOLAR_RATIO.value],
        june[Dimension.CARBON_FRACTIONATION.value],
        marker="x",
        cmap="tab10",
    )
    ax[0].set_title("June")
    ax[0].set_xlim(3, 6)
    ax[0].set_ylim(-19, -16)
    ax[0].set_xlabel("C/N")
    ax[0].set_ylabel("d13C")

    ax[1].scatter(
        july[Dimension.MOLAR_RATIO.value],
        july[Dimension.CARBON_FRACTIONATION.value],
        marker="x",
        cmap="tab10",
    )
    ax[1].set_title("July")
    ax[1].set_xlim(3, 6)
    ax[1].set_ylim(-19, -16)
    ax[1].set_yticks([])

    ax[2].scatter(
        august[Dimension.MOLAR_RATIO.value],
        august[Dimension.CARBON_FRACTIONATION.value],
        marker="x",
        cmap="tab10",
    )
    ax[2].set_title("August")
    ax[2].set_xlim(3, 6)
    ax[2].set_ylim(-19, -16)
    ax[2].set_xlabel("C/N")

    ax[3].scatter(
        september[Dimension.MOLAR_RATIO.value],
        september[Dimension.CARBON_FRACTIONATION.value],
        marker="x",
        cmap="tab10",
    )
    ax[3].set_title("September")
    ax[3].set_xlim(3, 6)
    ax[3].set_ylim(-19, -16)
    ax[3].set_xlabel("C/N")

    ax[4].scatter(
        october[Dimension.MOLAR_RATIO.value],
        october[Dimension.CARBON_FRACTIONATION.value],
        marker="x",
        cmap="tab10",
    )
    ax[4].set_title("October")
    ax[4].set_xlim(3, 6)
    ax[4].set_ylim(-19, -16)
    ax[4].set_xlabel("C/N")
    fig.legend(
        handles=[
            Patch(color="tab:blue", label="Farm"),
            Patch(color="tab:red", label="Wild"),
            Patch(color="tab:cyan", label="Farm Filter"),
            Patch(color="tab:pink", label="Wild Filter"),
        ],
        bbox_to_anchor=(1.05, 1),
    )
    fig.savefig(figures / "rawdata_scatter_monthly_gear.png")


def plot_scatter_gear_monthly():
    """
    Seaborn scatter plot of d13C vs C/N for muscle tissue colored by gear type and shaped by month.
    """
    fig, ax = subplots(figsize=(10, 8))
    tissue = load_scatter_data()
    muscle = tissue.groupby(Dimension.TISSUE.value).get_group("M")
    scatterplot(
        x=muscle[Dimension.MOLAR_RATIO.value],
        y=muscle[Dimension.CARBON_FRACTIONATION.value],
        hue=muscle[Dimension.GEAR.value],
        palette="tab10",
        style=muscle[Dimension.COLLECTION_DATE.value],
        legend="auto",  # depending on how you want the legend to look, use this or replace with False and plt.legend below
        s=150,
        ax=ax,
    )
    ax.set_xlabel("C/N")
    ax.set_ylabel("d13C")
    fig.savefig(figures / "rawdata_scatter_gear_monthly.png")


def outliers():
    """CHECKING FOR OUTLIERS in DATASET WITH A Z-SCORE GREATER THAN 3"""
    tissue = load_scatter_data()
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

def plot_tissue_by_month(tissue_type: str = "M"):
    """
    Render multiple plots that have boxplots of d13C, d15N, and C/N (molar) by month
    separated by gear type (net, cage, wild).
    """
    dim = 5
    context = subplots(dim, 1, figsize=(6, 7), squeeze=True)
    fig = context[0]
    ax: Sequence[Axes] = context[1]
    positions = [1.0, 2.0, 3.0, 4.0, 5.0]
    widths = 0.2
    groups = load_and_subset_source_data()
    group_by = [
        Dimension.COLLECTION_DATE.value,
        Dimension.GEAR.value,
        Dimension.TISSUE.value,
    ]
    tissue = load_tissue_data().groupby(group_by)
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
    fig.savefig(figures / "boxplot.png")


def plot_molar_ratio_by_tissue():
    """
    Boxplot figure comparing pre and post spawn for muscle and gonad
    """
    group_by = [
        Dimension.COLLECTION_DATE.value,
        Dimension.GEAR.value,
        Dimension.TISSUE.value,
    ]
    df = load_tissue_data()
    summary = df[[Dimension.MOLAR_RATIO.value, *group_by]].groupby(group_by)
    context = subplots(2, 1, figsize=(5, 2), squeeze=True)
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


def summarize_tissue_by_gear_type(
    filename: str = "tissue_by_gear_type_table.png",
    width: float = 10,
    height: float = 7,
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
    summary = load_tissue_data().groupby(group_by).agg(agg_map).round(2)
    text = DataFrame(index=summary.index)
    for key in analyze:
        text[key] = (
            summary[(key, "mean")].astype(str)
            + " ± "
            + summary[(key, "std")].astype(str)
        )
    fig, ax = subplots(figsize=(width, height))
    ax.axis("off")
    table = ax.table(
        cellText=text,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    fig.tight_layout()
    fig.savefig(figures / filename)


def summarize_temperature_by_month(
    temperature_threshold: float = 55.4,
    filename: str = "temperature_by_month_table",
    encoding: str = "png",
    width: float = 10,
    height: float = 7,
):
    """
    Calculate and plot temperature statistics:
    - Mean temperature by month for nets, cages, and wild.
    - Degree hours above threshold by month for nets, cages, and wild.
    """

    def degree_hours(x: Series) -> float:
        """Aggregate function for degree hours"""
        return (x > temperature_threshold).sum()

    df = load_temperature_data().drop(columns=[EnvDimension.DATE.value])
    groups = df.groupby("Month").aggregate(["mean", degree_hours]).T.round(2)
    groups.columns = ["June", "July", "August", "September", "October"]
    fig, ax = subplots(figsize=(width, height))
    ax.axis("off")
    table = ax.table(
        cellText=groups,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    fig.tight_layout()
    fig.savefig(figures / f"{filename}.{encoding}")


if __name__ == "__main__":
    summarize_temperature_by_month()
    summarize_tissue_by_gear_type()
    plot_tissue_by_month()
    plot_molar_ratio_by_tissue()
    # plot_scatter_gear_monthly()
    # plot_scatter_monthly_gear()
    # plot_pairs_seaborn()
    # plot_scatter_wild_lipid_extraction()
    # outliers()
