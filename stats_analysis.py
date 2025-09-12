"""
This module contains methods for statistical analysis of stable isotope data,
especially Principal Component Analysis (PCA) using the `sklearn` and `scipy`
libraries.
"""

from pathlib import Path
from typing import Dict
from enum import Enum
from pandas import DataFrame, read_csv, to_datetime, set_option
import statsmodels.api as sm
from statsmodels.formula.api import ols
from numpy import arange
from matplotlib.pyplot import subplots, savefig
from scipy.stats import levene
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

figures = Path(__file__).parent / "figures"
raw_data = Path(__file__).parent / "data" / "2023IsotopeDataReport-cleanedinexcel.csv"
bad_run_dates = {"9/6/23"}  # use Set() as more generic lookup than single value


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
    TISSUE = "Tissue Type (Gonad or Muscle)"
    SEX = "Sex"
    DATE_RUN = "Date Run"


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
    return sm.stats.anova_lm(model, type=2)  # Type II sum of squares



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
    set_option('future.no_silent_downcasting', True)  # suppress runtime warning
    for value in df[column_name]:
        replace = categories.get(value, 0)
        df[column_name] = df[column_name].replace(value, replace)
    non_zero_mask = df[column_name] != 0
    return df[non_zero_mask]


# When file is run directly, this block will execute.
# The reason to do this is so that as the methods are wrapped as functions
# and imported into other scripts, this block will not execute, when that
# happens.
if __name__ == "__main__":
    df = read_csv(
        raw_data,
        header=0,
        usecols=[
            Dimension.COLLECTION_DATE.value,
            Dimension.GEAR.value,
            Dimension.SEX.value,
            Dimension.TISSUE.value,
            "Mass (mg)", # possible PCA input
            Dimension.NITROGEN_PERCENTAGE.value,
            "N (umoles)", # possible PCA input if N limited?
            Dimension.NITROGEN_FRACTIONATION.value,
            Dimension.CARBON_PERCENTAGE.value,
            "C (umoles)", # possible PCA input if N limited?
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
    df = df.drop(columns=[
        Dimension.DATE_RUN.value
    ])

    # only scallops and filters are being plotted
    df.dropna(subset=[Dimension.GEAR.value], inplace=True)

    tissue = df.dropna(subset=[Dimension.TISSUE.value])
    mask = tissue[Dimension.TISSUE.value] == "G"
    data_muscle = tissue.drop(tissue[mask].index)

    fig, ax = subplots(figsize=(8, 6))
    # Confirm that the data is normally distributed
    for dim in [
        Dimension.NITROGEN_PERCENTAGE.value,
        Dimension.CARBON_FRACTIONATION.value,
        Dimension.NITROGEN_FRACTIONATION.value,
        Dimension.MOLAR_RATIO.value,
    ]:
        data_muscle[dim].hist(ax=ax, label=dim)
    ax.legend()
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
            Dimension.NITROGEN_PERCENTAGE.value,
            Dimension.CARBON_FRACTIONATION.value,
            Dimension.NITROGEN_FRACTIONATION.value,
            Dimension.MOLAR_RATIO.value,
        ]:
            print(f"\nDimension: {dim}")
            a = levenes_test_month(data_muscle, dim)
            b = levenes_test_gear(data_muscle, dim)
            print("Month:", a.statistic, "P-value:", a.pvalue, "(Passed)" if a.pvalue > 0.05 else "(Failed)")
            print("Gear:", b.statistic, "P-value:", b.pvalue, "(Passed)" if b.pvalue > 0.05 else "(Failed)")

    # Since ANOVA assumptions are not met, try PCA
    df = quantize_categorical_column(df, Dimension.GEAR.value, {"C": 1, "N": 2, "W": 3})
    df = quantize_categorical_column(df, Dimension.SEX.value, {"F": 1, "M": 2})
    df = quantize_categorical_column(df, Dimension.TISSUE.value, {"G": 1, "M": 2})

    std_scaler = StandardScaler()
    scaled_df = std_scaler.fit_transform(df)
    pca = PCA(n_components=None)
    components = pca.fit_transform(scaled_df)
    comp_df = DataFrame(pca.components_, columns=list(df.columns))

    comp_df.to_csv(f"{figures}/pca_components2.csv")
    explained_variance = pca.explained_variance_ratio_
    # loadings = pca.components_.T * np.sqrt(explained_variances)

    # Make a scree plot to visualize the proportion of variance
    # explained by each principal component
    fig, ax = subplots(figsize=(8, 6))
    pc_numbers = arange(len(explained_variance)) + 1
    ax.plot(pc_numbers, explained_variance, marker="o", linestyle="-")
    ax.set_title("Scree Plot")
    ax.set_xlabel("Principal Component Number")
    ax.set_ylabel("Proportion of Explained Variance")
    ax.grid(True)
    fig.savefig(f"{figures}/pca_scree_plot.png")

    fig, ax = subplots(figsize=(40, 10))
    ax.axis("off")

    # Make a table to summarize the PCA results
    pca.explained_variance_.tolist()
    pca.explained_variance_ratio_.tolist()
    pca.explained_variance_ratio_.cumsum().tolist()

    summary = [
        pca.explained_variance_.round(2),
        pca.explained_variance_ratio_.round(2),
        pca.explained_variance_ratio_.cumsum().round(2),
    ]

    table = ax.table(
        cellText=summary,
        colLabels=[
            "PC1",
            "PC2",
            "PC3",
            "PC4",
            "PC5",
            "PC6",
            "PC7",
            "PC8",
            "PC9",
            "PC10",
            "PC11",
            "PC12",
            "PC13",
        ],
        rowLabels=["Explained Var", "Explained Var Ratio", "Cum Explained Var Ratio"],
        cellLoc="center",
        rowLoc="center",
        loc="center",
    )
    fig.savefig(f"{figures}/pca_summary_table.png")

    # Look at score plots to visualize how samples relate to each
    # other in the space defined by the principal components
    fig, ax = subplots(figsize=(10, 8))
    column = Dimension.COLLECTION_DATE.value  # Dimension.GEAR.value
    for dates in df[column]:
        ax.scatter(components[:, 0], components[:, 1], c=df[column], cmap="viridis")
        ax.set_title("PCA Score Plot: PC1 vs PC2")
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)

    fig.savefig(f"{figures}/pca_score_plot.png")
