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
from typing import Dict
from pandas import DataFrame, read_csv, set_option
from numpy import arange, sqrt
from matplotlib.pyplot import subplots
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
from isotopes.options import (
    Dimension,
    bad_run_dates,
    figures,
    isotopes_no_outliers,
)
from isotopes.plot import plot


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



@plot.command("clustering")
def clustering_and_pca_analysis():
    """
    Perform clustering and PCA analysis on the muscle tissue data.
    """
    # only scallops and filters are being plotted
    df = read_csv(
        isotopes_no_outliers,
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

    # Variables used: CARBON_FRACTIONATION, NITROGEN_FRACTIONATION, MOLAR_RATIO, COLLECTION_DATE
    data_muscle[Dimension.COLLECTION_DATE.value] = data_muscle[Dimension.COLLECTION_DATE.value].map(dict((i, month_name[i]) for i in range(1, 13)))
    data_muscle[Dimension.GEAR.value] = data_muscle[Dimension.GEAR.value].map({1: "Farm", 2: "Farm", 3: "Wild"})
    df[Dimension.COLLECTION_DATE.value] = df[Dimension.COLLECTION_DATE.value].map(dict((i, month_name[i]) for i in range(1, 13)))
    df[Dimension.GEAR.value] = df[Dimension.GEAR.value].map({1: "Farm", 2: "Farm", 3: "Wild"})

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
    ax.table(
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
