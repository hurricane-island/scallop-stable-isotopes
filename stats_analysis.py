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
from numpy import arange, sqrt, column_stack, vstack
from matplotlib.pyplot import subplots, savefig, MultipleLocator
from scipy.stats import levene
from scipy.spatial import ConvexHull
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FactorAnalysis
from matplotlib import patches as mpatches
import seaborn as sns
import matplotlib.pyplot as plt
import prince 



figures = Path(__file__).parent / "figures"
raw_data = Path(__file__).parent / "data" / "2023IsotopeDataReport-no-outliers.csv"
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
    TISSUE = "Tissue Type" 
    SEX = "Sex"
    DATE_RUN = "Date Run"
    FARM_VS_WILD = "Farm vs Wild"  # New column for farm vs wild categorization, include?
    DEPTH = "Depth" # New column for depth categorization, include?


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

def return_column_to_categorical(
    df: DataFrame, column_name: str, categories: Dict[int, str]
) -> DataFrame:
    '''
    Replace integers with strings to return a categorical column of a DataFrame. 

    This is for the purpose of making plots more readable, 
    and should be used AFTER statistical analysis.

    This process changes the original dataframe.
    '''
    for value in df[column_name]:
        replace = categories.get(value, 0)
        df[column_name] = df[column_name].replace(value, replace)
    return df

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

    custom_colors = ('black', "blue", "red")
    fig, ax = subplots(figsize=(8, 6))
    # Confirm that the data is normally distributed
    for dim in [
        Dimension.NITROGEN_FRACTIONATION.value,
        Dimension.CARBON_FRACTIONATION.value,
        Dimension.MOLAR_RATIO.value,
    ]:
        data_muscle[dim].hist(ax=ax, label=dim, color = custom_colors[["d15N", "d13C", "C/N (Molar)"].index(dim)])
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
            print("Month:", a.statistic, "P-value:", a.pvalue, "(Passed)" if a.pvalue > 0.05 else "(Failed)")
            print("Gear:", b.statistic, "P-value:", b.pvalue, "(Passed)" if b.pvalue > 0.05 else "(Failed)")

    '''
    Since ANOVA assumptions are not met, try PCA
    Ensure data are quantized properly
    '''

    df = quantize_categorical_column(df, Dimension.GEAR.value, {"C": 1, "N": 2, "W": 3})
    df = quantize_categorical_column(df, Dimension.SEX.value, {"F": 1, "M": 2})
    df = quantize_categorical_column(df, Dimension.TISSUE.value, {"G": 1, "M": 2})
    data_muscle = quantize_categorical_column(data_muscle, Dimension.GEAR.value, {"C": 1, "N": 2, "W": 3})
    data_muscle = quantize_categorical_column(data_muscle, Dimension.SEX.value, {"F": 1, "M": 2})
    data_muscle = quantize_categorical_column(data_muscle, Dimension.TISSUE.value, {"G": 1, "M": 2})

    
    pca_df = data_muscle[[ 
            Dimension.NITROGEN_FRACTIONATION.value,
            Dimension.CARBON_FRACTIONATION.value,
            Dimension.MOLAR_RATIO.value,
    ]]

    std_scaler = StandardScaler()
    scaled_df = std_scaler.fit_transform(pca_df)
    pca = PCA(n_components=2)

    components = pca.fit_transform(scaled_df)
    explained_variance = pca.explained_variance_ratio_
    loadings = pca.components_.T * sqrt(explained_variance)


    # Make a scree plot to visualize the proportion of variance
    # explained by each principal component
    fig, ax = subplots(figsize=(4, 2))
    pc_numbers = arange(len(explained_variance)) + 1
    ax.plot(pc_numbers, explained_variance, marker="o", linestyle="-")
    ax.set_title("Scree Plot")
    ax.set_xlabel("Principal Component Number")
    ax.set_ylabel("Proportion of Explained Variance")
    ax.grid(True)
    fig.savefig(f"{figures}/pca_scree_plot.png")

    fig, ax = subplots(figsize=(10, 3))
    ax.axis("off")

    '''
    Factor Analysis of Mixed Data
    Variables used: CARBON_FRACTIONATION, NITROGEN_FRACTIONATION, MOLAR_RATIO, COLLECTION_DATE
    '''
    data_muscle = return_column_to_categorical(data_muscle, Dimension.COLLECTION_DATE.value, {6: 'June', 7: 'July', 8: 'August', 9: 'September', 10: 'October'}) 
    data_muscle = return_column_to_categorical(data_muscle, Dimension.GEAR.value, {1: 'Farm', 2: 'Farm', 3: 'Wild'}) 
    df = return_column_to_categorical(df, Dimension.COLLECTION_DATE.value, {6: 'June', 7: 'July', 8: 'August', 9: 'September', 10: 'October'}) 
    df = return_column_to_categorical(df, Dimension.GEAR.value, {1: 'Farm', 2: 'Farm', 3: 'Wild'}) 


    famd = prince.FAMD(
        n_components=2,
        n_iter=3,
        copy=True,
        check_input=True,
        random_state=None,
        engine="sklearn",
        handle_unknown="error"  # same parameter as sklearn.preprocessing.OneHotEncoder
)
    famd_data = data_muscle[[Dimension.CARBON_FRACTIONATION.value,
            Dimension.NITROGEN_FRACTIONATION.value,
            Dimension.MOLAR_RATIO.value,
            Dimension.COLLECTION_DATE.value]]
    famd = famd.fit(famd_data)
    
    print(famd.eigenvalues_summary)
    print(famd.column_contributions_)

    factors_famd = DataFrame(famd.row_coordinates(famd_data))
 
     #FAMD plot using seaborn to use different markers
    custom_colors = ('black', 'red')
    fig, ax = subplots(figsize=(10, 8))
    sns.scatterplot(
        x=factors_famd[0],
        y=factors_famd[1],
        hue = data_muscle[Dimension.GEAR.value],
        style = data_muscle[Dimension.GEAR.value],
        markers = ('o', 'D'),
        palette = custom_colors,
        legend = 'brief',
        s=30,
        ax=ax
    )
    for date in data_muscle[Dimension.COLLECTION_DATE.value].unique():
        if date == 'October':
            subset_mask = data_muscle[Dimension.COLLECTION_DATE.value] == date
            subset_points = column_stack((
                factors_famd.loc[subset_mask, 0],
                factors_famd.loc[subset_mask, 1]
            ))
        
            if len(subset_points) >= 3:
                hull = ConvexHull(subset_points)
                hull_points = subset_points[hull.vertices]
                hull_points = vstack([hull_points, hull_points[0]])  # close polygon

                ax.fill(
                    hull_points[:, 0],
                    hull_points[:, 1],
                    alpha=0,
                    label=f'{date}',
                    zorder=2
                )
                ax.plot(
                    hull_points[:, 0],
                    hull_points[:, 1],
                    lw=1,
                    linestyle='--',
                    color='black',
                    zorder=3
                )
        if date == 'June':
            subset_mask = data_muscle[Dimension.COLLECTION_DATE.value] == date
            subset_points = column_stack((
                factors_famd.loc[subset_mask, 0],
                factors_famd.loc[subset_mask, 1]
            ))
        
            if len(subset_points) >= 3:
                hull = ConvexHull(subset_points)
                hull_points = subset_points[hull.vertices]
                hull_points = vstack([hull_points, hull_points[0]])  # close polygon

                ax.fill(
                    hull_points[:, 0],
                    hull_points[:, 1],
                    alpha=0,
                    label=f'{date}',
                    zorder=2
                )
                ax.plot(
                    hull_points[:, 0],
                    hull_points[:, 1],
                    lw=1,
                    linestyle='-',
                    color='black',
                    zorder=3
                )

    plt.xlabel("Factor 1")
    plt.ylabel("Factor 2")
    plt.show()
    # fig.savefig(f"{figures}/convex_hull_famd_date_gear_plot.png")

    custom_colors = ('black', "red")
    fig, ax = subplots(figsize=(10, 8))
    sns.scatterplot(
        x=factors_famd[0],
        y=factors_famd[1],
        hue = data_muscle[Dimension.GEAR.value],
        style = data_muscle['Depth'],
        palette=custom_colors, 
        legend = 'auto',
        s=100,
        zorder = 2,
    )
    plt.xlim(-4,4)
    plt.ylim(-4,4)
    plt.xlabel("Factor 1")
    plt.ylabel("Factor 2")
    fig.savefig(f"{figures}/famd_gear_plot.png")

    df = quantize_categorical_column(df, Dimension.COLLECTION_DATE.value, {'June':6, 'July':7,'August':8,'September':9,'October':10}) 
    data_muscle = quantize_categorical_column(data_muscle, Dimension.COLLECTION_DATE.value, {'June':6, 'July':7,'August':8,'September':9,'October':10}) 

    '''
    3D plot showing 3 factors
    '''
    # x = factors_famd[0]
    # y = factors_famd[1]
    # z = factors_famd[2]
    # hue_var = data_muscle[Dimension.COLLECTION_DATE.value]
    # fig = plt.figure(figsize=(10, 8))
    # ax = fig.add_subplot(111, projection='3d')
    # scatter = ax.scatter(x, y, z, c=hue_var, cmap='icefire', s=50)
    # plt.show()

    # Make a table to summarize the PCA results
    pca.explained_variance_.tolist()
    pca.explained_variance_ratio_.tolist()
    pca.explained_variance_ratio_.cumsum().tolist()

    summary = [
        pca.explained_variance_.round(2),
        pca.explained_variance_ratio_.round(2),
        pca.explained_variance_ratio_.cumsum().round(2),
    ]
    fig, ax = subplots(figsize=(10, 8))
    table = ax.table(
        cellText=summary,
        colLabels=[
            "PC1",
            "PC2"
        ],
        rowLabels=["Explained Var", "Explained Var Ratio", "Cum Explained Var Ratio"],
        cellLoc="center",
        rowLoc="center",
        loc="center",
    )
    fig.savefig(f"{figures}/new_pca_summary_table.png")

    # Look at score plots to visualize how samples relate to each
    # other in the space defined by the principal components

    # Alternative PCA score plot using seaborn to use different markers
    custom_colors = ('black', "red")
    fig, ax = subplots(figsize=(10, 8))
    sns.scatterplot(
        x=components[:, 0],
        y=components[:, 1],
        hue=data_muscle[Dimension.GEAR.value],
        style=data_muscle['Depth'],
        palette =custom_colors, 
        legend = 'full', #depending on how you want the legend to look, use this or replace with False and plt.legend below
        s=100,
    )
    plt.xlim(-4,4)
    plt.ylim(-4,4)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    # plt.legend(handles=[
    #     mpatches.Patch(color='black', label='Wild'),
    #     mpatches.Patch(color='red', label='Farmed')],
    #     loc = 'upper right')
    fig.savefig(f"{figures}/pca_score_plot_gear.png")

    '''
    PCA loadings plot
    '''
    # print(loadings)
    fig, ax = subplots(figsize=(10, 8))
    sns.scatterplot(
        x=loadings[:, 0],
        y=loadings[:, 1],
        hue = loadings[:, 1],
        palette='tab10',
        legend = False, #depending on how you want the legend to look, use this or replace with False and plt.legend below
        s=150,
    )
    plt.xlim(-0.6,0.6)
    plt.ylim(-0.6,0.6)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    labels = ['d15N','d13C','C/N']
    for i, txt in enumerate(labels):
        plt.text(loadings[:, 0][i], loadings[:,1][i] + 0.02, txt, fontsize=12)
    plt.grid(True, 'major')
    fig.savefig(f"{figures}/pca_loadings.png")



