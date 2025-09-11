"""
This module contains methods for statistical analysis of stable isotope data,
especially Principal Component Analysis (PCA) using the `sklearn` and `scipy`
libraries.
"""
from pathlib import Path
from typing import Dict
from pandas import DataFrame, read_csv, to_datetime
import statsmodels.api as sm
from statsmodels.formula.api import ols
from numpy import arange
from matplotlib.pyplot import subplots, savefig
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

figures = Path(__file__).parent / "figures"
raw_data = Path(__file__).parent / "data" / "2023IsotopeDataReport-cleanedinexcel.csv"


def levenes_test(df: DataFrame,column: str):
    """
    Levene's test for homogeneity of variances for a given column.

    If p > 0.05, we can assume homogeneity of variances
    """
    monthly = df.groupby("Collection Date")
    june = monthly.get_group(6)
    july = monthly.get_group(7)
    august = monthly.get_group(8)
    september = monthly.get_group(9)
    october = monthly.get_group(10)

    gear_types = data_muscle.groupby("Gear Type")
    cages = gear_types.get_group("C")
    nets = gear_types.get_group("N")
    wild = gear_types.get_group("W")

    a = stats.levene(
        june[column],
        july[column],
        august[column],
        september[column],
        october[column],
    )
    b = stats.levene(cages[column], nets[column], wild[column])
    print(a)
    print(b)


def quantize_categorical_column(
    data: DataFrame, column_name: str, categories: Dict[str, int]
) -> DataFrame:
    """
    Replace strings with integers in a categorical column of a DataFrame,
    using a provided mapping.
    
    Rows with values not in the mapping are replaced with 0 and then
    removed from the DataFrame.

    This process changes the original dataframe.
    """
    for value in data[column_name]:
        replace = categories.get(value, 0)
        data[column_name] = data[column_name].replace(value, replace)
    non_zero_mask = data[column_name] != 0
    return data[non_zero_mask]


# When file is run directly, this block will execute.
# The reason to do this is so that as the methods are wrapped as functions
# and imported into other scripts, this block will not execute, when that
# happens.
if __name__ == "__main__":
    df = read_csv(
        raw_data,
        header=0,
        usecols=[
            "Analysis",
            "Sample ID",
            "Collection Date",
            "Gear Type",
            "Sex",
            "Tissue Type (Gonad or Muscle)",
            "Number in gear type",
            "Mass (mg)",
            "% N",
            "N (umoles)",
            "d15N",
            "%C",
            "C (umoles)",
            "d13C",
            "C/N (Molar)",
            "Date Run",
        ],
    )

    # Remove known 9/6/23 contaminated samples
    for date in range(len(df["Date Run"])):
        if df["Date Run"][date] == "9/6/23":
            df.drop(date, inplace=True)
        else:
            to_datetime(df["Date Run"][date], format="%m/%d/%y")

    # only scallops and filters are being plotted
    df.dropna(subset=["Gear Type"], inplace=True)

    tissue = df.dropna(subset=["Tissue Type (Gonad or Muscle)"])
    mask = tissue["Tissue Type (Gonad or Muscle)"] == "G"
    data_muscle = tissue.drop(tissue[mask].index)

    # Confirm that the data is normally distributed
    data_muscle['% N'].hist()
    data_muscle['d13C'].hist()
    data_muscle['d15N'].hist()
    data_muscle['C/N (Molar)'].hist()
    savefig(f"{figures}/muscle_tissue_histograms.png")

    mask = tissue["Tissue Type (Gonad or Muscle)"] == "M"
    data_gonad = tissue.drop(tissue[mask].index)

    data_muscle_female = data_muscle.drop(data_muscle[data_muscle["Sex"] == "M"].index)
    data_muscle_male = data_muscle.drop(data_muscle[data_muscle["Sex"] == "F"].index)

    data_gonad_female = data_gonad.drop(data_gonad[data_gonad["Sex"] == "M"].index)
    data_gonad_male = data_gonad.drop(data_gonad[data_gonad["Sex"] == "F"].index)

    # In the line below, change data_muscle to data_gonad depending
    # on which tissue you want to analyze
    anova_data = data_muscle[["% N", "Gear Type", "Collection Date"]].dropna()

    # Making columns categorical
    anova_data["Gear Type"] = anova_data["Gear Type"].astype("category")
    anova_data["Collection Date"] = anova_data["Collection Date"].astype("category")

    # Renaming columns
    anova_data = anova_data.rename(
        columns={"% N": "N", "Gear Type": "Gear", "Collection Date": "Month"}
    )

    # Ordinary least squares (OLS) model (ANOVA)
    model = ols("N ~ C(Gear) + C(Month) + C(Gear):C(Month)", data=anova_data).fit()

    # Type II sum of squares
    anova_table = sm.stats.anova_lm(model, type=2)

    print(anova_table)

    levenes_test(data_muscle, "% N")
    levenes_test(data_muscle, "d13C")
    levenes_test(data_muscle, "d15N")
    levenes_test(data_muscle, "C/N (Molar)")

    # Since ANOVA assumptions are not met, try PCA
    df = quantize_categorical_column(df, "Gear Type", {"C": 1, "N": 2, "W": 3})
    df = quantize_categorical_column(df, "Sex", {"F": 1, "M": 2})
    df = quantize_categorical_column(df, "Tissue Type (Gonad or Muscle)", {"G": 1, "M": 2})
    df = df.drop(columns=["Analysis", "Sample ID", "Date Run"])
    # print(df.isna().sum())


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
    ax.plot(pc_numbers, explained_variance, marker='o', linestyle='-')
    ax.set_title('Scree Plot')
    ax.set_xlabel('Principal Component Number')
    ax.set_ylabel('Proportion of Explained Variance')
    ax.grid(True)
    fig.savefig(f"{figures}/pca_scree_plot.png")

    fig, ax = subplots(figsize=(40, 10))
    ax.axis("off")

    # Make a table to summarize the PCA results


    pca.explained_variance_.tolist()
    pca.explained_variance_ratio_.tolist()
    pca.explained_variance_ratio_.cumsum().tolist()

    # summary = [
    #   pca.explained_variance_.round(2),
    #   pca.explained_variance_ratio_.round(2),
    #   pca.explained_variance_ratio_.cumsum().round(2)
    # ]

    # table = ax.table(cellText=summary,
    #                  colLabels=['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10', 'PC11', 'PC12', 'PC13'],
    #                  rowLabels=['Explained Var', 'Explained Var Ratio', 'Cum Explained Var Ratio'],
    #                  cellLoc = 'center',
    #                  rowLoc='center',
    #                  loc='center')
    # plt.show()

    # Look at score plots to visualize how samples relate to each 
    # other in the space defined by the principal components
    fig, ax = subplots(figsize=(10, 8))
    col_name = "Collection Date"  # "Gear Type"
    for dates in df[col_name]:
        ax.scatter(
            components[:, 0], components[:, 1], c=df[col_name], cmap="viridis"
        )
        ax.set_title("PCA Score Plot: PC1 vs PC2")
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)


    fig.savefig(f"{figures}/pca_score_plot.png")
