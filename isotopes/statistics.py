"""
Statistical analysis functions for isotopic data, including ANOVA and FAMD.

This is kept as a separate module to avoid circular imports, since some methods are
used in both the describe and plot modules.
"""
from pathlib import Path
from pandas import DataFrame, read_csv
from prince import FAMD
from numpy import sqrt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from isotopes.options import (
    Dimension,
    TissueType,
    bad_run_dates,
)

def calculate_pca(data: DataFrame) -> tuple[PCA, DataFrame, DataFrame, list[list[float]]]:
    """
    Perform PCA on the muscle tissue data and return the PCA components and explained variance.
    """
    std_scaler = StandardScaler()
    scaled_df = std_scaler.fit_transform(data)
    pca = PCA(n_components=2)
    components = pca.fit_transform(scaled_df)
    explained_variance = pca.explained_variance_ratio_
    summary = [
        pca.explained_variance_.round(2),
        pca.explained_variance_ratio_.round(2),
        pca.explained_variance_ratio_.cumsum().round(2),
    ]
    loadings = pca.components_.T * sqrt(explained_variance)
    return pca, components, loadings, summary


def calculate_famd(partition: DataFrame, components: int):
    """
    Perform Factor Analysis of Mixed Data (FAMD). This will be used both for 
    visualiation and summary, statistics.
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


def partition_data_by_tissue(
    filepath: Path, usecols: list[Dimension], tissue_type: TissueType
) -> DataFrame:
    """
    Partition the data by tissue type. This is useful for performing statistical analysis
    on each tissue type separately.
    """
    filter_columns = [
        Dimension.TISSUE.value,
        Dimension.DATE_RUN.value
    ]
    df = read_csv(
        filepath,
        header=0,
        usecols= [
            *([col.value for col in usecols]),
            *filter_columns,
        ],
    )
    # Remove known bad samples, and select only one tissue type for analysis
    mask = (~df[Dimension.DATE_RUN.value].isin(bad_run_dates)) & (
        df[Dimension.TISSUE.value] == tissue_type.value
    )
    return df[mask].drop(columns=filter_columns)