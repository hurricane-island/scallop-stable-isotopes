"""
Command line interface for statistical analysis and data summarization of isotopic data.

This module should use the local statistics module, and mostly have commands here.
"""
from enum import Enum
from calendar import month_name
from pandas import read_csv, Series, to_datetime
from numpy import absolute, asarray
from scipy.stats import levene, zscore
from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols
from click import option, group, Choice
from isotopes.options import (
    Dimension,
    EnvDimension,
    TissueType,
    Command,
    tissue_type_option,
    bad_run_dates,
    isotopes_no_outliers,
    env_data,
)
from isotopes.statistics import partition_data_by_tissue, calculate_pca

class DescribeGroupCommand(Enum):
    """
    Group of commands for statistical analysis and data summarization of isotopic data.
    """

    ANOVA = "anova"
    LEVENES = "levenes"
    OUTLIERS = "outliers"
    TEMPERATURE = "temperature"

@group()
def describe():
    """
    Perform statistical analysis and summarize raw data. The functionality is divided into
    commands for `tissue` and `environmental` data.
    """

@group()
def tissue():
    """
    Summarize isotopic data, including tables of averages and standard deviations.
    """

@group()
def environment():
    """
    Summarize environmental data, specifically temperature effects and patterns.
    """

describe.add_command(tissue)
describe.add_command(environment)


@tissue.command(DescribeGroupCommand.ANOVA.value)
@tissue_type_option(TissueType.MUSCLE)
def describe_tissue_analysis_of_variance(tissue_type: TissueType):
    """
    Summarize Analysis of Variance (ANOVA) for a given tissue type. This is only valid
    if the assumptions of ANOVA are met, which can be checked using Levene's test for
    homogeneity of variances.

    Can only be run for a single variable at a time.

    Because tissue types have an effect, it only makes sense to run this on a single
    tissue partition. Generally this will be muscle.
    """
    analysis = Dimension.NITROGEN_PERCENTAGE
    df = partition_data_by_tissue(
        isotopes_no_outliers,
        [
            Dimension.COLLECTION_DATE,
            Dimension.GEAR,
            analysis,
        ],
        tissue_type,
    )
    # Patsy syntax: C() = categorical, Q() = wrap special characters
    formula = (
        f"Q('{analysis.value}') ~ "
        f"C(Q('{Dimension.GEAR.value}')) + "
        f"C(Q('{Dimension.COLLECTION_DATE.value}')) + "
        f"C(Q('{Dimension.GEAR.value}')):C(Q('{Dimension.COLLECTION_DATE.value}'))"
    )
    model = ols(formula, data=df.dropna()).fit()
    result = anova_lm(model, type=2)  # Type II sum of squares
    print(
        (
            f"Analysis of Variance\n"
            f"Tissue: {tissue_type.name.lower()}\n"
            f"Variable: {analysis.value}\n"
        )
    )
    print(result)


@tissue.command(DescribeGroupCommand.LEVENES.value)
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
    type=Choice(
        [
            Dimension.NITROGEN_FRACTIONATION,
            Dimension.CARBON_FRACTIONATION,
            Dimension.MOLAR_RATIO,
        ],
        case_sensitive=False,
    ),
)
def describe_tissue_levenes_test(
    tissue_type: TissueType, group_by: Dimension, variable: Dimension
):
    """
    Summarize Levene's test for homogeneity of variances. This is used to determine whether the assumptions of ANOVA are met for a given tissue type. If the p-value is greater than 0.05, we can assume homogeneity of variances and proceed with ANOVA.
    """
    groups = (
        partition_data_by_tissue(
            isotopes_no_outliers,
            [
                group_by,
                variable,
            ],
            tissue_type,
        )
        .dropna()
        .groupby(group_by.value)
        .agg(list)
        .get(variable.value)
        .to_dict() # type: ignore
        .values()
    )
    result = levene(*groups)
    print(
        (
            f"\nLevene's Test of Homogeneity of Variance"
            f"\nTissue: {tissue_type.name.lower()}"
            f"\nDimension: {variable.name.lower()} ({variable.value})"
            f"\nGroup by: {group_by.name}"
            f"\nResult: {result.statistic}"
            f"\nP-value: {result.pvalue}"
            f"\nHomogenous: {result.pvalue > 0.05}"
        )
    )

@tissue.command(DescribeGroupCommand.OUTLIERS.value)
@tissue_type_option(TissueType.MUSCLE)
@option(
    "--variable",
    required=True,
    help="Dimension to test for outliers.",
    type=Choice(
        [
            Dimension.NITROGEN_FRACTIONATION,
            Dimension.CARBON_FRACTIONATION,
            Dimension.MOLAR_RATIO,
        ],
        case_sensitive=False,
    ),
)
def describe_tissue_outliers(
    tissue_type: TissueType,
    variable: Dimension,
):
    """
    CHECKING FOR OUTLIERS in DATASET WITH A Z-SCORE GREATER THAN 3
    """
    df = partition_data_by_tissue(
        isotopes_no_outliers,
        [
            variable,
        ],
        tissue_type,
    ).dropna()
    z = asarray(zscore(df[variable.value]))  # type coercion to suppress error
    result = df.loc[absolute(z) > 3]
    print(result)


@tissue.command(Command.GEAR_TYPE.value)
def describe_tissue_by_gear_type(
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
    agg_map = {key: ["mean", "std"] for key in analyze}
    df = read_csv(
        isotopes_no_outliers,
        header=0,
        usecols=[
            *group_by,
            *analyze,
            Dimension.DATE_RUN.value,
        ],
    ).dropna()
    mask = ~df[Dimension.DATE_RUN.value].isin(bad_run_dates) & (
        df[Dimension.TISSUE.value].isin(["M", "G"])
    )
    groups = (
        df[mask]
        .drop(columns=[Dimension.DATE_RUN.value])
        .groupby(group_by)
        .agg(agg_map)
        .round(2)
    )
    print(groups)


@environment.command(DescribeGroupCommand.TEMPERATURE.value)
@option(
    "--threshold",
    default=55.4,
    help="Count observations above this threshold.",
)
def describe_environment_temperature(
    threshold: float
):
    """
    Calculate mean environmental variable and total observations above a threshold, 
    and save as a pivot table. The rows are the culture method and statistic, 
    and the columns are the months.
    """

    def test(x: Series) -> int:
        """Aggregate function for observations above the threshold"""
        return int((x > threshold).sum())
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
    groups = (
        df
        .drop(columns=[EnvDimension.DATE.value])
        .groupby("Month")
        .aggregate(["mean", test])
        .round(2)
    )
    groups.index = [month_name[int(each)] for each in groups.index]
    for col in groups.columns:
        if col[1] == test.__name__:
            groups[col] = groups[col].astype(int)

    print(groups)


@tissue.command("pca")
def isotopes_describe_pca_clustering():
    """
    Perform PCA clustering analysis on the muscle tissue data.
    """
    df = partition_data_by_tissue(
        isotopes_no_outliers,
        [
            Dimension.GEAR,
            Dimension.NITROGEN_FRACTIONATION,
            Dimension.CARBON_FRACTIONATION,
            Dimension.MOLAR_RATIO,
        ],
        TissueType.MUSCLE
    ).dropna()
    pca_df = df[
        [
            Dimension.NITROGEN_FRACTIONATION.value,
            Dimension.CARBON_FRACTIONATION.value,
            Dimension.MOLAR_RATIO.value,
        ]
    ]
    pca, components, loadings, summary = calculate_pca(pca_df)
    print(summary)
