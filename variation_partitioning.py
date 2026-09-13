"""
Relative contributions of temperature, light, and culture method
in explaining the variation in stable isotope ratios of scallop tissues
"""

from datetime import date, timedelta
from enum import StrEnum, auto
from pathlib import Path

# from warnings import catch_warnings, simplefilter
from numpy import arange, zeros, nan, array, var
from matplotlib.pyplot import subplots, close
from pandas import DataFrame, to_datetime, read_csv, get_dummies
from statsmodels.api import add_constant
from statsmodels.formula.api import ols, mixedlm
from statsmodels.regression.mixed_linear_model import MixedLMResultsWrapper
from statsmodels.stats.outliers_influence import variance_inflation_factor

# from statsmodels.tools.sm_exceptions import ConvergenceWarning
from isotopes.statistics import partition_data_by_tissue
from isotopes.options import (
    Dimension,
    TissueType,
    isotopes_no_outliers,
    CultureMethod,
    EnvDimension,
    env_data,
    figures,
)


class DerivedDimension(StrEnum):
    """Synthetic columns appended for analysis"""

    TEMPERATURE = auto()
    LIGHT = auto()
    DATE = "Collection_Date_Group"


class Partitions(StrEnum):
    """Partitioning components"""

    GEAR = "Gear"
    ENVIRONMENT = "Environment"
    SHARED = "Gear & Environment"
    UNEXPLAINED = "Unexplained"


collection_date_map = {
    6: date(2023, 6, 15),
    7: date(2023, 7, 13),
    8: date(2023, 8, 14),
    9: date(2023, 9, 15),
    10: date(2023, 10, 12),
}

temperature_column = {
    CultureMethod.CAGE: EnvDimension.CAGE_TEMP,
    CultureMethod.NET: EnvDimension.NET_BOTTOM_TEMP,
    CultureMethod.WILD: EnvDimension.WILD_TEMP,
}
light_column = {
    CultureMethod.CAGE: EnvDimension.CAGE_LUM,
    CultureMethod.NET: EnvDimension.NET_BOTTOM_LUM,
}


def variation_partitioning_ols(
    data: DataFrame, response: Dimension, dims: list[DerivedDimension]
):
    """
    Variation partitioning between:
        Environment
        Gear

    Four models are fitted:
        Environment
        Gear
        Full
        Null

    The resulting fractions are:
        Unique environment
        Unique gear
        Shared environment + gear
        Unexplained
    """
    count = len(data)

    env_formula = f'Q("{response}") ~ {" + ".join(dims)}'
    env_model = ols(formula=env_formula, data=data).fit()

    gear_formula = f'Q("{response}") ~ C(Q("{Dimension.GEAR}"))'
    gear_model = ols(formula=gear_formula, data=data).fit()

    full_formula = f'Q("{response}") ~ {" + ".join(dims)} + C(Q("{Dimension.GEAR}"))'
    full_model = ols(formula=full_formula, data=data).fit()

    null_formula = f'Q("{response}") ~ 1'
    null_model = ols(formula=null_formula, data=data).fit()

    components = {
        Partitions.ENVIRONMENT: full_model.rsquared - gear_model.rsquared,
        Partitions.GEAR: full_model.rsquared - env_model.rsquared,
        Partitions.SHARED: env_model.rsquared
        + gear_model.rsquared
        - full_model.rsquared,
        Partitions.UNEXPLAINED: 1 - full_model.rsquared,
    }

    print(f"\n{response} (N={count})\n")
    print(f"{'R² environment:':30s}{env_model.rsquared:.4f}")
    print(f"{'R² gear:':30s}{gear_model.rsquared:.4f}")
    print(f"{'R² full:':30s}{full_model.rsquared:.4f}")
    print(f"{'R² null:':30s}{null_model.rsquared:.4f}")
    print()
    for key, value in components.items():
        token = key + ":"
        print(f"{token:30s}{value:.4f}")
    print(f"{'Total:':30s}{sum(components.values()):.4f}")

    return {
        "Response": response,
        "N": count,
        "Environment R2": env_model.rsquared,
        "Gear R2": gear_model.rsquared,
        "Full R2": full_model.rsquared,
        **components,
    }


def plot_partitions(response_df: DataFrame, title: str, outfile: Path):
    """
    Create stacked bar chart of variation partitioning results.
    """
    plot_colors = {
        Partitions.GEAR: "black",
        Partitions.SHARED: "red",
        Partitions.ENVIRONMENT: "blue",
    }
    fig, ax = subplots(figsize=(3, 4))
    x_positions = arange(len(response_df))
    bottom = zeros(len(response_df))
    for component, color in plot_colors.items():
        values = array(response_df[component].values)
        ax.bar(
            x_positions,
            values,
            bottom=bottom,
            label=component,
            color=color,
            edgecolor="black",
            linewidth=0.5,
        )
        bottom += values

    ax.set_xticks(x_positions, rotation=45, labels=response_df["Response"].values)
    ax.set_ylabel("Variation")
    ax.set_xlabel("Response")
    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=12)
    ax.legend(frameon=False, loc="best", reverse=True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(
        outfile,
        dpi=300,
        bbox_inches="tight",
    )
    close(fig)



def mixed_model_r2(result: MixedLMResultsWrapper):
    """
    Calculate approximate marginal and conditional R²
    for a random-intercept MixedLM.

    Marginal R²:
        variation explained by fixed effects

    Conditional R²:
        variation explained by fixed + random effects
    """
    fixed_prediction = result.model.exog @ result.fe_params
    var_fixed = var(fixed_prediction, ddof=1)
    var_random = result.cov_re.iloc[0, 0]
    var_residual = result.scale
    denominator = var_fixed + var_random + var_residual
    marginal_r2 = var_fixed / denominator
    conditional_r2 = (var_fixed + var_random) / denominator
    return (marginal_r2, conditional_r2)


def variation_partitioning_lmm(
    data: DataFrame,
    response: Dimension, 
    dims: list[DerivedDimension],
    reml: bool = False,
    method: str = "lbfgs",
    maxiter: int = 2000,
    disp: bool = False,
):
    """
    Variation partitioning using linear mixed models.

    Random effect:
        Collection_Date_Group

    Fixed-effect groups:
        Environment
        Gear

    Four models are fitted:
        Null
        Environment
        Gear
        Full

    Partitioning:
        Unique environment = Full R² - Gear R²
        Unique gear = Full R² - Environment R²
        Shared = Environment R² + Gear R² - Full R²
        Unexplained =  1 - Full R²
    """
    
    count = len(data)

    environment_formula = (
        f'Q("{response}") '
        f"~ {' + '.join(dims)}"
    )
    kwargs = {
        "reml": reml,
        "method": method,
        "maxiter": maxiter,
        "disp": disp
    }
    # pylint: disable=unexpected-keyword-arg
    environment_model = mixedlm(
        formula=environment_formula,
        data=data,
        groups=DerivedDimension.DATE
    ).fit(**kwargs)
    environment_marginal, environment_conditional = mixed_model_r2(environment_model)

    gear_formula = f'Q("{response}") ' f"~ C(Q('{Dimension.GEAR}'))"
    gear_model = mixedlm(
        formula=gear_formula,
        data=data,
        groups=DerivedDimension.DATE
    ).fit(**kwargs)
    gear_marginal, gear_conditional = mixed_model_r2(gear_model)

    full_formula = (
        f'Q("{response}") '
        f"~ {' + '.join(dims)} + C(Q('{Dimension.GEAR}'))"
    )
    full_model = mixedlm(
        formula=full_formula, data=data, groups=DerivedDimension.DATE
    ).fit(**kwargs)
    full_marginal, full_conditional = mixed_model_r2(full_model)

    null_formula = f'Q("{response}") ~ 1'
    null_model = mixedlm(
        formula=null_formula, data=data, groups=DerivedDimension.DATE
    ).fit(**kwargs)
    null_marginal, null_conditional = mixed_model_r2(null_model)

    components = {
        Partitions.ENVIRONMENT: full_marginal - gear_marginal,
        Partitions.GEAR: full_marginal - environment_marginal,
        Partitions.SHARED: environment_marginal + gear_marginal - full_marginal,
        Partitions.UNEXPLAINED: 1 - full_marginal,
    }

    print(f"\n{response} (N={count})\n")
    print(f"{'R² environment:':30s}{environment_marginal:.4f}")
    print(f"{'R² gear:':30s}{gear_marginal:.4f}")
    print(f"{'R² full:':30s}{full_marginal:.4f}")
    print(f"{'R² null:':30s}{null_marginal:.4f}")
    print(f"{'R² full (conditional):':30s}{full_conditional:.4f}")
    print()
    for key, value in components.items():
        token = key + ":"
        print(f"{token:30s}{value:.4f}")
    print(f"{'Total:':30s}{sum(components.values()):.4f}")

    return {
        "Response": response,
        "N": count,
        "Environment R2": environment_marginal,
        "Gear R2": gear_marginal,
        "Full R2": full_marginal,
        "Full conditional R2": full_conditional,
        **components
    }


def run_analysis(env_dims: list[DerivedDimension]):

    response_vars = [
        Dimension.CARBON_FRACTIONATION,
        Dimension.NITROGEN_FRACTIONATION,
        Dimension.MOLAR_RATIO,
    ]
    df = partition_data_by_tissue(
        filepath=isotopes_no_outliers,
        usecols=[
            Dimension.COLLECTION_DATE,
            Dimension.GEAR,
            *response_vars,
        ],
        tissue_type=TissueType.MUSCLE,
    ).dropna()

    # Add columns to populate with derived or environmental data
    df[DerivedDimension.TEMPERATURE] = nan
    df[DerivedDimension.LIGHT] = nan
    df["Collection_Date_Group"] = None

    env = read_csv(env_data)
    env[EnvDimension.DATE] = to_datetime(
        env[EnvDimension.DATE], format="%m/%d/%y %H:%M"
    )
    env_dates = env[EnvDimension.DATE].dt.date
    groupby = df.groupby([Dimension.GEAR, Dimension.COLLECTION_DATE])
    # Each group of method and collection date will share the same
    # environmental conditions
    for (gear, month), indx in groupby.groups.items():
        # Aggregate preceding 30 days of environmental data
        end_date = collection_date_map[int(month)]
        start_date = end_date - timedelta(days=30)
        env_subset = env[(env_dates >= start_date) & (env_dates <= end_date)]
        df.loc[indx, DerivedDimension.TEMPERATURE] = env_subset[
            temperature_column[gear]
        ].mean()
        df.loc[indx, DerivedDimension.LIGHT] = (
            env_subset[light_column[gear]].mean() if gear in light_column else nan
        )
        df.loc[indx, "Collection_Date_Group"] = end_date

    df = df.dropna(subset=env_dims)

    # Check for multi-collinearity using Variance Inflation Factor
    # Drop first to prevent dummy variable trap, where each is `inf`
    X: DataFrame = get_dummies(
        df[
            [
                *env_dims,
                Dimension.GEAR,
            ]
        ],
        drop_first=True,  # prevent dummy variable trap, where each is `inf`
        dtype=int,
    )
    X: DataFrame = add_constant(X)
    vif = DataFrame(
        {
            "Variable": X.columns,
            "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])],
        }
    )
    print("Variance Inflation Factor (VIF):")
    print(vif)

    for r in response_vars:
        result = mixedlm(
            f'Q("{r}") ~ {" + ".join(env_dims)} + C(Q("{Dimension.GEAR}"))',
            data=df,
            groups=df[Dimension.COLLECTION_DATE],
        ).fit()
        print(result.summary())

    analysis_var: list[str] = [
        each.lower() for each in [*env_dims, Dimension.GEAR]
    ]
    print(f"\nVARIATION PARTITIONING: {' + '.join(analysis_var)}")
    print("-" * 80)
    results_df_ols = DataFrame(
        [
            variation_partitioning_ols(df, r, dims=env_dims)
            for r in response_vars
        ]
    )
    results_df_mlm = DataFrame(
        [
            variation_partitioning_lmm(df, r, dims=env_dims)
            for r in response_vars
        ]
    )
    file_suffix = "_".join(analysis_var).replace(" ", "_")
    plot_partitions(
        results_df_ols,
        title=f"{' + '.join(analysis_var)}",
        outfile=figures / f"variation_partitioning_{file_suffix}.png",
    )
    plot_partitions(
        results_df_mlm,
        title=f"{' + '.join(analysis_var)}\n(random effect: collection date)",
        outfile=figures / f"variation_partitioning_{file_suffix}_random_date.png",
    )

if __name__ == "__main__":
    figures.mkdir(parents=True, exist_ok=True)
    run_analysis([DerivedDimension.TEMPERATURE, DerivedDimension.LIGHT])
    run_analysis([DerivedDimension.TEMPERATURE])
