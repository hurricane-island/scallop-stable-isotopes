"""
Relative contributions of temperature, light, and culture method
in explaining the variation in stable isotope ratios of scallop tissues
"""
from datetime import date, timedelta
from enum import Enum
from numpy import arange, zeros, nan, array
from matplotlib.pyplot import subplots, close
from pandas import DataFrame, to_datetime, read_csv, get_dummies, concat, to_numeric
import statsmodels.api as sm
from statsmodels.formula.api import ols, mixedlm
from statsmodels.regression.mixed_linear_model import MixedLMResultsWrapper
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.sm_exceptions import ConvergenceWarning
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
import numpy as np
from warnings import catch_warnings, simplefilter


class DerivedDimension(Enum):
    """Synthetic columns appended for analysis"""

    TEMPERATURE = "temperature"
    LIGHT = "light"


collection_date_map = {
    6: date(2023, 6, 15),
    7: date(2023, 7, 13),
    8: date(2023, 8, 14),
    9: date(2023, 9, 15),
    10: date(2023, 10, 12),
}

temperature_column = {
    CultureMethod.CAGE.value: EnvDimension.CAGE_TEMP.value,
    CultureMethod.NET.value: EnvDimension.NET_BOTTOM_TEMP.value,
    CultureMethod.WILD.value: EnvDimension.WILD_TEMP.value,
}
light_column = {
    CultureMethod.CAGE.value: EnvDimension.CAGE_LUM.value,
    CultureMethod.NET.value: EnvDimension.NET_BOTTOM_LUM.value,
}


def lmm_results_table(result: MixedLMResultsWrapper, response_name: Dimension):
    """Results into table format"""
    _table = DataFrame(
        {
            "Response": response_name.value,
            "Predictor": result.params.index,
            "Estimate": result.params.values,
            "SE": result.bse.values,
            "z-value": result.tvalues.values,
            "p-value": result.pvalues.values,
            "95% CI Lower": result.conf_int()[0],
            "95% CI Upper": result.conf_int()[1],
        }
    )
    # Remove intercept and random effect variance
    return _table[~_table["Predictor"].isin(["Intercept", "Group Var"])]


def mixedlm_analysis(data: DataFrame, dims: list[DerivedDimension], groups: Dimension):
    """
    Linear Mixed Models: Run for each isotopic variance
    """
    lookup = {}
    str_values = [dim.value for dim in dims]
    for response in [
        Dimension.CARBON_FRACTIONATION,
        Dimension.NITROGEN_FRACTIONATION,
        Dimension.MOLAR_RATIO,
    ]:
        result = mixedlm(
            f'Q("{response.value}") ~ {" + ".join(str_values)} + C(Q("{Dimension.GEAR.value}"))',
            data=data,
            groups=data[groups.value],
        ).fit()
        print(result.summary())
        lookup[response] = result

    light_table = concat([lmm_results_table(val, key) for key, val in lookup.items()])
    light_table.to_csv(
        figures / f"lmm_{'_'.join(str_values).lower()}_table.csv", index=False
    )


def variation_partitioning(
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
    str_values = [dim.value for dim in dims]
    subset = data.dropna(subset=[response.value, Dimension.GEAR.value, *str_values])

    env_formula = f'Q("{response.value}") ~ {" + ".join(str_values)}'
    env_model = ols(formula=env_formula, data=subset).fit()

    gear_formula = f'Q("{response.value}") ~ C(Q("{Dimension.GEAR.value}"))'
    gear_model = ols(formula=gear_formula, data=subset).fit()

    full_formula = f'Q("{response.value}") ~ {" + ".join(str_values)} + C(Q("{Dimension.GEAR.value}"))'
    full_model = ols(formula=full_formula, data=subset).fit()

    null_formula = f'Q("{response.value}") ~ 1'
    null_model = ols(formula=null_formula, data=subset).fit()

    components = {
        "Unique environment": full_model.rsquared - gear_model.rsquared,
        "Unique gear": full_model.rsquared - env_model.rsquared,
        "Shared environment + gear": env_model.rsquared
        + gear_model.rsquared
        - full_model.rsquared,
        "Unexplained": 1 - full_model.rsquared,
    }

    print(f"\n{response} (N={len(subset)})\n")
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
        "Response": response.value,
        "N": len(subset),
        "Environment R2": env_model.rsquared,
        "Gear R2": gear_model.rsquared,
        "Full R2": full_model.rsquared,
        **components,
    }


def make_partition_plot(results_df, output_file, title):

    plot_colors = {
        "Unique environment": "#2C7FB8",
        "Shared environment + gear": "#7FCDBB",
        "Unique gear": "#F03B20",
        "Unexplained": "#D9D9D9",
    }
    fig, ax = subplots(figsize=(9, 6))
    x_positions = np.arange(len(results_df))
    bottom = np.zeros(len(results_df))
    for component, color in plot_colors.items():
        values = results_df[component].values * 100
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
    ax.set_xticks(x_positions)
    ax.set_xticklabels(["δ¹³C", "δ¹⁵N", "C/N"])
    ax.set_ylabel("Variation explained (%)")
    ax.set_xlabel("Response variable")
    ax.set_ylim(0, 100)
    ax.set_title(title, fontsize=14)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    close(fig)

def plot_partitions(
    response_df: DataFrame, parts: list[str], responses: list[Dimension]
):
    """
    Create stacked bar chart of variation partitioning results.
    """
    plot_colors = {
        "Unique environment": "#2C7FB8",
        "Shared environment + gear": "#7FCDBB",
        "Unique gear": "#F03B20",
        "Unexplained": "#D9D9D9",
    }
    fig, ax = subplots(figsize=(4, 4))
    x_positions = arange(len(responses))
    bottom = zeros(len(responses))

    for component, color in plot_colors.items():
        values = []
        for each in responses:
            value = response_df.loc[
                response_df["Response"] == each.value, component
            ].iloc[0]
            values.append(value * 100)
        ax.bar(
            x_positions,
            values,
            bottom=bottom,
            label=component,
            color=color,
            edgecolor="black",
            linewidth=0.5,
        )
        bottom += array(values)

    ax.set_xticks(x_positions)
    ax.set_xticklabels([r.value for r in responses])
    ax.set_ylabel("Variation explained (%)")
    ax.set_xlabel("Response variable")
    ax.set_ylim(0, 100)
    ax.set_title(f"Variation partitioning: {' + '.join(parts)}", fontsize=14)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(
        figures / f"variation_partitioning_{'_'.join(parts)}.png",
        dpi=300,
        bbox_inches="tight",
    )
    close(fig)


def run_analysis(
    df: DataFrame, subset: list[DerivedDimension], responses: list[Dimension]
):
    """
    Run variation partitioning analysis.
    """
    analysis_var: list[str] = [each.value.lower() for each in [*subset, Dimension.GEAR]]
    print(f"\nVARIATION PARTITIONING: {' + '.join(analysis_var)}")
    print("-" * 80)
    results_df = DataFrame(
        [variation_partitioning(df, r, dims=subset) for r in responses]
    )
    plot_partitions(results_df, analysis_var, responses)




def fit_lmm(data, formula, response):
    """
    Fit a linear mixed model with collection date
    as the random intercept.

    Random effect:
        Collection_Date_Group

    Fixed effects are defined by the formula.
    """

    model = mixedlm(
        formula=formula, data=data, groups=data["Collection_Date_Group"]
    )

    # Try several optimizers because MixedLM can sometimes
    # have difficulty converging.
    optimizers = ["lbfgs", "powell", "bfgs", "cg"]
    last_error = None
    for optimizer in optimizers:
        try:
            with catch_warnings():
                simplefilter("ignore", ConvergenceWarning)
                result = model.fit(
                    reml=False, method=optimizer, maxiter=2000, disp=False
                )
            if result.converged:

                print(f"    Successful optimizer: " f"{optimizer}")
                return result
            last_error = f"{optimizer} did not converge"
        except Exception as e:

            last_error = f"{optimizer}: {e}"
    raise RuntimeError(
        f"\nMixed model failed for "
        f"{response}.\n"
        f"Formula: {formula}\n"
        f"Last error: {last_error}"
    )


def mixed_model_r2(result):
    """
    Calculate approximate marginal and conditional R²
    for a random-intercept MixedLM.

    Marginal R²:
        variation explained by fixed effects

    Conditional R²:
        variation explained by fixed + random effects
    """
    fixed_prediction = result.model.exog @ result.fe_params
    var_fixed = np.var(fixed_prediction, ddof=1)

    # Random-effect variance
    try:
        var_random = float(result.cov_re.iloc[0, 0])
    except Exception:
        var_random = 0.0

    # Residual variance
    var_residual = result.scale
    denominator = var_fixed + var_random + var_residual
    if denominator <= 0:

        return np.nan, np.nan

    marginal_r2 = var_fixed / denominator
    conditional_r2 = (var_fixed + var_random) / denominator
    return (marginal_r2, conditional_r2)


def variation_partitioning_lmm(data, response, include_light=True):
    """
    Variation partitioning using linear mixed models.

    Random effect:
        Collection_Date_Group

    Fixed-effect groups:

        Environment =
            Temperature (+ Light)

        Gear =
            Categorical gear

    Four models are fitted:

        1. Null
        2. Environment
        3. Gear
        4. Full

    Partitioning:

        Unique environment =
            Full R² - Gear R²

        Unique gear =
            Full R² - Environment R²

        Shared =
            Environment R²
            + Gear R²
            - Full R²

        Unexplained =
            1 - Full R²

    """

    required = [response, "Temperature", Dimension.GEAR.value, "Collection_Date_Group"]
    if include_light:
        required.append("Light")
    subset = data.dropna(subset=required).copy()
    # Remove unused category levels

    subset[Dimension.GEAR.value] = subset[Dimension.GEAR.value].cat.remove_unused_categories()
    subset["Collection_Date_Group"] = subset[
        "Collection_Date_Group"
    ].cat.remove_unused_categories()

    n_dates = subset["Collection_Date_Group"].nunique()
    if n_dates < 2:

        raise ValueError(
            f"Only {n_dates} collection date "
            f"group was found for {response}. "
            "At least two collection dates are "
            "required for a random effect."
        )

    if include_light:
        environment_formula = f'Q("{response}") ' f"~ Temperature + Light"
        full_formula = f'Q("{response}") ' f"~ Temperature + Light + C(Q('{Dimension.GEAR.value}'))"
    else:
        environment_formula = f'Q("{response}") ' f"~ Temperature"
        full_formula = f'Q("{response}") ' f"~ Temperature + C(Q('{Dimension.GEAR.value}'))"
    gear_formula = f'Q("{response}") ' f"~ C(Q('{Dimension.GEAR.value}'))"
    null_formula = f'Q("{response}") ~ 1'

    print("\n" + "=" * 70)

    if include_light:
        print(f"VARIATION PARTITIONING: " f"{response} — TEMPERATURE + LIGHT")
    else:
        print(f"VARIATION PARTITIONING: " f"{response} — TEMPERATURE ONLY")
    print("=" * 70)
    print(f"Samples: {len(subset)}")
    print(f"Collection dates: {n_dates}")
    print("\nCollection-date groups:")
    print(subset["Collection_Date_Group"].value_counts().sort_index())

    null_model = fit_lmm(subset, null_formula, response)
    environment_model = fit_lmm(subset, environment_formula, response)
    gear_model = fit_lmm(subset, gear_formula, response)
    full_model = fit_lmm(subset, full_formula, response)

    # R²

    environment_marginal, environment_conditional = mixed_model_r2(environment_model)
    gear_marginal, gear_conditional = mixed_model_r2(gear_model)
    full_marginal, full_conditional = mixed_model_r2(full_model)
    null_marginal, null_conditional = mixed_model_r2(null_model)

    # PARTITIONING

    unique_environment = full_marginal - gear_marginal
    unique_gear = full_marginal - environment_marginal
    shared = environment_marginal + gear_marginal - full_marginal
    unexplained = 1 - full_marginal

    # PRINT RESULTS

    print("\nR² values:")
    print(f"Environment marginal R²: " f"{environment_marginal:.4f}")
    print(f"Gear marginal R²:        " f"{gear_marginal:.4f}")
    print(f"Full marginal R²:        " f"{full_marginal:.4f}")
    print(f"Full conditional R²:     " f"{full_conditional:.4f}")
    print("\nVariation partitioning:")
    print(
        f"Unique environment: "
        f"{unique_environment:.4f} "
        f"({unique_environment * 100:.2f}%)"
    )
    print(f"Unique gear: " f"{unique_gear:.4f} " f"({unique_gear * 100:.2f}%)")
    print(f"Shared environment + gear: " f"{shared:.4f} " f"({shared * 100:.2f}%)")
    print(f"Unexplained: " f"{unexplained:.4f} " f"({unexplained * 100:.2f}%)")
    print(
        "\nComponents sum: "
        f"{unique_environment + unique_gear + shared + unexplained:.4f}"
    )


    try:
        random_variance = float(full_model.cov_re.iloc[0, 0])
    except Exception:
        random_variance = np.nan

    residual_variance = full_model.scale
    print("\nFull model variance components:")
    print(f"Collection-date variance: " f"{random_variance:.6f}")
    print(f"Residual variance: " f"{residual_variance:.6f}")

    results = {
        "Response": response,
        "N": len(subset),
        "Collection dates": n_dates,
        "Environment marginal R2": environment_marginal,
        "Gear marginal R2": gear_marginal,
        "Full marginal R2": full_marginal,
        "Full conditional R2": full_conditional,
        "Unique environment": unique_environment,
        "Unique gear": unique_gear,
        "Shared environment + gear": shared,
        "Unexplained": unexplained,
        "Collection date variance": random_variance,
        "Residual variance": residual_variance,
    }
    models = {
        "null": null_model,
        "environment": environment_model,
        "gear": gear_model,
        "full": full_model,
    }
    return (results, models)



if __name__ == "__main__":
    # VIF

    figures.mkdir(parents=True, exist_ok=True)
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
    
    df["Collection_Date_Code"] = to_numeric(
        df["Collection Date"], errors="coerce"
    )
    df["Collection_Date_Group"] = df["Collection_Date_Code"].map(
        collection_date_map
    )
    df["Collection_Date_Group"] = to_datetime(
        df["Collection_Date_Group"], errors="coerce"
    )
    df = df.dropna(subset=["Collection_Date_Group"])
    df["Month"] = df["Collection_Date_Group"].dt.month
    print("\nCollection dates:")
    print(df["Collection_Date_Group"].value_counts().sort_index())
    print("\nCollection months:")
    print(df["Month"].value_counts().sort_index())


    # Add columns to populate with environmental data
    df[DerivedDimension.TEMPERATURE.value] = nan
    df[DerivedDimension.LIGHT.value] = nan

    env = read_csv(env_data)
    env[EnvDimension.DATE.value] = to_datetime(env[EnvDimension.DATE.value], errors="coerce")
    env_dates = env[EnvDimension.DATE.value].dt.date

    print(df.columns)
    groupby = df.groupby([Dimension.GEAR.value, Dimension.COLLECTION_DATE.value])
    # Each group of method and collection date will share the same
    # environmental conditions
    for (gear, month), indx in groupby.groups.items():
        # Aggregate preceding 30 days of environmental data
        end_date = collection_date_map[int(month)]
        start_date = end_date - timedelta(days=30)
        env_subset = env[(env_dates >= start_date) & (env_dates <= end_date)]
        df.loc[indx, DerivedDimension.TEMPERATURE.value] = env_subset[
            temperature_column[gear]
        ].mean()
        df.loc[indx, DerivedDimension.LIGHT.value] = (
            env_subset[light_column[gear]].mean() if gear in light_column else nan
        )

    df_with_light = df.dropna(
        subset=[DerivedDimension.TEMPERATURE.value, DerivedDimension.LIGHT.value]
    )
    df_temp_only = df.dropna(subset=[DerivedDimension.TEMPERATURE.value])

    # VIF: Check for multicollinearity using Variance Inflation Factor (VIF)
    X = get_dummies(
        df_with_light[
            [
                DerivedDimension.TEMPERATURE.value,
                DerivedDimension.LIGHT.value,
                Dimension.GEAR.value,
            ]
        ],
        drop_first=True,
        dtype=float,
    )
    X: DataFrame = sm.add_constant(X)
    vif = DataFrame(
        {
            "Variable": X.columns,
            "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])],
        }
    )
    vif.to_csv(figures / "vif_results.csv", index=False)
    mixedlm_analysis(
        df_with_light,
        [DerivedDimension.TEMPERATURE, DerivedDimension.LIGHT],
        Dimension.COLLECTION_DATE,
    )
    mixedlm_analysis(
        df_temp_only, [DerivedDimension.TEMPERATURE], Dimension.COLLECTION_DATE
    )

    # ANALYSIS: TEMPERATURE + LIGHT + GEAR
    # Wild scallops are NOT included because they have no light.
    run_analysis(
        df_with_light,
        [DerivedDimension.TEMPERATURE, DerivedDimension.LIGHT],
        response_vars,
    )

    # ANALYSIS: TEMPERATURE + GEAR
    # Wild scallops ARE included.
    run_analysis(
        df_temp_only,
        [DerivedDimension.TEMPERATURE],
        response_vars,
    )


    analysis_data[Dimension.GEAR.value] = analysis_data[Dimension.GEAR.value].astype("category")

    # Make sure date group is categorical

    analysis_data["Collection_Date_Group"] = analysis_data["Collection_Date_Group"].astype(
        "category"
    )
    print("\nEnvironmental data availability:")
    print(analysis_data[["Temperature", "Light"]].isna().sum())
    print("\nGear counts after environmental merge:")
    print(analysis_data[Dimension.GEAR.value].value_counts())
    print("\nCollection dates used as random-effect groups:")

    print(analysis_data["Collection_Date_Group"].value_counts().sort_index())


    # ANALYSIS 1
    # TEMPERATURE + LIGHT + GEAR
    #
    # Wild is excluded because there is no light measurement.
    # Groups:
    #   C = Cage
    #   N = Net Bottom
    #
    # Fixed effects:
    #   Temperature
    #   Light
    #   Gear
    #
    # Random effect:
    #   Collection date

    light_data = analysis_data[analysis_data[Dimension.GEAR.value].isin(["C", "N"])].copy()
    print("\n\n")
    print("=" * 70)
    print("ANALYSIS 1: TEMPERATURE + LIGHT")
    print("=" * 70)
    print("\nGear counts:")
    print(light_data[Dimension.GEAR.value].value_counts())
    print("\nCollection dates:")
    print(light_data["Collection_Date_Group"].value_counts().sort_index())
    print("\nMissing environmental variables:")
    print(light_data[["Temperature", "Light"]].isna().sum())

    light_results = []
    light_models = {}

    for response in response_vars:

        result, models = variation_partitioning_lmm(
            light_data, response.value, include_light=True
        )
        light_results.append(result)
        light_models[response.value] = models


    light_results_df = DataFrame(light_results)
    light_results_df.to_csv(
        figures / "variation_partitioning_temperature_light_random_date_summary.csv",
        index=False,
    )
    light_plot_data = light_results_df[
        [
            "Response",
            "Unique environment",
            "Unique gear",
            "Shared environment + gear",
            "Unexplained",
        ]
    ].melt(id_vars="Response", var_name="Component", value_name="Proportion")
    light_plot_data["Percent"] = light_plot_data["Proportion"] * 100
    light_plot_data.to_csv(
        figures / "variation_partitioning_temperature_light_random_date.csv", index=False
    )


    make_partition_plot(
        light_results_df,
        figures / "variation_partitioning_temperature_light_random_date.png",
        "Variation partitioning: temperature + light + gear\n(random effect: collection date)",
    )


    # ANALYSIS 2
    # TEMPERATURE + GEAR
    # Wild is included because Wild has temperature measurements.
    # Groups:
    #   C = Cage
    #   N = Net Bottom
    #   W = Wild
    #
    # Fixed effects:
    #   Temperature
    #   Gear
    #
    # Random effect:
    #   Collection date


    temperature_data = analysis_data[analysis_data[Dimension.GEAR.value].isin(["C", "N", "W"])].copy()
    print("\n\n")
    print("=" * 70)
    print("ANALYSIS 2: TEMPERATURE ONLY")
    print("=" * 70)
    print("\nGear counts:")
    print(temperature_data[Dimension.GEAR.value].value_counts())
    print("\nCollection dates:")
    print(temperature_data["Collection_Date_Group"].value_counts().sort_index())
    print("\nMissing temperature:")
    print(temperature_data[["Temperature"]].isna().sum())

    temperature_results = []
    temperature_models = {}

    for response in response_vars:
        result, models = variation_partitioning_lmm(
            temperature_data, response.value, include_light=False
        )
        temperature_results.append(result)
        temperature_models[response] = models

    temperature_results_df = DataFrame(temperature_results)
    temperature_results_df.to_csv(
        figures / "variation_partitioning_temperature_random_date_summary.csv", index=False
    )
    temperature_plot_data = temperature_results_df[
        [
            "Response",
            "Unique environment",
            "Unique gear",
            "Shared environment + gear",
            "Unexplained",
        ]
    ].melt(id_vars="Response", var_name="Component", value_name="Proportion")
    temperature_plot_data["Percent"] = temperature_plot_data["Proportion"] * 100
    temperature_plot_data.to_csv(
        figures / "variation_partitioning_temperature_random_date.csv", index=False
    )
    make_partition_plot(
        temperature_results_df,
        figures / "variation_partitioning_temperature_random_date.png",
        "Variation partitioning: temperature + gear\n(random effect: collection date)",
    )

    # FINAL SUMMARY
    print("ANALYSIS 1 — TEMPERATURE + LIGHT")
    print("Random effect: Collection date")
    print("-" * 70)
    print(
        light_results_df[
            [
                "Response",
                "N",
                "Collection dates",
                "Environment marginal R2",
                "Gear marginal R2",
                "Full marginal R2",
                "Full conditional R2",
                "Unique environment",
                "Unique gear",
                "Shared environment + gear",
                "Unexplained",
                "Collection date variance",
            ]
        ].round(4)
    )
    print("\n")
    print("ANALYSIS 2 — TEMPERATURE ONLY")
    print("Random effect: Collection date")
    print("-" * 70)
    print(
        temperature_results_df[
            [
                "Response",
                "N",
                "Collection dates",
                "Environment marginal R2",
                "Gear marginal R2",
                "Full marginal R2",
                "Full conditional R2",
                "Unique environment",
                "Unique gear",
                "Shared environment + gear",
                "Unexplained",
                "Collection date variance",
            ]
        ].round(4)
    )
