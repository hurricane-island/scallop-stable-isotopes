"""
Relative contributions of temperature, light, and culture method
in explaining the variation in stable isotope ratios of scallop tissues
"""
from datetime import date, timedelta
from pathlib import Path
from enum import Enum
from numpy import arange, zeros, nan, array
from matplotlib.pyplot import subplots, close
from pandas import DataFrame, to_datetime, read_csv, get_dummies, concat
import statsmodels.api as sm
from statsmodels.formula.api import ols, mixedlm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from isotopes.statistics import partition_data_by_tissue
from isotopes.options import Dimension, TissueType, isotopes_no_outliers, CultureMethod, EnvDimension, env_data

figures = Path(__file__).parent / "figures"

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
    CultureMethod.WILD.value: EnvDimension.WILD_TEMP.value
}
light_column = {
    CultureMethod.CAGE.value: EnvDimension.CAGE_LUM.value,
    CultureMethod.NET.value: EnvDimension.NET_BOTTOM_LUM.value,
}

def lmm_results_table(result, response_name):
    """Results into table format"""
    _table = DataFrame(
        {
            "Response": response_name,
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


def mixedlm_analysis(data: DataFrame, dims: list[str], groups: Dimension):
    """
    Linear Mixed Models: Run for each isotopic variance
    """
    lookup = {}
    for response in [
        Dimension.CARBON_FRACTIONATION,
        Dimension.NITROGEN_FRACTIONATION,
        Dimension.MOLAR_RATIO,
    ]:
        result = mixedlm(
            f'Q("{response.value}") ~ {" + ".join(dims)} + C(Q("{Dimension.GEAR.value}"))',
            data=data,
            groups=data[groups.value],
        ).fit()
        print(result.summary())
        lookup[response] = result

    light_table = concat([lmm_results_table(val, key) for key, val in lookup.items()])
    light_table.to_csv(figures / f"lmm_{'_'.join(dims).lower()}_table.csv", index=False)


def variation_partitioning(data: DataFrame, response: str, dims: list[str]):
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
    subset = data.dropna(subset=[response, Dimension.GEAR.value, *dims])

    env_formula = f'Q("{response}") ~ {" + ".join(dims)}'
    env_model = ols(formula=env_formula, data=subset).fit()

    gear_formula = f'Q("{response}") ~ C(Q("{Dimension.GEAR.value}"))'
    gear_model = ols(formula=gear_formula, data=subset).fit()

    full_formula = f'Q("{response}") ~ {" + ".join(dims)} + C(Q("{Dimension.GEAR.value}"))'
    full_model = ols(formula=full_formula, data=subset).fit()

    null_formula = f'Q("{response}") ~ 1'
    null_model = ols(formula=null_formula, data=subset).fit()

    components = {
        "Unique environment": full_model.rsquared - gear_model.rsquared,
        "Unique gear": full_model.rsquared - env_model.rsquared,
        "Shared environment + gear": env_model.rsquared + gear_model.rsquared - full_model.rsquared,
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
        "Response": response,
        "N": len(subset),
        "Environment R2": env_model.rsquared,
        "Gear R2": gear_model.rsquared,
        "Full R2": full_model.rsquared,
        **components
    }

def plot_partitions(response_df: DataFrame, parts: list[str], responses: list[str]):
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
                response_df["Response"] == each, component
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
    ax.set_xticklabels(responses)
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


def run_analysis(df: DataFrame, subset: list[str], responses: list[str]):
    """
    Run variation partitioning analysis.
    """
    _data = df.dropna(subset=subset)
    analysis_var = [each.lower() for each in [*subset, Dimension.GEAR.value]]
    print(f"\nVARIATION PARTITIONING: {' + '.join(analysis_var)}")
    print("-" * 80)

    results_df = DataFrame([variation_partitioning(_data, r, dims=subset) for r in responses])
    plot_data = results_df[
        [
            "Response",
            "Unique environment",
            "Unique gear",
            "Shared environment + gear",
            "Unexplained",
        ]
    ].melt(id_vars="Response", var_name="Component", value_name="Proportion")
    plot_data["Percent"] = plot_data["Proportion"] * 100
    plot_partitions(results_df, analysis_var, responses)


if __name__ == "__main__":
    # VIF

    figures.mkdir(parents=True, exist_ok=True)

    df = partition_data_by_tissue(
        filepath=isotopes_no_outliers,
        usecols=[
            Dimension.COLLECTION_DATE,
            Dimension.GEAR,
            Dimension.NITROGEN_FRACTIONATION,
            Dimension.CARBON_FRACTIONATION,
            Dimension.MOLAR_RATIO,
        ],
        tissue_type=TissueType.MUSCLE,
    ).dropna()

    # Add columns to populate with environmental data
    df[DerivedDimension.TEMPERATURE.value] = nan
    df[DerivedDimension.LIGHT.value] = nan

    env = read_csv(env_data)
    env["Date-Time (EDT)"] = to_datetime(
        env["Date-Time (EDT)"], errors="coerce"
    )
    env_dates = env[EnvDimension.DATE.value].dt.date
    groups = df.groupby([Dimension.GEAR.value, Dimension.COLLECTION_DATE.value])
    for (gear, month), indx in groups.groups.items():
        # Each group of method and collection date will share the same
        # environmental conditions
        # Get preceding 30 days of environmental data
        end_date = collection_date_map[int(month)]
        start_date = end_date - timedelta(days=30)
        env_subset = env[(env_dates >= start_date) & (env_dates <= end_date)]
        df.loc[indx, DerivedDimension.TEMPERATURE.value] = env_subset[temperature_column[gear]].mean()
        df.loc[indx, DerivedDimension.LIGHT.value] = env_subset[light_column[gear]].mean() if gear in light_column else nan

    responses = [
        Dimension.CARBON_FRACTIONATION.value, 
        Dimension.NITROGEN_FRACTIONATION.value, 
        Dimension.MOLAR_RATIO.value
    ]

    df_with_light = df.dropna(subset=[DerivedDimension.TEMPERATURE.value, DerivedDimension.LIGHT.value])
    df_temp_only = df.dropna(subset=[DerivedDimension.TEMPERATURE.value])

    lmm_clean_with_light = lmm_data[
        ["d13C", "d15N", "C/N (Molar)", "Gear", "Collection Date", "Temperature", "Light"]
    ].dropna()
    lmm_clean_temp_only = lmm_data[
        ["d13C", "d15N", "C/N (Molar)", "Gear", "Collection Date", "Temperature"]
    ].dropna()

    # VIF: Check for multicollinearity using Variance Inflation Factor (VIF)
    X = get_dummies(
        lmm_clean_with_light[["Temperature", "Light", "Gear"]], drop_first=True, dtype=float
    )
    X = sm.add_constant(X)
    vif = DataFrame(
        {
            "Variable": X.columns,
            "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])],
        }
    )
    vif.to_csv(figures / "vif_results.csv", index=False)
    mixedlm_analysis(
        lmm_clean_with_light, ["Temperature", "Light"], Dimension.COLLECTION_DATE
    )
    mixedlm_analysis(lmm_clean_temp_only, ["Temperature"], Dimension.COLLECTION_DATE)

    # ANALYSIS: TEMPERATURE + LIGHT + GEAR
    # Wild scallops are NOT included because they have no light.
    run_analysis(df, ["Temperature", "Light"])

    # ANALYSIS: TEMPERATURE + GEAR
    # Wild scallops ARE included.
    run_analysis(df, ["Temperature"])
