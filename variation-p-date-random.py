import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from pathlib import Path
import matplotlib.pyplot as plt
import warnings

from statsmodels.tools.sm_exceptions import ConvergenceWarning

# VARIATION PARTITIONING WITH COLLECTION DATE AS RANDOM EFFECT
#
# TWO SEPARATE ANALYSES
#
# ANALYSIS 1:
#   Temperature + Light + Gear
#   C = Cage
#   N = Net Bottom
#   Wild excluded because there is no light measurement
#
# ANALYSIS 2:
#   Temperature + Gear
#   C = Cage
#   N = Net Bottom
#   W = Wild
#
# RANDOM EFFECT:
#   Collection date
#
# Collection date codes:
#   6  = 06/15/2023
#   7  = 07/13/2023
#   8  = 08/14/2023
#   9  = 09/15/2023
#   10 = 10/12/2023


BASE_DIR = Path(__file__).parent
data_file = BASE_DIR / "data" / "stable-isotopes-no-outliers.csv"
environment_file = BASE_DIR / "data" / "temperature-and-light.csv"
figures = BASE_DIR / "figures"
figures.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(data_file)


df = df[
    [
        "Analysis",
        "Sample ID",
        "Collection Date",
        "Gear Type",
        "Sex",
        "Tissue Type",
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
    ]
].copy()

df = df[df["Date Run"] != "9/6/23"].copy()
df = df.dropna(subset=["Gear Type"]).copy()
data_muscle = df.dropna(subset=["Tissue Type"]).copy()
data_muscle = data_muscle[data_muscle["Tissue Type"] == "M"].copy()
data_muscle = data_muscle.rename(columns={"Gear Type": "Gear"})
print("\nInitial gear counts:")
print(data_muscle["Gear"].value_counts())


collection_date_map = {
    6: "2023-06-15",
    7: "2023-07-13",
    8: "2023-08-14",
    9: "2023-09-15",
    10: "2023-10-12",
}
data_muscle["Collection_Date_Code"] = pd.to_numeric(
    data_muscle["Collection Date"], errors="coerce"
)
data_muscle["Collection_Date_Group"] = data_muscle["Collection_Date_Code"].map(
    collection_date_map
)
data_muscle["Collection_Date_Group"] = pd.to_datetime(
    data_muscle["Collection_Date_Group"], errors="coerce"
)
data_muscle = data_muscle.dropna(subset=["Collection_Date_Group"]).copy()
data_muscle["Month"] = data_muscle["Collection_Date_Group"].dt.month
print("\nCollection dates:")
print(data_muscle["Collection_Date_Group"].value_counts().sort_index())
print("\nCollection months:")
print(data_muscle["Month"].value_counts().sort_index())

# Check number of collection dates

n_collection_dates = data_muscle["Collection_Date_Group"].nunique()
print("\nNumber of collection dates:", n_collection_dates)
if n_collection_dates < 2:

    raise ValueError(
        "Fewer than two collection dates were found. "
        "Check the Collection Date coding."
    )


env = pd.read_csv(environment_file)

env["Date-Time (EDT)"] = pd.to_datetime(env["Date-Time (EDT)"], errors="coerce")
if env["Date-Time (EDT)"].isna().any():
    print("\nWARNING: Some environmental timestamps " "could not be parsed.")

env["Month"] = env["Date-Time (EDT)"].dt.month

env_monthly = env.groupby("Month").mean(numeric_only=True).reset_index()
print("\nMonthly environmental data:")
print(env_monthly)

analysis_data = pd.merge(data_muscle, env_monthly, on="Month", how="left")
print("\nEnvironmental data after merge:")
print(analysis_data[["Collection_Date_Group", "Month", "Gear"]].head())


def assign_environment(row):
    if row["Gear"] == "C":
        return pd.Series(
            {
                "Temperature": row["Cage, Temperature (°F)"],
                "Light": row["Cage, Light (lum)"],
            }
        )
    elif row["Gear"] == "N":
        return pd.Series(
            {
                "Temperature": row["Net Bottom, Temperature (°F)"],
                "Light": row["Net Bottom, Light (lum)"],
            }
        )
    elif row["Gear"] == "W":
        return pd.Series(
            {"Temperature": row["Wild, Temperature (°F)"], "Light": np.nan}
        )
    else:
        return pd.Series({"Temperature": np.nan, "Light": np.nan})


analysis_data[["Temperature", "Light"]] = analysis_data.apply(
    assign_environment, axis=1
)

# CLEAN VARIABLES

analysis_data["Gear"] = analysis_data["Gear"].astype("category")

# Make sure date group is categorical

analysis_data["Collection_Date_Group"] = analysis_data["Collection_Date_Group"].astype(
    "category"
)
print("\nEnvironmental data availability:")
print(analysis_data[["Temperature", "Light"]].isna().sum())
print("\nGear counts after environmental merge:")
print(analysis_data["Gear"].value_counts())
print("\nCollection dates used as random-effect groups:")

print(analysis_data["Collection_Date_Group"].value_counts().sort_index())


responses = ["d13C", "d15N", "C/N (Molar)"]

def fit_lmm(data, formula, response):
    """
    Fit a linear mixed model with collection date
    as the random intercept.

    Random effect:

        Collection_Date_Group

    Fixed effects are defined by the formula.
    """

    model = smf.mixedlm(
        formula=formula, data=data, groups=data["Collection_Date_Group"]
    )

    # Try several optimizers because MixedLM can sometimes
    # have difficulty converging.
    optimizers = ["lbfgs", "powell", "bfgs", "cg"]
    last_error = None
    for optimizer in optimizers:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                result = model.fit(
                    reml=False, method=optimizer, maxiter=2000, disp=False
                )
            # Check whether optimization succeeded

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


# MIXED MODEL R²

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


# LMM VARIATION PARTITIONING


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

    # COMPLETE CASES
    required = [response, "Temperature", "Gear", "Collection_Date_Group"]
    if include_light:

        required.append("Light")
    subset = data.dropna(subset=required).copy()
    # Remove unused category levels

    subset["Gear"] = subset["Gear"].cat.remove_unused_categories()
    subset["Collection_Date_Group"] = subset[
        "Collection_Date_Group"
    ].cat.remove_unused_categories()

    # CHECK RANDOM EFFECT

    n_dates = subset["Collection_Date_Group"].nunique()
    if n_dates < 2:

        raise ValueError(
            f"Only {n_dates} collection date "
            f"group was found for {response}. "
            "At least two collection dates are "
            "required for a random effect."
        )

    # FORMULAS

    if include_light:
        environment_formula = f'Q("{response}") ' f"~ Temperature + Light"

        full_formula = f'Q("{response}") ' f"~ Temperature + Light + C(Gear)"

    else:
        environment_formula = f'Q("{response}") ' f"~ Temperature"
        full_formula = f'Q("{response}") ' f"~ Temperature + C(Gear)"
    gear_formula = f'Q("{response}") ' f"~ C(Gear)"
    null_formula = f'Q("{response}") ~ 1'

    # PRINT HEADER

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

    # FIT MODELS

    print("\nFitting null model...")
    null_model = fit_lmm(subset, null_formula, response)
    print("Fitting environment model...")
    environment_model = fit_lmm(subset, environment_formula, response)
    print("Fitting gear model...")
    gear_model = fit_lmm(subset, gear_formula, response)
    print("Fitting full model...")
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


def make_partition_plot(results_df, output_file, title):
    plot_order = [
        "Unique environment",
        "Shared environment + gear",
        "Unique gear",
        "Unexplained",
    ]
    plot_colors = {
        "Unique environment": "#2C7FB8",
        "Shared environment + gear": "#7FCDBB",
        "Unique gear": "#F03B20",
        "Unexplained": "#D9D9D9",
    }
    fig, ax = plt.subplots(figsize=(9, 6))
    x_positions = np.arange(len(results_df))
    bottom = np.zeros(len(results_df))
    for component in plot_order:
        values = results_df[component].values * 100
        ax.bar(
            x_positions,
            values,
            bottom=bottom,
            label=component,
            color=plot_colors[component],
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
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()


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

light_data = analysis_data[analysis_data["Gear"].isin(["C", "N"])].copy()
print("\n\n")
print("=" * 70)
print("ANALYSIS 1: TEMPERATURE + LIGHT")
print("=" * 70)
print("\nGear counts:")
print(light_data["Gear"].value_counts())
print("\nCollection dates:")
print(light_data["Collection_Date_Group"].value_counts().sort_index())
print("\nMissing environmental variables:")
print(light_data[["Temperature", "Light"]].isna().sum())

light_results = []
light_models = {}

for response in responses:

    result, models = variation_partitioning_lmm(
        light_data, response, include_light=True
    )
    light_results.append(result)
    light_models[response] = models


light_results_df = pd.DataFrame(light_results)

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


temperature_data = analysis_data[analysis_data["Gear"].isin(["C", "N", "W"])].copy()
print("\n\n")
print("=" * 70)
print("ANALYSIS 2: TEMPERATURE ONLY")
print("=" * 70)
print("\nGear counts:")
print(temperature_data["Gear"].value_counts())
print("\nCollection dates:")
print(temperature_data["Collection_Date_Group"].value_counts().sort_index())
print("\nMissing temperature:")
print(temperature_data[["Temperature"]].isna().sum())

temperature_results = []
temperature_models = {}

for response in responses:
    result, models = variation_partitioning_lmm(
        temperature_data, response, include_light=False
    )
    temperature_results.append(result)
    temperature_models[response] = models

temperature_results_df = pd.DataFrame(temperature_results)

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

# PLOT

make_partition_plot(
    temperature_results_df,
    figures / "variation_partitioning_temperature_random_date.png",
    "Variation partitioning: temperature + gear\n(random effect: collection date)",
)

# FINAL SUMMARY
print("\n\n")
print("=" * 70)
print("VARIATION PARTITIONING COMPLETE")
print("=" * 70)
print("\n")
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

# SAVE COLLECTION-DATE INFORMATION

collection_date_summary = (
    analysis_data[["Collection_Date_Code", "Collection_Date_Group", "Month"]]
    .drop_duplicates()
    .sort_values("Collection_Date_Group")
)
collection_date_summary.to_csv(figures / "collection_date_mapping.csv", index=False)

