import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from pathlib import Path
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).parent
data_file = BASE_DIR / "data" / "stable-isotopes-no-outliers.csv"
environment_file = BASE_DIR / "data" / "temperature-and-light.csv"
figures = BASE_DIR / "figures"
figures.mkdir(parents=True, exist_ok=True)

# LOAD STABLE ISOTOPE DATA

df = pd.read_csv(data_file)
print("\nStable isotope columns:")
print(df.columns.tolist())
df = df[  # Keep only variables needed for this analysis
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

# REMOVE EXCLUDED DATE-RUN
df = df[df["Date Run"] != "9/6/23"].copy()

df = df.dropna(  # Remove observations without a gear type
    subset=["Gear Type", "Tissue Type"]
).copy()

data_muscle = df[df["Tissue Type"] == "M"].copy()

data_muscle = data_muscle.rename(  # Rename Gear Type to Gear
    columns={"Gear Type": "Gear"}
)

print("\nGear counts:")
print(data_muscle["Gear"].value_counts())

# LOAD ENVIRONMENTAL DATA

env = pd.read_csv(environment_file)
env["Date-Time (EDT)"] = pd.to_datetime(  # Convert environmental date/time
    env["Date-Time (EDT)"], errors="coerce"
)

env["Month"] = env["Date-Time (EDT)"].dt.month  # Extract month

# CALCULATE MONTHLY ENVIRONMENTAL MEANS
env_monthly = env.groupby("Month").mean(numeric_only=True).reset_index()
print("\nMonthly environmental data:")
print(env_monthly)

# CREATE MONTH VARIABLE FOR ISOTOPE DATA

data_muscle["Month"] = pd.to_numeric(data_muscle["Collection Date"], errors="coerce")

data_muscle = data_muscle.dropna(  # Remove observations without collection month
    subset=["Month"]
).copy()
data_muscle["Month"] = data_muscle["Month"].astype(int)

# MERGE ENVIRONMENTAL DATA

analysis_data = pd.merge(data_muscle, env_monthly, on="Month", how="left")
print("\nEnvironmental data after merge:")
print(analysis_data[["Collection Date", "Month", "Gear"]].head())


# ASSIGN ENVIRONMENT BASED ON GEAR
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

print("\nEnvironmental data availability:")
print(analysis_data[["Temperature", "Light"]].isna().sum())
print("\nGear counts after environmental merge:")
print(analysis_data["Gear"].value_counts())

# RESPONSES
responses = ["d13C", "d15N", "C/N (Molar)"]


def model_r2(data, response, formula):
    """Ordinary least squares (OLS) model."""
    model = smf.ols(formula=formula, data=data).fit()
    return (model.rsquared, model)


def variation_partitioning(data, response, include_light=True):
    """
    Variation partitioning between:
        Environment
        Gear

    If include_light=True:
        Environment =
            Temperature + Light

    If include_light=False:
        Environment =
            Temperature

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
    # REQUIRED VARIABLES
    variables = [response, "Temperature", "Gear"]
    if include_light:

        variables.append("Light")

    # COMPLETE CASES
    subset = data.dropna(subset=variables).copy()

    # FORMULAS
    if include_light:
        environment_formula = f'Q("{response}") ' f"~ Temperature + Light"
        full_formula = f'Q("{response}") ' f"~ Temperature + Light + C(Gear)"
    else:
        environment_formula = f'Q("{response}") ' f"~ Temperature"
        full_formula = f'Q("{response}") ' f"~ Temperature + C(Gear)"

    gear_formula = f'Q("{response}") ' f"~ C(Gear)"
    null_formula = f'Q("{response}") ~ 1'

    # FIT MODELS
    r2_environment, environment_model = model_r2(subset, response, environment_formula)
    r2_gear, gear_model = model_r2(subset, response, gear_formula)
    r2_full, full_model = model_r2(subset, response, full_formula)
    r2_null, null_model = model_r2(subset, response, null_formula)

    # VARIATION PARTITIONING
    unique_environment = r2_full - r2_gear
    unique_gear = r2_full - r2_environment
    shared = r2_environment + r2_gear - r2_full
    unexplained = 1 - r2_full

    # RESULTS

    components = {
        "Unique environment": unique_environment,
        "Unique gear": unique_gear,
        "Shared environment + gear": shared,
        "Unexplained": unexplained,
    }

    # PRINT RESULTS
    print("\n" + "=" * 70)

    if include_light:
        print(f"VARIATION PARTITIONING: " f"{response} — TEMPERATURE + LIGHT")
    else:
        print(f"VARIATION PARTITIONING: " f"{response} — TEMPERATURE ONLY")
    print("=" * 70)
    print(f"Samples: {len(subset)}")
    print(f"R² environment: " f"{r2_environment:.4f}")
    print(f"R² gear:        " f"{r2_gear:.4f}")
    print(f"R² full:        " f"{r2_full:.4f}")
    print()
    for key, value in components.items():
        print(f"{key:30s}: " f"{value:.4f} " f"({value * 100:.2f}%)")

    print(f"\nComponents sum: " f"{sum(components.values()):.4f}")

    return {
        "Response": response,
        "N": len(subset),
        "Environment R2": r2_environment,
        "Gear R2": r2_gear,
        "Full R2": r2_full,
        "Unique environment": unique_environment,
        "Unique gear": unique_gear,
        "Shared environment + gear": shared,
        "Unexplained": unexplained,
    }


# ANALYSIS 1:
# TEMPERATURE + LIGHT + GEAR
#
# Wild scallops are NOT included here because they have
# no light measurements.
#
# Therefore this analysis compares:
#     Cage (C)
#     Net Bottom (N)
#
# Environment:
#     Temperature
#     Light
# Gear:
#     C vs N


light_data = analysis_data[analysis_data["Gear"].isin(["C", "N"])].copy()

print("\n")
print("=" * 70)
print("ANALYSIS 1: TEMPERATURE + LIGHT")
print("=" * 70)

print("\nGear counts:")
print(light_data["Gear"].value_counts())

print("\nMissing environmental variables:")
print(light_data[["Temperature", "Light"]].isna().sum())

light_results = []

for response in responses:
    result = variation_partitioning(light_data, response, include_light=True)
    light_results.append(result)


light_results_df = pd.DataFrame(light_results)

# SAVE ANALYSIS 1 RESULTS

light_results_df.to_csv(
    figures / "variation_partitioning_temperature_light_summary.csv", index=False
)

# Long-format version
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
    figures / "variation_partitioning_temperature_light.csv", index=False
)

# PLOT ANALYSIS 1

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
x_positions = np.arange(len(responses))
bottom = np.zeros(len(responses))

for component in plot_order:

    values = []

    for response in responses:

        value = light_results_df.loc[
            light_results_df["Response"] == response, component
        ].iloc[0]

        values.append(value * 100)

    values = np.array(values)

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
ax.set_title("Variation partitioning: temperature + light + gear", fontsize=14)
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(
    figures / "variation_partitioning_temperature_light.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close()

# ANALYSIS 2:
# TEMPERATURE + GEAR
# Wild scallops ARE included here because Wild has
# temperature data. Therefore this analysis compares:
#     Cage (C)
#     Net Bottom (N)
#     Wild (W)
#
# Environment:
#     Temperature
#
# Gear:
#     C vs N vs W

temperature_data = analysis_data[analysis_data["Gear"].isin(["C", "N", "W"])].copy()

print("\n")
print("=" * 70)
print("ANALYSIS 2: TEMPERATURE ONLY")
print("=" * 70)
print("\nGear counts:")
print(temperature_data["Gear"].value_counts())
print("\nMissing temperature:")
print(temperature_data[["Temperature"]].isna().sum())

temperature_results = []
for response in responses:
    result = variation_partitioning(temperature_data, response, include_light=False)
    temperature_results.append(result)

temperature_results_df = pd.DataFrame(temperature_results)

# SAVE ANALYSIS 2 RESULTS

temperature_results_df.to_csv(
    figures / "variation_partitioning_temperature_summary.csv", index=False
)

temperature_plot_data = temperature_results_df[  # Long-format version
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
    figures / "variation_partitioning_temperature.csv", index=False
)

#  PLOT ANALYSIS 2

fig, ax = plt.subplots(figsize=(9, 6))
x_positions = np.arange(len(responses))
bottom = np.zeros(len(responses))

for component in plot_order:
    values = []
    for response in responses:
        value = temperature_results_df.loc[
            temperature_results_df["Response"] == response, component
        ].iloc[0]
        values.append(value * 100)
    values = np.array(values)
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
ax.set_title("Variation partitioning: temperature + gear", fontsize=14)
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(
    figures / "variation_partitioning_temperature.png", dpi=300, bbox_inches="tight"
)
plt.close()

# FINAL SUMMARY

print("\n")
print("=" * 70)
print("VARIATION PARTITIONING COMPLETE")
print("=" * 70)
print("\n")
print("ANALYSIS 1 — TEMPERATURE + LIGHT")
print("--------------------------------")
print(
    light_results_df[
        [
            "Response",
            "N",
            "Full R2",
            "Unique environment",
            "Unique gear",
            "Shared environment + gear",
            "Unexplained",
        ]
    ].round(4)
)
print("\n")
print("ANALYSIS 2 — TEMPERATURE ONLY")
print("--------------------------------")
print(
    temperature_results_df[
        [
            "Response",
            "N",
            "Full R2",
            "Unique environment",
            "Unique gear",
            "Shared environment + gear",
            "Unexplained",
        ]
    ].round(4)
)
print("\n")
print("Figures and result tables saved to:")
print(figures)
print("\nFiles created:")
print("  - variation_partitioning_temperature_light.png")
print("  - variation_partitioning_temperature_light_summary.csv")
print("  - variation_partitioning_temperature_light.csv")
print("  - variation_partitioning_temperature.png")
print("  - variation_partitioning_temperature_summary.csv")
print("  - variation_partitioning_temperature.csv")
print("\nDone.")
