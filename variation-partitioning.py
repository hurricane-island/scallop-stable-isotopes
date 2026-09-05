"""
Relative contributions of temperature, light, and culture method
in explaining the variation in stable isotope ratios of scallop tissues
"""
from pandas import DataFrame, read_csv, to_datetime, to_numeric, merge, Series
from numpy import arange, zeros, nan, array
import statsmodels.formula.api as smf
from pathlib import Path
from matplotlib.pyplot import subplots, close

BASE_DIR = Path(__file__).parent
data_file = BASE_DIR / "data" / "stable-isotopes-no-outliers.csv"
environment_file = BASE_DIR / "data" / "temperature-and-light.csv"
figures = BASE_DIR / "figures"
figures.mkdir(parents=True, exist_ok=True)


responses = ["d13C", "d15N", "C/N (Molar)"]
df = read_csv(data_file)[
    [
        "Collection Date",
        "Gear Type",
        "Tissue Type",
        "d15N",
        "d13C",
        "C/N (Molar)",
        "Date Run",
    ]
]

df = df[df["Date Run"] != "9/6/23"]

df = df.dropna(  # Remove observations without a gear type
    subset=["Gear Type", "Tissue Type"]
)

data_muscle = df[df["Tissue Type"] == "M"]

data_muscle = data_muscle.rename(  # Rename Gear Type to Gear
    columns={"Gear Type": "Gear"}
)

# LOAD ENVIRONMENTAL DATA
env = read_csv(environment_file)
env["Date-Time (EDT)"] = to_datetime(
    env["Date-Time (EDT)"], errors="coerce"
)
env["Month"] = env["Date-Time (EDT)"].dt.month  # Extract month
env_monthly = env.groupby("Month").mean(numeric_only=True).reset_index()

data_muscle["Month"] = to_numeric(data_muscle["Collection Date"], errors="coerce")

data_muscle = data_muscle.dropna(  # Remove observations without collection month
    subset=["Month"]
)
data_muscle["Month"] = data_muscle["Month"].astype(int)

# MERGE ENVIRONMENTAL DATA
analysis_data = merge(data_muscle, env_monthly, on="Month", how="left")
print("\nEnvironmental data after merge:")
print(analysis_data[["Collection Date", "Month", "Gear"]].head())


def assign_environment(row):
    """
    Map gear type to environmental variables available based on
    a priori knowledge
    """
    if row["Gear"] == "C":
        return Series(
            {
                "Temperature": row["Cage, Temperature (°F)"],
                "Light": row["Cage, Light (lum)"],
            }
        )
    elif row["Gear"] == "N":
        return Series(
            {
                "Temperature": row["Net Bottom, Temperature (°F)"],
                "Light": row["Net Bottom, Light (lum)"],
            }
        )
    elif row["Gear"] == "W":
        return Series(
            {"Temperature": row["Wild, Temperature (°F)"], "Light": nan}
        )
    else:
        return Series({"Temperature": nan, "Light": nan})


analysis_data[["Temperature", "Light"]] = analysis_data.apply(
    assign_environment, axis=1
)

analysis_data["Gear"] = analysis_data["Gear"].astype("category")

print("\nEnvironmental data availability:")
print(analysis_data[["Temperature", "Light"]].isna().sum())
print("\nGear counts after environmental merge:")
print(analysis_data["Gear"].value_counts())

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
    variables = [response, "Temperature", "Gear"]
    if include_light:
        variables.append("Light")
    subset = data.dropna(subset=variables)
    if include_light:
        environment_formula = f'Q("{response}") ~ Temperature + Light'
        full_formula = f'Q("{response}") ~ Temperature + Light + C(Gear)'
    else:
        environment_formula = f'Q("{response}") ~ Temperature'
        full_formula = f'Q("{response}") ~ Temperature + C(Gear)'

    gear_formula = f'Q("{response}") ~ C(Gear)'
    null_formula = f'Q("{response}") ~ 1'

    r2_environment, environment_model = model_r2(subset, response, environment_formula)
    r2_gear, gear_model = model_r2(subset, response, gear_formula)
    r2_full, full_model = model_r2(subset, response, full_formula)
    r2_null, null_model = model_r2(subset, response, null_formula)

    # VARIATION PARTITIONING
    unique_environment = r2_full - r2_gear
    unique_gear = r2_full - r2_environment
    shared = r2_environment + r2_gear - r2_full
    unexplained = 1 - r2_full
    components = {
        "Unique environment": unique_environment,
        "Unique gear": unique_gear,
        "Shared environment + gear": shared,
        "Unexplained": unexplained,
    }

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



def plot_partitions(response_df: DataFrame, parts: list[str]):
    """
    Create stacked bar chart of variation partitioning results.
    """
    plot_colors = {
        "Unique environment": "#2C7FB8",
        "Shared environment + gear": "#7FCDBB",
        "Unique gear": "#F03B20",
        "Unexplained": "#D9D9D9",
    }
    fig, ax = subplots(figsize=(9, 6))
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


light_data = analysis_data[analysis_data["Gear"].isin(["C", "N"])]
analysis_var = ["temperature", "light", "gear"]
print("\n")
print("=" * 70)
print(f"ANALYSIS 1: {' + '.join(analysis_var)}")
print("=" * 70)
print("\nGear counts:")
print(light_data["Gear"].value_counts())
print("\nMissing environmental variables:")
print(light_data[["Temperature", "Light"]].isna().sum())

light_results_df = DataFrame([variation_partitioning(light_data, r, include_light=True) for r in responses])
light_results_df.to_csv(
    figures / f"variation_partitioning_{'_'.join(analysis_var)}_summary.csv", index=False
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
    figures / f"variation_partitioning_{'_'.join(analysis_var)}.csv", index=False
)
plot_partitions(light_results_df, analysis_var)

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

temperature_data = analysis_data[analysis_data["Gear"].isin(["C", "N", "W"])]
analysis_var = ["temperature", "gear"]

print("\n")
print("=" * 70)
print(f"ANALYSIS 2: {' + '.join(analysis_var)}")
print("=" * 70)
print("\nGear counts:")
print(temperature_data["Gear"].value_counts())
print("\nMissing temperature:")
print(temperature_data[["Temperature"]].isna().sum())

temperature_results_df = DataFrame([
    variation_partitioning(temperature_data, response, include_light=False) for response in responses
])
temperature_results_df.to_csv(
    figures / f"variation_partitioning_{'_'.join(analysis_var)}_summary.csv", index=False
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
    figures / f"variation_partitioning_{'_'.join(analysis_var)}.csv", index=False
)
plot_partitions(temperature_results_df, analysis_var)

print("\n")
print("Figures and result tables saved to:")
print(figures)
