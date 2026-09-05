import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from pathlib import Path
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt

figures = Path(__file__).parent / "figures"
figures.mkdir(parents=True, exist_ok=True)
rawdata = Path(__file__).parent / "data" / "stable-isotopes-no-outliers.csv"

df = pd.read_csv(rawdata, header=0)
print(df.columns.tolist())

df = pd.read_csv(
    rawdata,
    header=0,
    usecols=[
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
    ],
)


for i in range(len(df["Date Run"])):
    if df["Date Run"][i] == "9/6/23":
        df.drop(i, inplace=True)
    else:
        pd.to_datetime(df["Date Run"][i], format="%m/%d/%y")


df.dropna(
    subset=["Gear Type"], inplace=True
)  # only scallops and filters are being plotted

data_muscle = df.dropna(subset=["Tissue Type"])
data_muscle = data_muscle.drop(data_muscle[data_muscle["Tissue Type"] == "G"].index)

pd.DataFrame(data_muscle)

environment_file = Path(__file__).parent / "data" / "temperature-and-light.csv"
env = pd.read_csv(environment_file)
env["Date-Time (EDT)"] = pd.to_datetime(env["Date-Time (EDT)"])
env["Date"] = env["Date-Time (EDT)"].dt.date

# Average hourly environmental data into daily means
env_daily = env.groupby("Date").mean(numeric_only=True).reset_index()

# lmm with light (therefore no wild scallops included)
lmm_data = data_muscle[
    ["d13C", "d15N", "C/N (Molar)", "Gear Type", "Collection Date"]
]
lmm_data = lmm_data.rename(columns={"Gear Type": "Gear"})
lmm_data = lmm_data[lmm_data["Gear"].isin(["C", "N", "W"])]
print(lmm_data["Gear"].value_counts())

lmm_data["Date"] = pd.to_datetime(
    "2023-" + lmm_data["Collection Date"].astype(int).astype(str).str.zfill(2) + "-15",
    format="%Y-%m-%d",
).dt.date

lmm_data = pd.merge(lmm_data, env_daily, on="Date", how="left")


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


lmm_data[["Temperature", "Light"]] = lmm_data.apply(assign_environment, axis=1)
print(lmm_data.head())
print(lmm_data.isna().sum())
print(lmm_data["Gear"].value_counts())
print(lmm_data["Collection Date"].value_counts())

lmm_clean = lmm_data[
    ["d13C", "d15N", "C/N (Molar)", "Gear", "Collection Date", "Temperature", "Light"]
].dropna()

print("LMM samples retained:", len(lmm_clean))
print(lmm_clean.isna().sum())

# VIF: Check for multicollinearity using Variance Inflation Factor (VIF)
X = pd.get_dummies(
    lmm_clean[["Temperature", "Light", "Gear"]], drop_first=True, dtype=float
)
X = sm.add_constant(X)
vif = pd.DataFrame(
    {
        "Variable": X.columns,
        "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])],
    }
)
# Remove intercept from reporting table
vif = vif[vif["Variable"] != "const"]
# Round values
vif = vif.round(3)
print(vif)
# plot the VIF table
fig, ax = plt.subplots(figsize=(5, 2))
ax.axis("off")
table = ax.table(cellText=vif.values, colLabels=vif.columns, loc="center")
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 2)
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_text_props(weight="bold")
plt.title("Variance Inflation Factor (VIF)")
plt.show()

# run lmm for each isotopic variance
model_d13C = smf.mixedlm(
    "d13C ~ Temperature + Light + C(Gear)",
    data=lmm_clean,
    groups=lmm_clean["Collection Date"],
)
result_d13C = model_d13C.fit()
print(result_d13C.summary())
model_d15N = smf.mixedlm(
    "d15N ~ Temperature + Light + C(Gear)",
    data=lmm_clean,
    groups=lmm_clean["Collection Date"],
)
result_d15N = model_d15N.fit()
print(result_d15N.summary())
model_CN = smf.mixedlm(
    "Q('C/N (Molar)') ~ Temperature + Light + C(Gear)",
    data=lmm_clean,
    groups=lmm_clean["Collection Date"],
)
result_CN = model_CN.fit()
print(result_CN.summary())

# lmm with no light therefore wild scallops included
lmm_env = data_muscle[
    ["d13C", "d15N", "C/N (Molar)", "Gear Type", "Collection Date"]
].copy()
lmm_env = lmm_env.rename(columns={"Gear Type": "Gear"})

# Create dates
lmm_env["Date"] = pd.to_datetime(
    "2023-" + lmm_env["Collection Date"].astype(int).astype(str).str.zfill(2) + "-15",
    format="%Y-%m-%d",
).dt.date

# Merge environment data
lmm_env = pd.merge(lmm_env, env_daily, on="Date", how="left")


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


lmm_env[["Temperature", "Light"]] = lmm_env.apply(assign_environment, axis=1)
# Keep only complete environmental observations
lmm_env_clean = lmm_env.dropna(
    subset=[
        "d13C",
        "d15N",
        "C/N (Molar)",
        "Temperature",
    ]
)
print("Environmental LMM samples:", len(lmm_env_clean))
print(lmm_env_clean["Gear"].value_counts())
# d13C model
model_d13C_env = smf.mixedlm(
    "d13C ~ Temperature + C(Gear)",
    data=lmm_env_clean,
    groups=lmm_env_clean["Collection Date"],
)
result_d13C_env = model_d13C_env.fit()
print(result_d13C_env.summary())
# d15N model
model_d15N_env = smf.mixedlm(
    "d15N ~ Temperature + C(Gear)",
    data=lmm_env_clean,
    groups=lmm_env_clean["Collection Date"],
)
result_d15N_env = model_d15N_env.fit()
print(result_d15N_env.summary())
# C/N model
model_CN_env = smf.mixedlm(
    "Q('C/N (Molar)') ~ Temperature + C(Gear)",
    data=lmm_env_clean,
    groups=lmm_env_clean["Collection Date"],
)
result_CN_env = model_CN_env.fit()
print(result_CN_env.summary())


# results into tables
def lmm_results_table(result, response_name):
    table = pd.DataFrame(
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
    table = table[~table["Predictor"].isin(["Intercept", "Group Var"])]

    return table


# Create combined LMM table
light_table = pd.concat(
    [
        lmm_results_table(result_d13C, "d13C"),
        lmm_results_table(result_d15N, "d15N"),
        lmm_results_table(result_CN, "C/N"),
    ]
)
light_table = light_table.round(3)
light_table.to_csv(figures / "LMM_light_table.csv", index=False)
print("LMM Results: Light Included")
print(light_table)
# cage used as reference level for gear type

# plot lmm with light
fig, ax = plt.subplots(figsize=(12, 6))
ax.axis("off")
table = ax.table(
    cellText=light_table.values,
    colLabels=light_table.columns,
    loc="center",
    cellLoc="center",
)
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1, 1.5)
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_text_props(weight="bold")
        cell.set_facecolor("#d9eaf7")

plt.title("Linear Mixed Model Results Light + Temperature", fontsize=14)
plt.tight_layout()
plt.savefig(figures / "LMM_light_table.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# Save table
light_table.to_csv(figures / "LMM_light_table.csv", index=False)

# plot table with no light
temp_table = pd.concat(
    [
        lmm_results_table(result_d13C_env, "d13C"),
        lmm_results_table(result_d15N_env, "d15N"),
        lmm_results_table(result_CN_env, "C/N"),
    ]
)
temp_table = temp_table.round(3)
temp_table.to_csv(figures / "LMM_temperature_only_table.csv", index=False)
print("LMM Results: Temperature Only")
print(temp_table)
# cage used as reference level for gear type

# plot lmm with no light
fig, ax = plt.subplots(figsize=(12, 6))
ax.axis("off")
table = ax.table(
    cellText=temp_table.values,
    colLabels=temp_table.columns,
    loc="center",
    cellLoc="center",
)
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1, 1.5)
# Make header row bold
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_text_props(weight="bold")
        cell.set_facecolor("#d9eaf7")

plt.title("Linear Mixed Model Results Temperature Only", fontsize=14)
plt.tight_layout()
plt.savefig(figures / "LMM_temperature_only_table.png", dpi=300, bbox_inches="tight")
plt.close(fig)

vif.to_csv(figures / "VIF_results.csv", index=False)

# =======
# AIC
# =======


def compare_aic(models, response):
    """
    Create AIC comparison table for mixed models.
    """
    aic_table = pd.DataFrame(
        {
            "Response": response,
            "Model": list(models.keys()),
            "AIC": [model.aic for model in models.values()],
        }
    )

    # Calculate delta AIC
    aic_table["Delta AIC"] = aic_table["AIC"] - aic_table["AIC"].min()

    return aic_table.sort_values("AIC")


# d13C candidate models
d13C_models = {
    "Temperature + Light + Gear": result_d13C,
    "Temperature + Light": smf.mixedlm(
        "d13C ~ Temperature + Light",
        data=lmm_clean,
        groups=lmm_clean["Collection Date"],
    ).fit(),
    "Gear only": smf.mixedlm(
        "d13C ~ C(Gear)", data=lmm_clean, groups=lmm_clean["Collection Date"]
    ).fit(),
    "Null": smf.mixedlm(
        "d13C ~ 1", data=lmm_clean, groups=lmm_clean["Collection Date"]
    ).fit(),
}
d13C_AIC = compare_aic(d13C_models, "d13C")
print("\nAIC comparison: d13C")
print(d13C_AIC)

# temp data not merging properly so need
# to look into this! also because wild temp
# doesnt exist for the first month only 4 months are being used.

# def linear_mixed_model(df):
#    subset = df[
#        [
#            Dimension.NITROGEN_FRACTIONATION.value,
#           Dimension.GEAR.value,
#           Dimension.COLLECTION_DATE.value,
#           Dimension.TISSUE.value,
#       ]
#   ].dropna()
#
#   subset = subset.rename(columns={
#       Dimension.NITROGEN_FRACTIONATION.value: "d15N",
#       Dimension.GEAR.value: "Gear",
#       Dimension.COLLECTION_DATE.value: "Month",
#       Dimension.TISSUE.value: "Tissue",
#   })

#   model = mixedlm(
#       "d15N ~ C(Gear) + C(Tissue)",
#       data=subset,
#       groups=subset["Month"],
#   )
#   result = model.fit()
#   print(result.summary())
