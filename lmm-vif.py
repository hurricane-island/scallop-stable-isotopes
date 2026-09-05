from pandas import DataFrame, to_datetime, read_csv, merge, get_dummies, concat, Series
from numpy import nan
import statsmodels.formula.api as smf
from pathlib import Path
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


figures = Path(__file__).parent / "figures"
figures.mkdir(parents=True, exist_ok=True)
rawdata = Path(__file__).parent / "data" / "stable-isotopes-no-outliers.csv"
environment_file = Path(__file__).parent / "data" / "temperature-and-light.csv"

df = read_csv(
    rawdata,
    header=0,
    usecols=[
        "Collection Date",
        "Gear Type",
        "Tissue Type",
        "d15N",
        "d13C",
        "C/N (Molar)",
        "Date Run",
    ],
)


for i in range(len(df["Date Run"])):
    if df["Date Run"][i] == "9/6/23":
        df.drop(i, inplace=True)
    else:
        to_datetime(df["Date Run"][i], format="%m/%d/%y")


df = df.dropna(
    subset=["Gear Type", "Tissue Type"]
)
data_muscle = df[df["Tissue Type"] == "M"]

env = read_csv(environment_file)
env["Date-Time (EDT)"] = to_datetime(env["Date-Time (EDT)"])
env["Date"] = env["Date-Time (EDT)"].dt.date

# Average hourly environmental data into daily means
env_daily = env.groupby("Date").mean(numeric_only=True).reset_index()

# lmm with light (therefore no wild scallops included)
lmm_data = data_muscle[
    ["d13C", "d15N", "C/N (Molar)", "Gear Type", "Collection Date"]
]
lmm_data = lmm_data.rename(columns={"Gear Type": "Gear"})
lmm_data = lmm_data[lmm_data["Gear"].isin(["C", "N", "W"])]

lmm_data["Date"] = to_datetime(
    "2023-" + lmm_data["Collection Date"].astype(int).astype(str).str.zfill(2) + "-15",
    format="%Y-%m-%d",
).dt.date

lmm_data = merge(lmm_data, env_daily, on="Date", how="left")


def assign_environment(row):
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

lmm_data[["Temperature", "Light"]] = lmm_data.apply(assign_environment, axis=1)
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

# Linear Mixed Models: Run for each isotopic variance
model_d13C = smf.mixedlm(
    "d13C ~ Temperature + Light + C(Gear)",
    data=lmm_clean_with_light,
    groups=lmm_clean_with_light["Collection Date"],
)
result_d13C = model_d13C.fit()
print(result_d13C.summary())
model_d15N = smf.mixedlm(
    "d15N ~ Temperature + Light + C(Gear)",
    data=lmm_clean_with_light,
    groups=lmm_clean_with_light["Collection Date"],
)
result_d15N = model_d15N.fit()
print(result_d15N.summary())
model_CN = smf.mixedlm(
    "Q('C/N (Molar)') ~ Temperature + Light + C(Gear)",
    data=lmm_clean_with_light,
    groups=lmm_clean_with_light["Collection Date"],
)
result_CN = model_CN.fit()
print(result_CN.summary())
light_table = concat(
    [
        lmm_results_table(result_d13C, "d13C"),
        lmm_results_table(result_d15N, "d15N"),
        lmm_results_table(result_CN, "C/N"),
    ]
)
light_table.to_csv(figures / "lmm_light_table.csv", index=False)


# d13C model
model_d13C_env = smf.mixedlm(
    "d13C ~ Temperature + C(Gear)",
    data=lmm_clean_temp_only,
    groups=lmm_clean_temp_only["Collection Date"],
)
result_d13C_env = model_d13C_env.fit()
print(result_d13C_env.summary())
# d15N model
model_d15N_env = smf.mixedlm(
    "d15N ~ Temperature + C(Gear)",
    data=lmm_clean_temp_only,
    groups=lmm_clean_temp_only["Collection Date"],
)
result_d15N_env = model_d15N_env.fit()
print(result_d15N_env.summary())
# C/N model
model_CN_env = smf.mixedlm(
    "Q('C/N (Molar)') ~ Temperature + C(Gear)",
    data=lmm_clean_temp_only,
    groups=lmm_clean_temp_only["Collection Date"],
)
result_CN_env = model_CN_env.fit()
print(result_CN_env.summary())

# temperature only
temp_table = concat(
    [
        lmm_results_table(result_d13C_env, "d13C"),
        lmm_results_table(result_d15N_env, "d15N"),
        lmm_results_table(result_CN_env, "C/N"),
    ]
)
temp_table = temp_table.round(3)
temp_table.to_csv(figures / "lmm_temperature_table.csv", index=False)
