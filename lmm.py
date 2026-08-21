import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from pathlib import Path

figures = Path(__file__).parent / "figures"
figures.mkdir(parents=True, exist_ok=True)
rawdata = Path(__file__).parent / "data" / "stable-isotopes-no-outliers.csv"

df = pd.read_csv(rawdata, header=0)
print(df.columns.tolist())

df = pd.read_csv(rawdata, header=0, usecols = [
    'Analysis',
    'Sample ID',
    'Collection Date',
    'Gear Type',
    'Sex',
    'Tissue Type',
    'Number in gear type',
    'Mass (mg)',
    '% N',
    'N (umoles)',
    'd15N',
    '%C',
    'C (umoles)',
    'd13C',
    'C/N (Molar)',
    'Date Run'])


for i in range(len(df['Date Run'])):
    if df['Date Run'][i] == '9/6/23':
        df.drop(i, inplace=True)
    else:
        pd.to_datetime(df['Date Run'][i], format = '%m/%d/%y')


df.dropna(subset=['Gear Type'], inplace=True) #only scallops and filters are being plotted

data_muscle = df.dropna(subset = ['Tissue Type'])
data_muscle = data_muscle.drop(data_muscle[data_muscle['Tissue Type'] == 'G'].index)

pd.DataFrame(data_muscle)

environment_file = Path(__file__).parent / "data" / "temperature-and-light.csv"
env = pd.read_csv(environment_file)
env["Date-Time (EDT)"] = pd.to_datetime(
    env["Date-Time (EDT)"]
)
# Extract date
env["Date"] = env["Date-Time (EDT)"].dt.date

# Average hourly environmental data into daily means
env_daily = (
    env.groupby("Date")
    .mean(numeric_only=True)
    .reset_index()
)

#lmm with light (therefore no wild scallops included)
lmm_data = data_muscle[
    [
        "d13C",
        "d15N",
        "C/N (Molar)",
        "Gear Type",
        "Collection Date"
    ]
].copy()
lmm_data = lmm_data.rename(columns={"Gear Type": "Gear"})
lmm_data = lmm_data[lmm_data["Gear"].isin(["C", "N", "W"])]
print(lmm_data["Gear"].value_counts())

lmm_data["Date"] = pd.to_datetime(
    "2023-" +
    lmm_data["Collection Date"].astype(int).astype(str).str.zfill(2) +
    "-15",
    format="%Y-%m-%d"
).dt.date

lmm_data = pd.merge(
    lmm_data,
    env_daily,
    on="Date",
    how="left"
)

def assign_environment(row):
    if row["Gear"] == "C":
        return pd.Series({
            "Temperature": row["Cage, Temperature (°F)"],
            "Light": row["Cage, Light (lum)"]
        })
    elif row["Gear"] == "N":
        return pd.Series({
            "Temperature": row["Net Bottom, Temperature (°F)"],
            "Light": row["Net Bottom, Light (lum)"]
        })
    elif row["Gear"] == "W":
        return pd.Series({
            "Temperature": row["Wild, Temperature (°F)"],
            "Light": np.nan
        })

lmm_data[
    ["Temperature", "Light"]
] = lmm_data.apply(assign_environment, axis=1)
print(lmm_data.head())
print(lmm_data.isna().sum())
print(lmm_data["Gear"].value_counts())
print(lmm_data["Collection Date"].value_counts())

lmm_clean = lmm_data[
    [
        "d13C",
        "d15N",
        "C/N (Molar)",
        "Gear",
        "Collection Date",
        "Temperature",
        "Light"
    ]
].dropna()

print("LMM samples retained:", len(lmm_clean))
print(lmm_clean.isna().sum())

model_d13C = smf.mixedlm(
    "d13C ~ Temperature + Light + C(Gear)",
    data=lmm_clean,
    groups=lmm_clean["Collection Date"]
)

result_d13C = model_d13C.fit()
print(result_d13C.summary())


model_d15N = smf.mixedlm(
    "d15N ~ Temperature + Light + C(Gear)",
    data=lmm_clean,
    groups=lmm_clean["Collection Date"]
)
result_d15N = model_d15N.fit()
print(result_d15N.summary())
model_CN = smf.mixedlm(
    "Q('C/N (Molar)') ~ Temperature + Light + C(Gear)",
    data=lmm_clean,
    groups=lmm_clean["Collection Date"]
)
result_CN = model_CN.fit()
print(result_CN.summary())

def linear_mixed_model(df):
    subset = df[
        [
            Dimension.NITROGEN_FRACTIONATION.value,
            Dimension.GEAR.value,
            Dimension.COLLECTION_DATE.value,
            Dimension.TISSUE.value,
        ]
    ].dropna()

    subset = subset.rename(columns={
        Dimension.NITROGEN_FRACTIONATION.value: "d15N",
        Dimension.GEAR.value: "Gear",
        Dimension.COLLECTION_DATE.value: "Month",
        Dimension.TISSUE.value: "Tissue",
    })

    model = mixedlm(
        "d15N ~ C(Gear) + C(Tissue)",
        data=subset,
        groups=subset["Month"],
    )
    result = model.fit()
    print(result.summary())

#lmm with no light therefore wild scallops included

lmm_env = data_muscle[
    [
        "d13C",
        "d15N",
        "C/N (Molar)",
        "Gear Type",
        "Collection Date"
    ]
].copy()
lmm_env = lmm_env.rename(columns={"Gear Type": "Gear"})

# Create dates
lmm_env["Date"] = pd.to_datetime(
    "2023-" +
    lmm_env["Collection Date"].astype(int).astype(str).str.zfill(2) +
    "-15",
    format="%Y-%m-%d"
).dt.date

# Merge environment data
lmm_env = pd.merge(
    lmm_env,
    env_daily,
    on="Date",
    how="left"
)

def assign_environment(row):
    if row["Gear"] == "C":
        return pd.Series({
            "Temperature": row["Cage, Temperature (°F)"],
            "Light": row["Cage, Light (lum)"]
        })

    elif row["Gear"] == "N":
        return pd.Series({
            "Temperature": row["Net Bottom, Temperature (°F)"],
            "Light": row["Net Bottom, Light (lum)"]
        })

    elif row["Gear"] == "W":
        return pd.Series({
            "Temperature": row["Wild, Temperature (°F)"],
            "Light": np.nan
        })

lmm_env[["Temperature", "Light"]] = lmm_env.apply(
    assign_environment,
    axis=1
)

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
    groups=lmm_env_clean["Collection Date"]
)
result_d13C_env = model_d13C_env.fit()
print(result_d13C_env.summary())

# d15N model
model_d15N_env = smf.mixedlm(
    "d15N ~ Temperature + C(Gear)",
    data=lmm_env_clean,
    groups=lmm_env_clean["Collection Date"]
)
result_d15N_env = model_d15N_env.fit()
print(result_d15N_env.summary())
# C/N model
model_CN_env = smf.mixedlm(
    "Q('C/N (Molar)') ~ Temperature + C(Gear)",
    data=lmm_env_clean,
    groups=lmm_env_clean["Collection Date"]
)
result_CN_env = model_CN_env.fit()
print(result_CN_env.summary())

#temp data not merging properly so need 
# to look into this! also because wild temp
# doesnt exist for the first month only 4 months are being used.