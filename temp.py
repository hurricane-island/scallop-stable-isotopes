from pathlib import Path
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots
from matplotlib import patches as mpatches
import seaborn as sns
from enum import Enum
from matplotlib.dates import DateFormatter


figures = Path(__file__).parent / "figures"
rawdata = Path(__file__).parent / "data" / "2023_2022_GSI_Environmental_Data_2023_Temperature_and_Light.csv"

class Dimension(Enum):

    DATE = "Date-Time (EDT)"
    NET_TOP_TEMP = "Net Top, Temperature (°F)"
    CAGE_TEMP = "Cage, Temperature (°F)"
    CAGE_LUM = "Cage, Light (lum)"
    NET_BOTTOM_TEMP = "Net Bottom, Temperature (°F)"
    NET_BOTTOM_LUM = "Net Bottom, Light (lum)"
    WILD_TEMP = "Wild, Temperature (°F)"

if __name__ == "__main__":
    df = pd.read_csv(
        rawdata,
        header=0,
        usecols=[
                Dimension.DATE.value, 
                Dimension.NET_TOP_TEMP.value, 
                Dimension.CAGE_TEMP.value, 
                Dimension.CAGE_LUM.value, 
                Dimension.NET_BOTTOM_TEMP.value, 
                Dimension.NET_BOTTOM_LUM.value,
                Dimension.WILD_TEMP.value])
    pd.to_datetime(df[Dimension.DATE.value],format = '%m/%d/%y %H:%M')
    df_temp = df[[Dimension.DATE.value, Dimension.NET_BOTTOM_TEMP.value,Dimension.CAGE_TEMP.value,Dimension.WILD_TEMP.value]]
    df_temp = df_temp.melt(id_vars = Dimension.DATE.value, var_name='Gear', value_name='Temp')
    custom_color = ('black', 'blue', 'red')
    fig, ax = subplots(figsize=(10, 8))
    sns.lineplot(
        data = df_temp,
        x = Dimension.DATE.value,
        y = 'Temp',
        hue = 'Gear',
        palette=custom_color,
        legend = False, #depending on how you want the legend to look, use this or replace with False and plt.legend below
    )
    plt.xticks((df_temp[Dimension.DATE.value].iloc[1],
                df_temp[Dimension.DATE.value].iloc[673],
                df_temp[Dimension.DATE.value].iloc[1441],
                df_temp[Dimension.DATE.value].iloc[2209],
                df_temp[Dimension.DATE.value].iloc[2857]),
                rotation=45, 
                ha='right')
    plt.xlabel("Date")
    plt.ylabel("Temperature (F)")
    plt.legend(handles=[
        mpatches.Patch(color='black', label='Nets'),
        mpatches.Patch(color='blue', label='Cages'),
        mpatches.Patch(color= 'red', label = 'Wild')],
        loc = 'upper right')
    fig.savefig(f"{figures}/temp_series.png")
    

july_cage = round(df[Dimension.CAGE_TEMP.value].iloc[673-384:673].mean(), 2)
aug_cage = round(df[Dimension.CAGE_TEMP.value].iloc[1057:1441].mean(), 2)
sept_cage = round(df[Dimension.CAGE_TEMP.value].iloc[2209-384:2209].mean(), 2)
oct_cage = round(df[Dimension.CAGE_TEMP.value].iloc[2857-384:2857].mean(), 2)

july_net = round(df[Dimension.NET_BOTTOM_TEMP.value].iloc[673-384:673].mean(), 2)
aug_net = round(df[Dimension.NET_BOTTOM_TEMP.value].iloc[1057:1441].mean(), 2)
sept_net = round(df[Dimension.NET_BOTTOM_TEMP.value].iloc[2209-384:2209].mean(), 2)
oct_net = round(df[Dimension.NET_BOTTOM_TEMP.value].iloc[2857-384:2857].mean(), 2)

july_wild = round(df[Dimension.WILD_TEMP.value].iloc[673-384:673].mean(), 2)
aug_wild = round(df[Dimension.WILD_TEMP.value].iloc[1057:1441].mean(), 2)
sept_wild = round(df[Dimension.WILD_TEMP.value].iloc[2209-384:2209].mean(), 2)
oct_wild = round(df[Dimension.WILD_TEMP.value].iloc[2857-384:2857].mean(), 2)



summary = [
        [july_net, aug_net, sept_net, oct_net],
        [july_cage, aug_cage, sept_cage, oct_cage],
        [july_wild, aug_wild, sept_wild, oct_wild],
    ]
    
fig, ax = plt.subplots()
ax.axis('off')
ax.table(cellText=summary,
                 colLabels= ['July', 'August', 'Sept', 'Oct'],
                 rowLabels = ['Nets', 'Cages', 'Wild'],
                 cellLoc = 'center',
                 loc='center')

plt.savefig(figures / "temp_avg.png")
    
degree_hour = 55.4
july_cage_count = 0
july_net_count = 0 
july_wild_count = 0
aug_cage_count = 0
aug_net_count = 0 
aug_wild_count = 0
sept_cage_count = 0
sept_net_count = 0 
sept_wild_count = 0
oct_cage_count = 0
oct_net_count = 0 
oct_wild_count = 0

for i in range(289,673):
    if df[Dimension.CAGE_TEMP.value][i] > degree_hour:
        july_cage_count = july_cage_count + 1
    if df[Dimension.NET_BOTTOM_TEMP.value][i] > degree_hour:
        july_net_count = july_net_count + 1
    if df[Dimension.WILD_TEMP.value][i] > degree_hour:
        july_wild_count = july_wild_count + 1

for i in range(1057,1441):
    if df[Dimension.CAGE_TEMP.value][i] > degree_hour:
        aug_cage_count = aug_cage_count + 1
    if df[Dimension.NET_BOTTOM_TEMP.value][i] > degree_hour:
        aug_net_count = aug_net_count + 1
    if df[Dimension.WILD_TEMP.value][i] > degree_hour:
        aug_wild_count = aug_wild_count + 1

for i in range(1825,2209):
    if df[Dimension.CAGE_TEMP.value][i] > degree_hour:
        sept_cage_count = sept_cage_count + 1
    if df[Dimension.NET_BOTTOM_TEMP.value][i] > degree_hour:
        sept_net_count = sept_net_count + 1
    if df[Dimension.WILD_TEMP.value][i] > degree_hour:
        sept_wild_count = sept_wild_count + 1

for i in range(2473,2857):
    if df[Dimension.CAGE_TEMP.value][i] > degree_hour:
        oct_cage_count = oct_cage_count + 1
    if df[Dimension.NET_BOTTOM_TEMP.value][i] > degree_hour:
        oct_net_count = oct_net_count + 1
    if df[Dimension.WILD_TEMP.value][i] > degree_hour:
        oct_wild_count = oct_wild_count + 1


summary = [
        [july_net_count, aug_net_count, sept_net_count, oct_net_count],
        [july_cage_count, aug_cage_count, sept_cage_count, oct_cage_count],
        [july_wild_count, aug_wild_count, sept_wild_count, oct_wild_count],
    ]
    
fig, ax = plt.subplots()
ax.axis('off')
ax.table(cellText=summary,
                 colLabels= ['July', 'August', 'Sept', 'Oct'],
                 rowLabels = ['Nets', 'Cages', 'Wild'],
                 cellLoc = 'center',
                 loc='center')

plt.savefig(figures / "temp_degreehours.png")