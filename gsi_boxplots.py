'''
This looks at the GSI data to determine the time of year of spawning for gear types 
and how that aligns with isotopic signatures
'''


from pathlib import Path
from typing import Dict
from enum import Enum
from pandas import DataFrame, read_csv, to_datetime, set_option
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots, savefig, subplot
import numpy as np
from matplotlib import patches as mpatches


figures = Path(__file__).parent / "figures"
raw_data = Path(__file__).parent / "data" / "2023_StableIsotope_GSI_data.csv"

class Dimension(Enum):
    """
    Let the linter help with making sure we use consistent column names.
    Only needed for columns that are used many times, where a typo is more likely
    from copy-pasting.
    """

    COLLECTION_DATE = "Month"
    GEAR = "Gear Type"
    SEX = "Sex"
    SHELL_HEIGHT = "Shell_Height"
    TOTAL_VISCERA_WEIGHT = "Total_Viscera_Weight"
    MUSCLE_WEIGHT = "Meat_Weight"
    GONAD_WEIGHT = "Gonad_Weight"
    GSI = "GSI"

if __name__ == "__main__":
    df = read_csv(
        raw_data,
        header=0,
        usecols=[
            Dimension.COLLECTION_DATE.value,
            Dimension.GEAR.value,
            Dimension.SEX.value,
            Dimension.SHELL_HEIGHT.value,
            Dimension.TOTAL_VISCERA_WEIGHT.value,
            Dimension.MUSCLE_WEIGHT.value,
            Dimension.GONAD_WEIGHT.value,
            Dimension.GSI.value
        ],
    )

    df = df.dropna(subset=[Dimension.GSI.value, Dimension.COLLECTION_DATE.value, Dimension.GEAR.value])

    # Separate by gear type
    cage = df[df[Dimension.GEAR.value] == 'C']
    net = df[df[Dimension.GEAR.value] == 'N']
    farm = df[df[Dimension.GEAR.value] != 'W']
    wild = df[df[Dimension.GEAR.value] == 'W']

    # Separate by gear type and month
    july_cage = cage[cage[Dimension.COLLECTION_DATE.value] == 7]
    august_cage = cage[cage[Dimension.COLLECTION_DATE.value] == 8]
    sept_cage = cage[cage[Dimension.COLLECTION_DATE.value] == 9]
    oct_cage = cage[cage[Dimension.COLLECTION_DATE.value] == 10]

    july_net = net[net[Dimension.COLLECTION_DATE.value] == 7]
    august_net = net[net[Dimension.COLLECTION_DATE.value] == 8]
    sept_net = net[net[Dimension.COLLECTION_DATE.value] == 9]
    oct_net = net[net[Dimension.COLLECTION_DATE.value] == 10]

    july_wild = wild[wild[Dimension.COLLECTION_DATE.value] == 7]
    august_wild = wild[wild[Dimension.COLLECTION_DATE.value] == 8]
    sept_wild = wild[wild[Dimension.COLLECTION_DATE.value] == 9]
    oct_wild = wild[wild[Dimension.COLLECTION_DATE.value] == 10]

    july_farm = farm[farm[Dimension.COLLECTION_DATE.value] == 7]
    august_farm = farm[farm[Dimension.COLLECTION_DATE.value] == 8]
    sept_farm = farm[farm[Dimension.COLLECTION_DATE.value] == 9]
    oct_farm = farm[farm[Dimension.COLLECTION_DATE.value] == 10]

    # Boxplot design
    box_properties_1 = dict(facecolor = 'white', color='black', linewidth=1)
    median_properties_1 = dict(color='black', linewidth=1.5)
    whisker_properties_1 = dict(color='black')
    cap_properties_1 = dict(color='black', linewidth=1)
    flier_properties_1 = dict(marker='o', color = 'black', markerfacecolor='black', markersize=2)

    box_properties_2 = dict(facecolor = 'white', color='red', linewidth=1)
    median_properties_2 = dict(color='red', linewidth=1.5)
    whisker_properties_2 = dict(color='red')
    cap_properties_2 = dict(color='red', linewidth=1)
    flier_properties_2 = dict(marker='o', color = 'red', markerfacecolor='red', markersize=2)
    

    plt.boxplot([july_farm[Dimension.GSI.value], august_farm[Dimension.GSI.value], sept_farm[Dimension.GSI.value], oct_farm[Dimension.GSI.value]], 
                positions = [1,2,3,4], 
                widths = 0.3,
                patch_artist=True, 
                boxprops=box_properties_1, 
                medianprops=median_properties_1,
                whiskerprops=whisker_properties_1,
                capprops=cap_properties_1,
                flierprops=flier_properties_1)
    plt.boxplot([july_wild[Dimension.GSI.value], august_wild[Dimension.GSI.value], sept_wild[Dimension.GSI.value], oct_wild[Dimension.GSI.value]],
                positions = [1.3,2.3,3.3,4.3], 
                widths = 0.3,
                patch_artist=True, 
                boxprops=box_properties_2, 
                medianprops=median_properties_2,
                whiskerprops=whisker_properties_2,
                capprops=cap_properties_2,
                flierprops=flier_properties_2)
    plt.xticks([1.15,2.15,3.15,4.15], ['July', 'August', 'September', 'October'])
    plt.ylim(0, 35)
    plt.ylabel('GSI')
    plt.legend(handles = [
            mpatches.Patch(color='black', label='Farm'),
            mpatches.Patch(color='red', label='Wild'),
    ])
    plt.savefig(figures / "GSI_farm_vs_wild_boxplot.png")

