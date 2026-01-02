from pathlib import Path
from pandas import read_csv, to_datetime
from matplotlib import patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots, savefig, subplot
from stats_analysis import Dimension, quantize_categorical_column
import seaborn as sns
import numpy as np
from numpy import unique
from stats_analysis import quantize_categorical_column, return_column_to_categorical
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from scipy import stats


figures = Path(__file__).parent / "figures"
raw_data = Path(__file__).parent / "data" / "2023IsotopeDataReport-no-outliers.csv"
bad_run_dates = {"9/6/23"}  # use Set() as more generic lookup than single value


# When file is run directly, this block will execute.
# The reason to do this is so that as the methods are wrapped as functions
# and imported into other scripts, this block will not execute, when that
# happens.
if __name__ == "__main__":
    df = read_csv(
        raw_data,
        header=0,
        usecols=[
            Dimension.COLLECTION_DATE.value,
            Dimension.GEAR.value,
            Dimension.SEX.value,
            Dimension.TISSUE.value,
            "Mass (mg)", # possible PCA input
            Dimension.NITROGEN_PERCENTAGE.value,
            "N (umoles)", # possible PCA input if N limited?
            Dimension.NITROGEN_FRACTIONATION.value,
            Dimension.CARBON_PERCENTAGE.value,
            "C (umoles)", # possible PCA input if N limited?
            Dimension.CARBON_FRACTIONATION.value,
            Dimension.MOLAR_RATIO.value,
            Dimension.DATE_RUN.value,
        ],
    )

    # Remove known contaminated samples
    for date in range(len(df[Dimension.DATE_RUN.value])):
        if df[Dimension.DATE_RUN.value][date] in bad_run_dates:
            df.drop(date, inplace=True)
        else:
            to_datetime(df[Dimension.DATE_RUN.value][date], format="%m/%d/%y")

    # Don't need the date run for analysis, only pre-filtering
    df = df.drop(columns=[
        Dimension.DATE_RUN.value
    ])

    # only scallops and filters are being plotted
    df.dropna(subset=[Dimension.GEAR.value], inplace=True)

    tissue = df.dropna(subset=[Dimension.TISSUE.value])
    mask = tissue[Dimension.TISSUE.value] == "G"
    data_muscle = tissue.drop(tissue[mask].index)

    '''
    Pairplots of raw data
    change hue and columns as needed to color by different categories
    '''
    df = quantize_categorical_column(df, Dimension.GEAR.value, {"C": 1, "N": 2, "W": 3}) #ALL GEAR TYPES
    df = return_column_to_categorical(df, Dimension.GEAR.value, {1: "Cage", 2: 'Net', 3: "Wild"})

    data_muscle = quantize_categorical_column(data_muscle, Dimension.GEAR.value, {"C": 1, "N": 2, "W": 3}) #ALL GEAR TYPES
    data_muscle = return_column_to_categorical(data_muscle, Dimension.GEAR.value, {1: "Cage", 2: 'Net', 3: "Wild"})


    cages_muscle = data_muscle[data_muscle[Dimension.GEAR.value]== "Cage"]
    nets_muscle = data_muscle[data_muscle[Dimension.GEAR.value]== "Net"]
    wild_muscle = data_muscle[data_muscle[Dimension.GEAR.value]== "Wild"]


    custom_colors = ('black', "blue", "red")
    fig, ax = subplots(figsize=(10, 8))
    sns.scatterplot(
        x=wild[Dimension.MOLAR_RATIO.value],
        y=wild[Dimension.CARBON_FRACTIONATION.value],
        # hue = data_muscle[Dimension.GEAR.value],
        # palette = custom_colors,
        # style = data_muscle[Dimension.GEAR.value],
        # edgecolor = 'black',
        # facecolor = 'black',
        legend = 'auto',
        s=30,
        ax=ax
    )
    fig.savefig(f"{figures}/wild_lipid_extraction.png")

    
    
    custom_colors = ('black', "blue", "red")
    fig, ax = subplots(figsize=(10, 8))
    sns.scatterplot(
        x=df[Dimension.COLLECTION_DATE.value],
        y=df[Dimension.MOLAR_RATIO.value],
        hue = df[Dimension.GEAR.value],
        palette = custom_colors,
        style = df[Dimension.TISSUE.value],
        # edgecolor = 'black',
        # facecolor = 'black',
        legend = 'auto',
        s=30,
        ax=ax
    )
    fig.savefig(f"{figures}/CN_timeline.png")


    pairplot = sns.pairplot(df[[
        Dimension.NITROGEN_FRACTIONATION.value, 
        Dimension.CARBON_FRACTIONATION.value,
        Dimension.MOLAR_RATIO.value, 
        Dimension.GEAR.value]], 
        hue=Dimension.GEAR.value, 
        palette=custom_colors,)
    plt.savefig(figures / "new-pairplot-all-tissue-gear.png")



    '''
    Scatter plot of d13C vs d15N colored by farm vs wild with filters separated by month
    '''
    
    data_muscle = quantize_categorical_column(data_muscle, 
                                              Dimension.GEAR.value, 
                                              {"C": 1, "N": 2, "W": 3}) #FARMED VS WILD "CF": 4, "NF": 4, "WF":3
    
    cages = data_muscle[data_muscle[Dimension.GEAR.value] == 1]
    nets = data_muscle[data_muscle[Dimension.GEAR.value] == 2]
    wild = data_muscle[data_muscle[Dimension.GEAR.value] == 3]

    # print(wild[Dimension.CARBON_FRACTIONATION.value].mean())
    # print(wild[Dimension.NITROGEN_FRACTIONATION.value].mean())
    # print(wild[Dimension.MOLAR_RATIO.value].mean())
    
    print(data_muscle[Dimension.CARBON_FRACTIONATION.value].describe())
    print(data_muscle[Dimension.NITROGEN_FRACTIONATION.value].describe())
    print(data_muscle[Dimension.MOLAR_RATIO.value].describe())


    june = data_muscle[data_muscle[Dimension.COLLECTION_DATE.value] == 6]
    july = data_muscle[data_muscle[Dimension.COLLECTION_DATE.value] == 7]
    august = data_muscle[data_muscle[Dimension.COLLECTION_DATE.value] == 8]
    september = data_muscle[data_muscle[Dimension.COLLECTION_DATE.value] == 9]
    october = data_muscle[data_muscle[Dimension.COLLECTION_DATE.value] == 10]
    
    june_series = june[Dimension.GEAR.value] 
    july_series = july[Dimension.GEAR.value] 
    august_series = august[Dimension.GEAR.value] 
    september_series = september[Dimension.GEAR.value] 
    october_series = october[Dimension.GEAR.value] 

    
    fig, ax = subplots(1, 5, figsize =(10,3), sharex=False, sharey = False)
    for x in june_series:
        plt.subplot(1, 5, 1)
        plt.scatter(june[Dimension.MOLAR_RATIO.value],
                    june[Dimension.CARBON_FRACTIONATION.value], 
                    c = june_series, 
                    marker = 'x', 
                    cmap = 'tab10')
        plt.title("June")
        plt.xlim(3,6)
        plt.ylim(-19,-16)
        plt.xlabel('C/N')
        plt.ylabel('d13C')
    for x in july_series:
        plt.subplot(1, 5, 2)
        plt.scatter(july[Dimension.MOLAR_RATIO.value],
                    july[Dimension.CARBON_FRACTIONATION.value], 
                    c = july_series, 
                    marker = 'x', 
                    cmap = 'tab10')
        plt.title("July")
        plt.xlim(3,6)
        plt.ylim(-19,-16)
        plt.xlabel('C/N')
    for x in august_series:
        plt.subplot(1, 5, 3)
        plt.scatter(august[Dimension.MOLAR_RATIO.value],
                    august[Dimension.CARBON_FRACTIONATION.value],
                     c = august_series, 
                     marker = 'x',
                    cmap = 'tab10')
        plt.title("August")
        plt.xlim(3,6)
        plt.ylim(-19,-16)
        plt.xlabel('C/N')
    for x in september_series:
        plt.subplot(1, 5, 4)
        plt.scatter(september[Dimension.MOLAR_RATIO.value],
                    september[Dimension.CARBON_FRACTIONATION.value], 
                    c = september_series, 
                    marker = 'x', 
                    cmap = 'tab10')
        plt.title("September")
        plt.xlim(3,6)
        plt.ylim(-19,-16)
        plt.xlabel('C/N')
    for x in october_series:
        plt.subplot(1, 5, 5)
        plt.scatter(october[Dimension.MOLAR_RATIO.value],
                    october[Dimension.CARBON_FRACTIONATION.value], 
                    c = october_series, 
                    marker = 'x', 
                    cmap = 'tab10')
        plt.title("October")
        plt.xlim(3,6)
        plt.ylim(-19,-16)
        plt.xlabel('C/N')
    plt.legend(handles=[
        mpatches.Patch(color='tab:blue', label='Farm'),
        mpatches.Patch(color='tab:red', label='Wild'),
        mpatches.Patch(color='tab:cyan', label='Farm Filter'),
        mpatches.Patch(color='tab:pink', label='Wild Filter')], 
        bbox_to_anchor=(1.05, 1))
    fig.savefig(figures / "rawdata_scatter_monthly_gear.png")
    

    fig, ax = subplots(figsize=(10, 8))
    sns.scatterplot(
        x=data_muscle[Dimension.MOLAR_RATIO.value],
        y=data_muscle[Dimension.CARBON_FRACTIONATION.value],
        hue = data_muscle[Dimension.GEAR.value],
        palette='tab10',
        style= data_muscle[Dimension.COLLECTION_DATE.value],
        legend = 'auto', #depending on how you want the legend to look, use this or replace with False and plt.legend below
        s=150,
    )
    plt.xlabel("C/N")
    plt.ylabel("d13C")
    fig.savefig(f"{figures}/rawdata_scatter_gear_monthly.png")

    '''
    CHECKING FOR OUTLIERS in DATASET WITH A Z-SCORE GREATER THAN 3
    '''

    new_df = data_muscle[[Dimension.MOLAR_RATIO.value, 
                          Dimension.CARBON_FRACTIONATION.value,
                          Dimension.COLLECTION_DATE.value, 
                          Dimension.NITROGEN_FRACTIONATION.value,
                          Dimension.GEAR.value, 
                          Dimension.SEX.value ]]
    

    z = np.abs(stats.zscore(data_muscle[Dimension.CARBON_FRACTIONATION.value]))
    a = np.abs(stats.zscore(data_muscle[Dimension.NITROGEN_FRACTIONATION.value]))
    b = np.abs(stats.zscore(data_muscle[Dimension.MOLAR_RATIO.value]))
    data_muscle['d13C z score'] = z
    data_muscle['d15N z score'] = a
    data_muscle['C/N z score'] = b
    d13C_outliers = data_muscle.loc[data_muscle['d13C z score'] > 3]
    d15N_outliers = data_muscle.loc[data_muscle['d15N z score'] > 3]
    CN_outliers = data_muscle.loc[data_muscle['C/N z score'] > 3]
    print(d13C_outliers)
    print(d15N_outliers)
    print(CN_outliers)

    '''
    Dataset was saved after removing outliers. Clean data: 2023IsotopeDataReport-no-outliers.csv 
    '''
