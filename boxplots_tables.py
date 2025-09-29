import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.patches as mpatches

figures = Path(__file__).parent / "figures"
rawdata = Path(__file__).parent / "data" / "2023IsotopeDataReport-cleanedinexcel.csv"


data = pd.read_csv(rawdata, header=0, usecols = [
    'Analysis', 
    'Sample ID', 
    'Collection Date',
    'Gear Type',		
    'Sex', 	
    'Tissue Type (Gonad or Muscle)',	
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


for i in range(len(data['Date Run'])):
    if data['Date Run'][i] == '9/6/23':
        data.drop(i, inplace=True)
    else:
        pd.to_datetime(data['Date Run'][i], format = '%m/%d/%y')
        pass
    # 9/6/23 samples were contaiminated 

data.dropna(subset=['Gear Type'], inplace=True) #only scallops and filters are being plotted

data.insert(5, 'Farmed or Wild', 0)
mapping_farmvswild = {'C': 'F', 'N': 'F', 'W': 'W'}
data['Farmed or Wild'] = data['Gear Type'].map(mapping_farmvswild) 

data_muscle = data.dropna(subset = ['Tissue Type (Gonad or Muscle)'])
data_muscle = data_muscle.drop(data_muscle[data_muscle['Tissue Type (Gonad or Muscle)'] == 'G'].index)

data_gonad = data.dropna(subset = ['Tissue Type (Gonad or Muscle)'])
data_gonad = data_gonad.drop(data_gonad[data_gonad['Tissue Type (Gonad or Muscle)'] == 'M'].index)

data_muscle_female = data_muscle.drop(data_muscle[data_muscle['Sex']=='M'].index)
data_muscle_male = data_muscle.drop(data_muscle[data_muscle['Sex']=='F'].index)

data_gonad_female = data_gonad.drop(data_gonad[data_gonad['Sex']=='M'].index)
data_gonad_male = data_gonad.drop(data_gonad[data_gonad['Sex']=='F'].index)

colormap = {
    'C': 'b', 
    'N': 'g', 
    'W': 'r', 
    'CF' : 'b',
    'NF' : 'g', 
    'WF' : 'r'
}
markermap = {
     'C': 'o', 
    'N': 'o', 
    'W': 'o', 
    'CF' : 'x',
    'NF' : 'x', 
    'WF' : 'x'
}

'''
In the line below, change data_gonad to data_muscle if you want to plot muscle data
'''
mapping = {'C': 'B', 'N': 'T', 'W': 'B'}
data_muscle['Column Height'] = data_muscle['Gear Type'].map(mapping) #C and W are at the same depth

monthly = data_muscle.groupby('Collection Date') #GONAD OR MUSCLE!!!
june = monthly.get_group(6)
july = monthly.get_group(7)
august = monthly.get_group(8)
sept = monthly.get_group(9)
october = monthly.get_group(10)

june_df = pd.DataFrame(june)
july_df = pd.DataFrame(july)
august_df = pd.DataFrame(august)
sept_df = pd.DataFrame(sept)
oct_df = pd.DataFrame(october)

# jgear = june_df.groupby('Gear Type')
# j_cage = jgear.get_group('C')
# j_net = jgear.get_group('N')
# j_wild = jgear.get_group('W')
# j_cage_filt = jgear.get_group('CF')
# j_net_filt = jgear.get_group('NF')
# j_wild_filt = jgear.get_group('WF')


# jugear = july_df.groupby('Gear Type')
# ju_cage = jugear.get_group('C')
# ju_net = jugear.get_group('N')
# ju_wild = jugear.get_group('W')
# ju_cage_filt = jugear.get_group('CF')
# ju_net_filt = jugear.get_group('NF')
# ju_wild_filt = jugear.get_group('WF')

# agear = august_df.groupby('Gear Type')
# a_cage = agear.get_group('C')
# a_net = agear.get_group('N')
# a_wild = agear.get_group('W')
# a_cage_filt = agear.get_group('CF')
# a_net_filt = agear.get_group('NF')
# a_wild_filt = agear.get_group('WF')

# sgear = sept_df.groupby('Gear Type')
# s_cage = sgear.get_group('C')
# s_net = sgear.get_group('N')
# s_wild = sgear.get_group('W')
# s_cage_filt = sgear.get_group('CF')
# s_net_filt = sgear.get_group('NF')
# s_wild_filt = sgear.get_group('WF')

# ogear = oct_df.groupby('Gear Type')
# o_cage = ogear.get_group('C')
# o_net = ogear.get_group('N')
# o_wild = ogear.get_group('W')
# o_cage_filt = ogear.get_group('CF')
# o_net_filt = ogear.get_group('NF')
# o_wild_filt = ogear.get_group('WF')

june_df = pd.DataFrame(june)
july_df = pd.DataFrame(july)
august_df = pd.DataFrame(august)
sept_df = pd.DataFrame(sept)
oct_df = pd.DataFrame(october)

# jgear = june_df.groupby('Column Height')
# j_top = jgear.get_group('T')
# j_bottom = jgear.get_group('B')

# jugear = july_df.groupby('Column Height')
# ju_top = jugear.get_group('T')
# ju_bottom = jugear.get_group('B')

# agear = august_df.groupby('Column Height')
# a_top = agear.get_group('T')
# a_bottom = agear.get_group('B')

# sgear = sept_df.groupby('Column Height')
# s_top = sgear.get_group('T')
# s_bottom = sgear.get_group('B')

# ogear = oct_df.groupby('Column Height')
# o_top = ogear.get_group('T')
# o_bottom = ogear.get_group('B')

jgear = june_df.groupby('Farmed or Wild')
j_farm = jgear.get_group('F')
j_wild = jgear.get_group('W')

jugear = july_df.groupby('Farmed or Wild')
ju_farm = jugear.get_group('F')
ju_wild = jugear.get_group('W')

agear = august_df.groupby('Farmed or Wild')
a_farm = agear.get_group('F')
a_wild = agear.get_group('W')

sgear = sept_df.groupby('Farmed or Wild')
s_farm = sgear.get_group('F')
s_wild = sgear.get_group('W')

ogear = oct_df.groupby('Farmed or Wild')
o_farm = ogear.get_group('F')
o_wild = ogear.get_group('W')

'''
Box plots for d13C by month separated by gear type
'''
# fig = plt.figure(constrained_layout=True, figsize=(10, 6))
# subfigs = fig.subfigures(1, 5, wspace=0.01)

# subfigs[0].suptitle('June')
# subfigs[0].add_subplot(111).boxplot([j_cage['d13C'], j_net['d13C'], j_wild['d13C']], labels=['Cage', 'Net', 'Wild'])
# subfigs[0].axes[0].set_ylim(-18.5,-16.5)
# subfigs[0].axes[0].set_ylabel('d13C (‰)')

# subfigs[1].suptitle('July')
# subfigs[1].add_subplot(111).boxplot([ju_cage['d13C'], ju_net['d13C'], ju_wild['d13C']], labels=['Cage', 'Net', 'Wild'])
# subfigs[1].axes[0].set_ylim(-20,-16.5)

# subfigs[2].suptitle('August')
# subfigs[2].add_subplot(111).boxplot([a_cage['d13C'], a_net['d13C'], a_wild['d13C']], labels=['Cage', 'Net', 'Wild'])
# subfigs[2].axes[0].set_ylim(-18.5,-16.5)

# subfigs[3].suptitle('September')
# subfigs[3].add_subplot(111).boxplot([s_cage['d13C'], s_net['d13C'], s_wild['d13C']], labels=['Cage', 'Net', 'Wild'])
# subfigs[3].axes[0].set_ylim(-18.5,-16.5)

# subfigs[4].suptitle('October')
# subfigs[4].add_subplot(111).boxplot([o_cage['d13C'], o_net['d13C'], o_wild['d13C']], labels=['Cage', 'Net', 'Wild'])
# subfigs[4].axes[0].set_ylim(-18.5,-16.5)

# plt.show()

# fig = plt.figure(constrained_layout=True, figsize=(10, 6))

'''
Box plots for d13C by gear type separated by month
'''

# fig = plt.figure(constrained_layout=True, figsize=(6, 6))
# subfigs = fig.subfigures(3, 1, wspace=0.01)

# subfigs[1].suptitle('Cages')
# subfigs[1].add_subplot(111).boxplot([j_cage['d13C'], ju_cage['d13C'], a_cage['d13C'], s_cage['d13C'], o_cage['d13C']], labels=['June', 'July', 'August', 'September', 'October'])
# subfigs[1].axes[0].set_ylim(-21,-16)
# subfigs[1].axes[0].set_ylabel('d13C (‰)')

# subfigs[0].suptitle('Nets')
# subfigs[0].add_subplot(111).boxplot([j_net['d13C'], ju_net['d13C'], a_net['d13C'], s_net['d13C'], o_net['d13C']], labels=['June', 'July', 'August', 'September', 'October'])
# subfigs[0].axes[0].set_ylim(-21,-16)
# subfigs[0].axes[0].set_ylabel('d13C (‰)')

# subfigs[2].suptitle('Wild')
# subfigs[2].add_subplot(111).boxplot([j_wild['d13C'], ju_wild['d13C'], a_wild['d13C'], s_wild['d13C'], o_wild['d13C']], labels=['June', 'July', 'August', 'September', 'October'])
# subfigs[2].axes[0].set_ylim(-21,-16)
# subfigs[2].axes[0].set_ylabel('d13C (‰)')

#plt.show()

# fig = plt.figure(constrained_layout=True, figsize=(6, 6))
# subfigs = fig.subfigures(2, 1, wspace=0.01)

# subfigs[0].suptitle('Farmed')
# subfigs[0].add_subplot(111).boxplot([j_farm['d13C'], ju_farm['d13C'], a_farm['d13C'], s_farm['d13C'], o_farm['d13C']], labels=['June', 'July', 'August', 'September', 'October'])
# subfigs[0].axes[0].set_ylim(-21,-16)
# subfigs[0].axes[0].set_ylabel('d13C (‰)')

# subfigs[1].suptitle('Wild')
# subfigs[1].add_subplot(111).boxplot([j_wild['d13C'], ju_wild['d13C'], a_wild['d13C'], s_wild['d13C'], o_wild['d13C']], labels=['June', 'July', 'August', 'September', 'October'])
# subfigs[1].axes[0].set_ylim(-21,-16)
# subfigs[1].axes[0].set_ylabel('d13C (‰)')


# plt.show()

'''
Box plots for d15N by gear type separated by month
'''
# fig = plt.figure(constrained_layout=True, figsize=(6, 6))
# subfigs = fig.subfigures(3, 1, wspace=0.01)

# subfigs[1].suptitle('Cages')
# subfigs[1].add_subplot(111).boxplot([j_cage['d15N'], ju_cage['d15N'], a_cage['d15N'], s_cage['d15N'], o_cage['d15N']], labels=['June', 'July', 'August', 'September', 'October'])
# subfigs[1].axes[0].set_ylim(3,12)
# subfigs[1].axes[0].set_ylabel('d15N (‰)')

# subfigs[0].suptitle('Nets')
# subfigs[0].add_subplot(111).boxplot([j_net['d15N'], ju_net['d15N'], a_net['d15N'], s_net['d15N'], o_net['d15N']], labels=['June', 'July', 'August', 'September', 'October'])
# subfigs[0].axes[0].set_ylim(3,12)
# subfigs[0].axes[0].set_ylabel('d15N (‰)')

# subfigs[2].suptitle('Wild')
# subfigs[2].add_subplot(111).boxplot([j_wild['d15N'], ju_wild['d15N'], a_wild['d15N'], s_wild['d15N'], o_wild['d15N']], labels=['June', 'July', 'August', 'September', 'October'])
# subfigs[2].axes[0].set_ylim(3,12)
# subfigs[2].axes[0].set_ylabel('d15N (‰)')

#plt.show()

# fig = plt.figure(constrained_layout=True, figsize=(6, 6))
# subfigs = fig.subfigures(2, 1, wspace=0.01)

# subfigs[0].suptitle('Farmed')
# subfigs[0].add_subplot(111).boxplot([j_farm['d15N'], ju_farm['d15N'], a_farm['d15N'], s_farm['d15N'], o_farm['d15N']], labels=['June', 'July', 'August', 'September', 'October'])
# subfigs[0].axes[0].set_ylim(3,12)
# subfigs[0].axes[0].set_ylabel('d15N (‰)')

# subfigs[1].suptitle('Wild')
# subfigs[1].add_subplot(111).boxplot([j_wild['d15N'], ju_wild['d15N'], a_wild['d15N'], s_wild['d15N'], o_wild['d15N']], labels=['June', 'July', 'August', 'September', 'October'])
# subfigs[1].axes[0].set_ylim(3,12)
# subfigs[1].axes[0].set_ylabel('d15N (‰)')

# plt.show()

'''
This shows a boxplot of C/N (molar) by gear type separated by month
'''
# fig = plt.figure(constrained_layout=True, figsize=(6, 6))
# subfigs = fig.subfigures(3, 1, wspace=0.01)

# subfigs[1].suptitle('Cages')
# subfigs[1].add_subplot(111).boxplot([j_cage['C/N (Molar)'], ju_cage['C/N (Molar)'], a_cage['C/N (Molar)'], s_cage['C/N (Molar)'], o_cage['C/N (Molar)']], labels=['June', 'July', 'August', 'September', 'October'])
# subfigs[1].axes[0].set_ylim(3,6)
# subfigs[1].axes[0].set_ylabel('C/N (Molar)')

# subfigs[0].suptitle('Nets')
# subfigs[0].add_subplot(111).boxplot([j_net['C/N (Molar)'], ju_net['C/N (Molar)'], a_net['C/N (Molar)'], s_net['C/N (Molar)'], o_net['C/N (Molar)']], labels=['June', 'July', 'August', 'September', 'October'])
# subfigs[0].axes[0].set_ylim(3,6)
# subfigs[0].axes[0].set_ylabel('C/N (Molar)')

# subfigs[2].suptitle('Wild')
# subfigs[2].add_subplot(111).boxplot([j_wild['C/N (Molar)'], ju_wild['C/N (Molar)'], a_wild['C/N (Molar)'], s_wild['C/N (Molar)'], o_wild['C/N (Molar)']], labels=['June', 'July', 'August', 'September', 'October'])
# subfigs[2].axes[0].set_ylim(3,6)
# subfigs[2].axes[0].set_ylabel('C/N (Molar)')

#plt.show()

# fig = plt.figure(constrained_layout=True, figsize=(6, 6))
# subfigs = fig.subfigures(2, 1, wspace=0.01)

# subfigs[0].suptitle('Farmed')
# subfigs[0].add_subplot(111).boxplot([j_farm['C/N (Molar)'], ju_farm['C/N (Molar)'], a_farm['C/N (Molar)'], s_farm['C/N (Molar)'], o_farm['C/N (Molar)']], labels=['June', 'July', 'August', 'September', 'October'])
# subfigs[0].axes[0].set_ylim(3,6)
# subfigs[0].axes[0].set_ylabel('C/N (Molar)')

# subfigs[1].suptitle('Wild')
# subfigs[1].add_subplot(111).boxplot([j_wild['C/N (Molar)'], ju_wild['C/N (Molar)'], a_wild['C/N (Molar)'], s_wild['C/N (Molar)'], o_wild['C/N (Molar)']], labels=['June', 'July', 'August', 'September', 'October'])
# subfigs[1].axes[0].set_ylim(3,6)
# subfigs[1].axes[0].set_ylabel('C/N (Molar)')

# plt.show()


'''
This shows a boxplot of %N by gear type separated by month
'''

# fig = plt.figure(constrained_layout=True, figsize=(6, 6))
# subfigs = fig.subfigures(3, 1, wspace=0.01)

# subfigs[1].suptitle('Cages')
# subfigs[1].add_subplot(111).boxplot([j_cage['% N'], ju_cage['% N'], a_cage['% N'], s_cage['% N'], o_cage['% N']], labels=['June', 'July', 'August', 'September', 'October'])
# subfigs[1].axes[0].set_ylim(7,14)
# subfigs[1].axes[0].set_ylabel('% N')

# subfigs[0].suptitle('Nets')
# subfigs[0].add_subplot(111).boxplot([j_net['% N'], ju_net['% N'], a_net['% N'], s_net['% N'], o_net['% N']], labels=['June', 'July', 'August', 'September', 'October'])
# subfigs[0].axes[0].set_ylim(7,14)
# subfigs[0].axes[0].set_ylabel('% N')

# subfigs[2].suptitle('Wild')
# subfigs[2].add_subplot(111).boxplot([j_wild['% N'], ju_wild['% N'], a_wild['% N'], s_wild['% N'], o_wild['% N']], labels=['June', 'July', 'August', 'September', 'October'])
# subfigs[2].axes[0].set_ylim(7,14)
# subfigs[2].axes[0].set_ylabel('% N')

# plt.show()

# fig = plt.figure(constrained_layout=True, figsize=(6, 6))
# subfigs = fig.subfigures(2, 1, wspace=0.01)

# subfigs[0].suptitle('Farmed')
# subfigs[0].add_subplot(111).boxplot([j_farm['% N'], ju_farm['% N'], a_farm['% N'], s_farm['% N'], o_farm['% N']], labels=['June', 'July', 'August', 'September', 'October'])
# subfigs[0].axes[0].set_ylim(7,14)
# subfigs[0].axes[0].set_ylabel('% N')

# subfigs[1].suptitle('Wild')
# subfigs[1].add_subplot(111).boxplot([j_wild['% N'], ju_wild['% N'], a_wild['% N'], s_wild['% N'], o_wild['% N']], labels=['June', 'July', 'August', 'September', 'October'])
# subfigs[1].axes[0].set_ylim(7,14)
# subfigs[1].axes[0].set_ylabel('% N')

# plt.show()

'''
WORKING TEMPLATE FOR BOXPLOTS!
I still need to go through this file and clean it up
'''

box_properties = dict(facecolor = 'white', color='black', linewidth=1)
median_properties = dict(color='black', linewidth=1.5)
whisker_properties = dict(color='black')
cap_properties = dict(color='black', linewidth=1)
flier_properties = dict(marker='o', color = 'black', markerfacecolor='black', markersize=2)

box_properties_2 = dict(facecolor = 'white', color='red', linewidth=1)
median_properties_2 = dict(color='red', linewidth=1.5)
whisker_properties_2 = dict(color='red')
cap_properties_2 = dict(color='red', linewidth=1)
flier_properties_2 = dict(marker='o', color = 'red', markerfacecolor='red', markersize=2)

plt.boxplot([j_farm['C/N (Molar)'], ju_farm['C/N (Molar)'], a_farm['C/N (Molar)'], s_farm['C/N (Molar)'], o_farm['C/N (Molar)']], 
            positions = [1,2,3,4,5], 
            widths = 0.3,
            patch_artist=True, 
            boxprops=box_properties, 
            medianprops=median_properties,
            whiskerprops=whisker_properties,
            capprops=cap_properties,
            flierprops=flier_properties)
plt.boxplot([j_wild['C/N (Molar)'], ju_wild['C/N (Molar)'], a_wild['C/N (Molar)'], s_wild['C/N (Molar)'], o_wild['C/N (Molar)']],
            positions = [1.3,2.3,3.3,4.3,5.3], 
            widths = 0.3,
            patch_artist=True, 
            boxprops=box_properties_2, 
            medianprops=median_properties_2,
            whiskerprops=whisker_properties_2,
            capprops=cap_properties_2,
            flierprops=flier_properties_2)
plt.xticks([1.3,2.3,3.3,4.3,5.3], ['June', 'July', 'August', 'September', 'October'])
plt.ylim(3, 6)
plt.ylabel('C/N (Molar)')
plt.legend(handles = [
        mpatches.Patch(color='black', label='Farm'),
        mpatches.Patch(color='red', label='Wild'),
])
plt.show()
'''
Table of averages and standard deviations for d13C, d15N, and C/N (molar) by gear type my month
'''

fig, ax = plt.subplots(figsize=(8, 6)) 
ax.axis('off')

summary = [
    ['June Farm', round(j_farm['d13C'].mean(),2), round(j_farm['d13C'].std(),2), round(j_farm['d15N'].mean(),2), round(j_farm['d15N'].std(),2), round(j_farm['C/N (Molar)'].mean(),2), round(j_farm['C/N (Molar)'].std(),2)],
    ['June Wild', round(j_wild['d13C'].mean(),2), round(j_wild['d13C'].std(),2), round(j_wild['d15N'].mean(),2), round(j_wild['d15N'].std(),2), round(j_wild['C/N (Molar)'].mean(),2), round(j_wild['C/N (Molar)'].std(),2)],
    ['July Farm', round(ju_farm['d13C'].mean(),2), round(ju_farm['d13C'].std(),2), round(ju_farm['d15N'].mean(),2), round(ju_farm['d15N'].std(),2), round(ju_farm['C/N (Molar)'].mean(),2), round(ju_farm['C/N (Molar)'].std(),2)],
    ['July Wild', round(ju_wild['d13C'].mean(),2), round(ju_wild['d13C'].std(),2), round(ju_wild['d15N'].mean(),2), round(ju_wild['d15N'].std(),2), round(ju_wild['C/N (Molar)'].mean(),2), round(ju_wild['C/N (Molar)'].std(),2)],
    ['August Farm', round(a_farm['d13C'].mean(),2), round(a_farm['d13C'].std(),2), round(a_farm['d15N'].mean(),2), round(a_farm['d15N'].std(),2), round(a_farm['C/N (Molar)'].mean(),2), round(a_farm['C/N (Molar)'].std(),2)],
    ['August Wild', round(a_wild['d13C'].mean(),2), round(a_wild['d13C'].std(),2), round(a_wild['d15N'].mean(),2), round(a_wild['d15N'].std(),2), round(a_wild['C/N (Molar)'].mean(),2), round(a_wild['C/N (Molar)'].std(),2)],
    ['September Farm', round(s_farm['d13C'].mean(),2), round(s_farm['d13C'].std(),2), round(s_farm['d15N'].mean(),2), round(s_farm['d15N'].std(),2), round(s_farm['C/N (Molar)'].mean(),2), round(s_farm['C/N (Molar)'].std(),2)],
    ['September Wild', round(s_wild['d13C'].mean(),2), round(s_wild['d13C'].std(),2), round(s_wild['d15N'].mean(),2), round(s_wild['d15N'].std(),2), round(s_wild['C/N (Molar)'].mean(),2), round(s_wild['C/N (Molar)'].std(),2)],
    ['October Farm', round(o_farm['d13C'].mean(),2), round(o_farm['d13C'].std(),2), round(o_farm['d15N'].mean(),2), round(o_farm['d15N'].std(),2), round(o_farm['C/N (Molar)'].mean(),2), round(o_farm['C/N (Molar)'].std(),2)],
    ['October Wild', round(o_wild['d13C'].mean(),2), round(o_wild['d13C'].std(),2), round(o_wild['d15N'].mean(),2), round(o_wild['d15N'].std(),2), round(o_wild['C/N (Molar)'].mean(),2), round(o_wild['C/N (Molar)'].std(),2)]
]

table = ax.table(cellText=summary,
                 colLabels= ['Gear', 'd13C Mean', 'd13C SD', 'd15N Mean', 'd15N SD', 'C/N Mean', 'C/N SD'],
                 cellLoc = 'center',
                 loc='center')
plt.show()

# filters_summary = [
#     ['June Net Filter', round(j_net_filt['d13C'].mean(),2), round(j_net_filt['d15N'].mean(),2), round(j_net_filt['C/N (Molar)'].mean(),2)],
#     ['June Cage Filter', round(j_cage_filt['d13C'].mean(),2), round(j_cage_filt['d15N'].mean(),2),  round(j_cage_filt['C/N (Molar)'].mean(),2)],
#     ['June Wild Filter', round(j_wild_filt['d13C'].mean(),2), round(j_wild_filt['d15N'].mean(),2),  round(j_wild_filt['C/N (Molar)'].mean(),2)],
#     ['July Net Filter', round(ju_net_filt['d13C'].mean(),2), round(ju_net_filt['d15N'].mean(),2),  round(ju_net_filt['C/N (Molar)'].mean(),2)],
#     ['July Cage Filter', round(ju_cage_filt['d13C'].mean(),2), round(ju_cage_filt['d15N'].mean(),2), round(ju_cage_filt['C/N (Molar)'].mean(),2)],
#     ['July Wild Filter', round(ju_wild_filt['d13C'].mean(),2),  round(ju_wild_filt['d15N'].mean(),2),  round(ju_wild_filt['C/N (Molar)'].mean(),2)],
#     ['August Net Filter', round(a_net_filt['d13C'].mean(),2),  round(a_net_filt['d15N'].mean(),2), round(a_net_filt['C/N (Molar)'].mean(),2)],
#     ['August Cage Filter', round(a_cage_filt['d13C'].mean(),2),  round(a_cage_filt['d15N'].mean(),2),  round(a_cage_filt['C/N (Molar)'].mean(),2)],
#     ['August Wild Filter', round(a_wild_filt['d13C'].mean(),2),  round(a_wild_filt['d15N'].mean(),2),  round(a_wild_filt['C/N (Molar)'].mean(),2)],
#     ['September Net Filter', round(s_net_filt['d13C'].mean(),2),  round(s_net_filt['d15N'].mean(),2),  round(s_net_filt['C/N (Molar)'].mean(),2)],
#     ['September Cage Filter', round(s_cage_filt['d13C'].mean(),2),  round(s_cage_filt['d15N'].mean(),2),  round(s_cage_filt['C/N (Molar)'].mean(),2)],
#     ['September Wild Filter', round(s_wild_filt['d13C'].mean(),2), round(s_wild_filt['d15N'].mean(),2), round(s_wild_filt['C/N (Molar)'].mean(),2)],
#     ['October Net Filter', round(o_net_filt['d13C'].mean(),2),  round(o_net_filt['d15N'].mean(),2),  round(o_net_filt['C/N (Molar)'].mean(),2)],
#     ['October Cage Filter', round(o_cage_filt['d13C'].mean(),2),  round(o_cage_filt['d15N'].mean(),2),  round(o_cage_filt['C/N (Molar)'].mean(),2)],
#     ['October Wild Filter', round(o_wild_filt['d13C'].mean(),2),  round(o_wild_filt['d15N'].mean(),2),  round(o_wild_filt['C/N (Molar)'].mean(),2)],
# ]

# table = ax.table(cellText=filters_summary,
#                  colLabels= ['Gear', 'd13C Mean', 'd15N Mean', 'C/N (Molar) Mean'],
#                  cellLoc = 'center',
#                  loc='center')
#plt.show()

'''
Table of the difference between the average d13C of samples by gear type and month and the average d13C of the corresponding filter samples
Diet focused question, what is the difference between scallop and filter d15N and d13C values - does this indicate one trophic level increase?

But, need to fix the code for this table if this is something to focus on
'''


# filters_summary = [
#     ['June Net Diff', round(j_net['d13C'].mean(),2) - round(j_net_filt['d13C'].mean(),2), round(j_net['d15N'].mean(),2) - round(j_net_filt['d15N'].mean(),2), round(j_net['C/N (Molar)'].mean(),2) - round(j_net_filt['C/N (Molar)'].mean(),2)],
#     ['June Cage Diff', round(j_cage['d13C'].mean(),2) - round(j_cage_filt['d13C'].mean(),2), round(j_cage['d15N'].mean(),2) - round(j_cage_filt['d15N'].mean(),2), round(j_cage['C/N (Molar)'].mean(),2) - round(j_cage_filt['C/N (Molar)'].mean(),2)],
#     ['June Wild Diff', round(j_wild['d13C'].mean(),2) - round(j_wild_filt['d13C'].mean(),2), round(j_wild['d15N'].mean(),2) - round(j_wild_filt['d15N'].mean(),2), round(j_wild['C/N (Molar)'].mean(),2) - round(j_wild_filt['C/N (Molar)'].mean(),2)],
#     ['July Net Diff', round(ju_net['d13C'].mean(),2) - round(ju_net_filt['d13C'].mean(),2), round(ju_net['d15N'].mean(),2) - round(ju_net_filt['d15N'].mean(),2), round(ju_net['C/N (Molar)'].mean(),2) - round(ju_net_filt['C/N (Molar)'].mean(),2)],
#     ['July Cage Diff', round(ju_cage['d13C'].mean(),2) - round(ju_cage_filt['d13C'].mean(),2), round(ju_cage['d15N'].mean(),2) - round(ju_cage_filt['d15N'].mean(),2), round(ju_cage['C/N (Molar)'].mean(),2) - round(ju_cage_filt['C/N (Molar)'].mean(),2)],
#     ['July Wild Diff', round(ju_wild['d13C'].mean(),2) - round(ju_wild_filt['d13C'].mean(),2), round(ju_wild['d15N'].mean(),2) - round(ju_wild_filt['d15N'].mean(),2), round(ju_wild['C/N (Molar)'].mean(),2) - round(ju_wild_filt['C/N (Molar)'].mean(),2)],
#     ['August Net Diff', round(a_net['d13C'].mean(),2) - round(a_net_filt['d13C'].mean(),2), round(a_net['d15N'].mean(),2) - round(a_net_filt['d15N'].mean(),2), round(a_net['C/N (Molar)'].mean(),2) - round(a_net_filt['C/N (Molar)'].mean(),2)],
#     ['August Cage Diff', round(a_cage['d13C'].mean(),2) - round(a_cage_filt['d13C'].mean(),2), round(a_cage['d15N'].mean(),2) - round(a_cage_filt['d15N'].mean(),2), round(a_cage['C/N (Molar)'].mean(),2) - round(a_cage_filt['C/N (Molar)'].mean(),2)],
#     ['August Wild Diff', round(a_wild['d13C'].mean(),2) - round(a_wild_filt['d13C'].mean(),2), round(a_wild['d15N'].mean(),2) - round(a_wild_filt['d15N'].mean(),2), round(a_wild['C/N (Molar)'].mean(),2) - round(a_wild_filt['C/N (Molar)'].mean(),2)],
#     ['Sept Net Diff', round(s_net['d13C'].mean(),2) - round(s_net_filt['d13C'].mean(),2), round(s_net['d15N'].mean(),2) - round(s_net_filt['d15N'].mean(),2), round(s_net['C/N (Molar)'].mean(),2) - round(s_net_filt['C/N (Molar)'].mean(),2)],
#     ['Sept Cage Diff', round(s_cage['d13C'].mean(),2) - round(s_cage_filt['d13C'].mean(),2), round(s_cage['d15N'].mean(),2) - round(s_cage_filt['d15N'].mean(),2), round(s_cage['C/N (Molar)'].mean(),2) - round(s_cage_filt['C/N (Molar)'].mean(),2)],
#     ['Sept Wild Diff', round(s_wild['d13C'].mean(),2) - round(s_wild_filt['d13C'].mean(),2), round(s_wild['d15N'].mean(),2) - round(s_wild_filt['d15N'].mean(),2), round(s_wild['C/N (Molar)'].mean(),2) - round(s_wild_filt['C/N (Molar)'].mean(),2)],
#     ['Oct Net Diff', round(o_net['d13C'].mean(),2) - round(o_net_filt['d13C'].mean(),2), round(o_net['d15N'].mean(),2) - round(o_net_filt['d15N'].mean(),2), round(o_net['C/N (Molar)'].mean(),2) - round(o_net_filt['C/N (Molar)'].mean(),2)],
#     ['Oct Cage Diff', round(o_cage['d13C'].mean(),2) - round(o_cage_filt['d13C'].mean(),2), round(o_cage['d15N'].mean(),2) - round(o_cage_filt['d15N'].mean(),2), round(o_cage['C/N (Molar)'].mean(),2) - round(o_cage_filt['C/N (Molar)'].mean(),2)],
#     ['Oct Wild Diff', round(o_wild['d13C'].mean(),2) - round(o_wild_filt['d13C'].mean(),2), round(o_wild['d15N'].mean(),2) - round(o_wild_filt['d15N'].mean(),2), round(o_wild['C/N (Molar)'].mean(),2) - round(o_wild_filt['C/N (Molar)'].mean(),2)],
# ]

# table = ax.table(cellText=filters_summary,
#                  colLabels= ['Gear', 'd13C Mean', 'd15N Mean', 'C/N (Molar) Mean'],
#                  cellLoc = 'center',
#                  loc='center')
# plt.show()

# summary = [
#     ['June Net', round(j_net['% N'].mean(),2), round(j_net['% N'].std(),2)],
#     ['June Cage', round(j_cage['% N'].mean(),2), round(j_cage['% N'].std(),2)], 
#     ['June Wild', round(j_wild['% N'].mean(),2), round(j_wild['% N'].std(),2)],
#     ['July Net', round(ju_net['% N'].mean(),2), round(ju_net['% N'].std(),2)],
#     ['July Cage', round(ju_cage['% N'].mean(),2), round(ju_cage['% N'].std(),2)],
#     ['July Wild', round(ju_wild['% N'].mean(),2), round(ju_wild['% N'].std(),2)],
#     ['August Net', round(a_net['% N'].mean(),2), round(a_net['% N'].std(),2)],
#     ['August Cage', round(a_cage['% N'].mean(),2), round(a_cage['% N'].std(),2)],
#     ['August Wild', round(a_wild['% N'].mean(),2), round(a_wild['% N'].std(),2)],
#     ['September Net', round(s_net['% N'].mean(),2), round(s_net['% N'].std(),2)],
#     ['September Cage', round(s_cage['% N'].mean(),2), round(s_cage['% N'].std(),2)],
#     ['October Net', round(o_net['% N'].mean(),2), round(o_net['% N'].std(),2)],
#     ['October Cage', round(o_cage['% N'].mean(),2), round(o_cage['% N'].std(),2)],
#     ['October Wild', round(o_wild['% N'].mean(),2), round(o_wild['% N'].std(),2)]]

# table = ax.table(cellText=summary,
#                  colLabels= ['Gear', '%N', '%N SD'],
#                  cellLoc = 'center',
#                  loc='center')
#plt.savefig(figures / "table.png")

# net_sd = [
#     ['June Net', round(j_net['d13C'].std(),2), round(j_net['d15N'].std(),2), round(j_net['C/N (Molar)'].std(),2), round(j_net['% N'].std(),2),],
#     ['July Net', round(ju_net['d13C'].std(),2), round(ju_net['d15N'].std(),2), round(ju_net['C/N (Molar)'].std(),2), round(ju_net['% N'].std(),2),],
#     ['August Net', round(a_net['d13C'].std(),2), round(a_net['d15N'].std(),2), round(a_net['C/N (Molar)'].std(),2), round(a_net['% N'].std(),2)],
#     ['September Net', round(s_net['d13C'].std(),2), round(s_net['d15N'].std(),2), round(s_net['C/N (Molar)'].std(),2), round(s_net['% N'].std(),2)],
#     ['October Net', round(o_net['d13C'].std(),2), round(o_net['d15N'].std(),2), round(o_net['C/N (Molar)'].std(),2), round(o_net['% N'].std(),2)],
# ]

# table = ax.table(cellText=net_sd,
#                  colLabels= ['Month', 'd13C SD','d15N SD', 'C/N (Molar) SD', '%N SD'],
#                  cellLoc = 'center',
#                  loc='center')
# ax.set_title('Net')
# plt.show()

# cage_sd = [
#     ['June', round(j_cage['d13C'].std(),2), round(j_cage['d15N'].std(),2), round(j_cage['C/N (Molar)'].std(),2), round(j_cage['% N'].std(),2),],
#     ['July', round(ju_cage['d13C'].std(),2), round(ju_cage['d15N'].std(),2), round(ju_cage['C/N (Molar)'].std(),2), round(ju_cage['% N'].std(),2),],
#     ['August', round(a_cage['d13C'].std(),2), round(a_cage['d15N'].std(),2), round(a_cage['C/N (Molar)'].std(),2), round(a_cage['% N'].std(),2)],
#     ['September', round(s_cage['d13C'].std(),2), round(s_cage['d15N'].std(),2), round(s_cage['C/N (Molar)'].std(),2), round(s_cage['% N'].std(),2)],
#     ['October', round(o_cage['d13C'].std(),2), round(o_cage['d15N'].std(),2), round(o_cage['C/N (Molar)'].std(),2), round(o_cage['% N'].std(),2)],
# ]
# table = ax.table(cellText=cage_sd,
#                  colLabels= ['Month', 'd13C SD','d15N SD', 'C/N (Molar) SD', '%N SD'],
#                  cellLoc = 'center',
#                  loc='center')
# ax.set_title('Cage')
# plt.show()

# wild_sd = [
#     ['June', round(j_wild['d13C'].std(),2), round(j_wild['d15N'].std(),2), round(j_wild['C/N (Molar)'].std(),2), round(j_wild['% N'].std(),2),],
#     ['July', round(ju_wild['d13C'].std(),2), round(ju_wild['d15N'].std(),2), round(ju_wild['C/N (Molar)'].std(),2), round(ju_wild['% N'].std(),2),],
#     ['August', round(a_wild['d13C'].std(),2), round(a_wild['d15N'].std(),2), round(a_wild['C/N (Molar)'].std(),2), round(a_wild['% N'].std(),2)],
#     ['September', round(s_wild['d13C'].std(),2), round(s_wild['d15N'].std(),2), round(s_wild['C/N (Molar)'].std(),2), round(s_wild['% N'].std(),2)],
#     ['October', round(o_wild['d13C'].std(),2), round(o_wild['d15N'].std(),2), round(o_wild['C/N (Molar)'].std(),2), round(o_wild['% N'].std(),2)],
# ]

# table = ax.table(cellText= wild_sd,
#                  colLabels= ['Month', 'd13C SD','d15N SD', 'C/N (Molar) SD', '%N SD'],
#                  cellLoc = 'center',
#                  loc='center')
# ax.set_title('Wild')
# plt.show()

'''
Bar chart of standard deviations for d13C, d15N, C/N (molar), and %N by month for gear types
'''

# barWidth = 0.2
# fig = plt.subplots(figsize =(20, 8))

# wild_d13C_sds = [round(j_wild['d13C'].std(),2), round(ju_wild['d13C'].std(),2), round(a_wild['d13C'].std(),2), round(s_wild['d13C'].std(),2), round(o_wild['d13C'].std(),2)]
# wild_d15N_sds = [round(j_wild['d15N'].std(),2), round(ju_wild['d15N'].std(),2), round(a_wild['d15N'].std(),2), round(s_wild['d15N'].std(),2), round(o_wild['d15N'].std(),2)]
# wild_CN_sds = [round(j_wild['C/N (Molar)'].std(),2), round(ju_wild['C/N (Molar)'].std(),2), round(a_wild['C/N (Molar)'].std(),2), round(s_wild['C/N (Molar)'].std(),2), round(o_wild['C/N (Molar)'].std(),2)]
# wild_N_sds = [round(j_wild['% N'].std(),2), round(ju_wild['% N'].std(),2), round(a_wild['% N'].std(),2), round(s_wild['% N'].std(),2), round(o_wild['% N'].std(),2)]    

# net_d13C_sds = [round(j_net['d13C'].std(),2), round(ju_net['d13C'].std(),2), round(a_net['d13C'].std(),2), round(s_net['d13C'].std(),2), round(o_net['d13C'].std(),2)]
# net_d15N_sds = [round(j_net['d15N'].std(),2), round(ju_net['d15N'].std(),2), round(a_net['d15N'].std(),2), round(s_net['d15N'].std(),2), round(o_net['d15N'].std(),2)]
# net_CN_sds = [round(j_net['C/N (Molar)'].std(),2), round(ju_net['C/N (Molar)'].std(),2), round(a_net['C/N (Molar)'].std(),2), round(s_net['C/N (Molar)'].std(),2), round(o_net['C/N (Molar)'].std(),2)]
# net_N_sds = [round(j_net['% N'].std(),2), round(ju_net['% N'].std(),2), round(a_net['% N'].std(),2), round(s_net['% N'].std(),2), round(o_net['% N'].std(),2)]  

# cage_d13C_sds = [round(j_cage['d13C'].std(),2), round(ju_cage['d13C'].std(),2), round(a_cage['d13C'].std(),2), round(s_cage['d13C'].std(),2), round(o_cage['d13C'].std(),2)]
# cage_d15N_sds = [round(j_cage['d15N'].std(),2), round(ju_cage['d15N'].std(),2), round(a_cage['d15N'].std(),2), round(s_cage['d15N'].std(),2), round(o_cage['d15N'].std(),2)]
# cage_CN_sds = [round(j_cage['C/N (Molar)'].std(),2), round(ju_cage['C/N (Molar)'].std(),2), round(a_cage['C/N (Molar)'].std(),2), round(s_cage['C/N (Molar)'].std(),2), round(o_cage['C/N (Molar)'].std(),2)]
# cage_N_sds = [round(j_cage['% N'].std(),2), round(ju_cage['% N'].std(),2), round(a_cage['% N'].std(),2), round(s_cage['% N'].std(),2), round(o_cage['% N'].std(),2)]    


# br1 = np.arange(len(wild_d13C_sds)) 
# br2 = [x + barWidth for x in br1] 
# br3 = [x + barWidth for x in br2] 
# br4 = [x + barWidth for x in br3] 
# br5 = [x + barWidth for x in br4]

# '''
# Change the next four lines to plot the gear type you want
# '''

# plt.bar(br1, cage_d13C_sds, color='b', width=0.2, label = 'd13C SD')
# plt.bar(br2, cage_d15N_sds, color='r', width=0.2, label = 'd15N SD')
# plt.bar(br3, cage_CN_sds, color='g', width=0.2, label = 'C/N (Molar) SD')
# plt.bar(br4, cage_N_sds, color='y', width=0.2, label = '%N SD')

# catwidth = 0.3

# plt.ylim(0,2.5)
# plt.xticks([r + catwidth for r in range(len(wild_d13C_sds))], 
#         ['June', 'July', 'August', 'Sept', 'Oct'])

# plt.legend()
# plt.show()

'''
Bar chart showing spread of data for farmed vs wild
'''
# barWidth = 0.2
# fig = plt.subplots(figsize =(20, 8))

# wild_d13C_sds = [round(j_wild['d13C'].std(),2), round(ju_wild['d13C'].std(),2), round(a_wild['d13C'].std(),2), round(s_wild['d13C'].std(),2), round(o_wild['d13C'].std(),2)]
# wild_d15N_sds = [round(j_wild['d15N'].std(),2), round(ju_wild['d15N'].std(),2), round(a_wild['d15N'].std(),2), round(s_wild['d15N'].std(),2), round(o_wild['d15N'].std(),2)]
# wild_CN_sds = [round(j_wild['C/N (Molar)'].std(),2), round(ju_wild['C/N (Molar)'].std(),2), round(a_wild['C/N (Molar)'].std(),2), round(s_wild['C/N (Molar)'].std(),2), round(o_wild['C/N (Molar)'].std(),2)]
# wild_N_sds = [round(j_wild['% N'].std(),2), round(ju_wild['% N'].std(),2), round(a_wild['% N'].std(),2), round(s_wild['% N'].std(),2), round(o_wild['% N'].std(),2)]    

# farm_d13C_sds = [round(j_farm['d13C'].std(),2), round(ju_farm['d13C'].std(),2), round(a_farm['d13C'].std(),2), round(s_farm['d13C'].std(),2), round(o_farm['d13C'].std(),2)]
# farm_d15N_sds = [round(j_farm['d15N'].std(),2), round(ju_farm['d15N'].std(),2), round(a_farm['d15N'].std(),2), round(s_farm['d15N'].std(),2), round(o_farm['d15N'].std(),2)]
# farm_CN_sds = [round(j_farm['C/N (Molar)'].std(),2), round(ju_farm['C/N (Molar)'].std(),2), round(a_farm['C/N (Molar)'].std(),2), round(s_farm['C/N (Molar)'].std(),2), round(o_farm['C/N (Molar)'].std(),2)]
# farm_N_sds = [round(j_farm['% N'].std(),2), round(ju_farm['% N'].std(),2), round(a_farm['% N'].std(),2), round(s_farm['% N'].std(),2), round(o_farm['% N'].std(),2)]  



# br1 = np.arange(len(wild_d13C_sds)) 
# br2 = [x + barWidth for x in br1] 
# br3 = [x + barWidth for x in br2] 
# br4 = [x + barWidth for x in br3] 
# br5 = [x + barWidth for x in br4]

# '''
# Change the next four lines to plot the gear type you want
# '''

# plt.bar(br1, farm_d13C_sds, color='b', width=0.2, label = 'd13C SD')
# plt.bar(br2, farm_d15N_sds, color='r', width=0.2, label = 'd15N SD')
# plt.bar(br3, farm_CN_sds, color='g', width=0.2, label = 'C/N (Molar) SD')
# plt.bar(br4, farm_N_sds, color='y', width=0.2, label = '%N SD')

# catwidth = 0.3

# plt.ylim(0,2.5)
# plt.xticks([r + catwidth for r in range(len(wild_d13C_sds))], 
#         ['June', 'July', 'August', 'Sept', 'Oct'])

# plt.legend()
# plt.show()


