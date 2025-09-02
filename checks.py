import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


file_path = '/Users/adelejordan/Downloads/Hurricane/Isotopes/2023IsotopeDataReport-cleanedinexcel.csv'
data = pd.read_csv(file_path, header=0, usecols = [
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

data.dropna(subset=['Gear Type'], inplace=True)
data.dropna(subset=['Sex'], inplace=True)

types = data.groupby('Gear Type')
cages = types.get_group('C')
nets = types.get_group('N')
wild = types.get_group('W')

# plt.subplots(1, 3, figsize = (4,3), sharex=False, sharey=True)

# plt.subplot(1,3,1)
# plt.scatter(cages['%C'], cages['d13C'])

# plt.subplot(1,3,2)
# plt.scatter(nets['%C'], nets['d13C'])

# plt.subplot(1,3,3)
# plt.scatter(wild['%C'], wild['d13C'])

# plt.show()

colormap = {
    'C': 'b', 
    'N': 'g', 
    'W': 'r',
}

'''
A positive relationship would indicate that there is influence of inorganic carbon (not the case)
'''
# for x in data['Gear Type']:
#     if x in colormap:
#         plt.scatter(data[data['Gear Type'] == x]['%C'], 
#                     data[data['Gear Type'] == x]['d13C'], 
#                     c=colormap[x], 
#                     s=10)  # s is the size of the markers


# plt.show()

'''
A negative relationship would indicate that there is influence of lipids (possibly lipid influence in Nets and Wild, but looks uniform between dates processed so likely okay)
'''

datemap = {'9/1/23' : 'g',
'8/31/23' : 'b',
'9/5/23' : 'y', 
'9/6/23' : 'r',
'9/12/23' : 'm',
'11/6/23' : 'c',
'11/27/23' : 'k', 
'11/28/23' : 'w',
'11/29/23' : 'tab:green',
'11/30/23' :'tab:blue'
}

# plt.scatter(data['C/N (Molar)'], data['d13C'])

for x in data['Date Run']:
    if x in datemap:
        plt.scatter(data[data['Date Run'] == x]['C/N (Molar)'], 
                    data[data['Date Run'] == x]['d13C'], 
                    c=datemap[x], 
                    s=10)  # s is the size of the markers
plt.show()

