'''
This code shows a scatter plot of d15N vs d13C values based on different gear types including filters.

'''

import pandas as pd
import matplotlib.pyplot as plt

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

gear = data.groupby('Gear Type')
cages = gear.get_group('C')
nets = gear.get_group('N')
wild = gear.get_group('W')
cfilt = gear.get_group('CF')
nfilt = gear.get_group('NF')
wfilt = gear.get_group('WF')

data.dropna(subset=['Gear Type'], inplace=True)

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

for x in data['Gear Type']:
    if x in colormap:
        plt.scatter(data[data['Gear Type'] == x]['d15N'], 
                    data[data['Gear Type'] == x]['d13C'], 
                    c=colormap[x],
                    marker=markermap[x],
                    s=50)  # s is the size of the markers

plt.legend(loc='upper left', title='Gear Type')
plt.show()

