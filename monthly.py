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
    # 9/6/23 samples were contaiminated 

data.dropna(subset=['Gear Type'], inplace=True) #only scallops and filters are being plotted
# data_muscle = data[data['Tissue Type (Gonad or Muscle)'] == 'M'] 
# data_muscle = pd.DataFrame(data_muscle)
# data_gonad = data[data['Tissue Type (Gonad or Muscle)'] == 'G']
# data_gonad = pd.DataFrame(data_gonad)

data_muscle = data.dropna(subset = ['Tissue Type (Gonad or Muscle)'])
data_muscle = data_muscle.drop(data_muscle[data_muscle['Tissue Type (Gonad or Muscle)'] == 'G'].index)


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



monthly = data_muscle.groupby('Collection Date')
june = monthly.get_group(6)
july = monthly.get_group(7)
august = monthly.get_group(8)
sept = monthly.get_group(9)
october = monthly.get_group(10)


plt.subplots(1 , 5, figsize=(6,5), sharex=True, sharey=True)

# for x in june['Gear Type']:
#     if x in colormap:
#         plt.subplot(1, 5, 1)
#         plt.scatter(june[june['Gear Type'] == x]['d13C'], 
#                     june[june['Gear Type'] == x]['d15N'], 
#                     c=colormap[x],
#                     marker=markermap[x],
#                     s=20, 
#                     label = 'June')

# for x in july['Gear Type']:
#     if x in colormap:
#         plt.subplot(1, 5, 2)
#         plt.scatter(july[july['Gear Type'] == x]['d13C'], 
#                     july[july['Gear Type'] == x]['d15N'], 
#                     c=colormap[x], 
#                     marker=markermap[x],
#                     s=20,
#                     label = 'July')

# for x in august['Gear Type']:
#     if x in colormap:
#         plt.subplot(1, 5, 3)
#         plt.scatter(august[august['Gear Type'] == x]['d13C'], 
#                     august[august['Gear Type'] == x]['d15N'], 
#                     c=colormap[x], 
#                     marker=markermap[x],
#                     s=20,
#                     label = 'August')

# for x in sept['Gear Type']:
#     if x in colormap:
#         plt.subplot(1, 5, 4)
#         plt.scatter(sept[sept['Gear Type'] == x]['d13C'], 
#                     sept[sept['Gear Type'] == x]['d15N'], 
#                     c=colormap[x],
#                     marker=markermap[x], 
#                     s=20, 
#                     label = 'September')
        
# for x in october['Gear Type']:
#     if x in colormap:
#         plt.subplot(1, 5, 5)
#         plt.scatter(october[october['Gear Type'] == x]['d13C'], 
#                     october[october['Gear Type'] == x]['d15N'], 
#                     c=colormap[x], 
#                     marker=markermap[x],
#                     s=20, 
#                     label = 'October')


# plt.subplot(1,5,1)
# plt.scatter(june['d13C'], june['d15N'])

# plt.subplot(1,5,2)
# plt.scatter(july['d13C'], july['d15N'])

# plt.subplot(1,5,3)
# plt.scatter(august['d13C'], august['d15N'])

# plt.subplot(1,5,4)
# plt.scatter(sept['d13C'], sept['d15N'])

# plt.subplot(1,5,5)
# plt.scatter(october['d13C'], october['d15N'])
#plt.show()



