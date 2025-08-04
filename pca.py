import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


file_path = '/Users/adelejordan/Downloads/Hurricane/Isotopes/2023IsotopeDataReport.csv'
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

typemap = {
    'C' : 1, 
    'N': 2, 
    'W': 3
}
for x in data['Gear Type']:
    if x in typemap:
        data['Gear Type'] = data['Gear Type'].replace(x, typemap[x])
    else:
        data['Gear Type'] = data['Gear Type'].replace(x, 0)

for x in range(len(data['Gear Type'])):
    if data['Gear Type'][x] == 0:
        data.drop(x, inplace=True)

sexmap = {
    'F' : 1, 
    'M': 2
}

for x in data['Sex']:
    if x in sexmap:
        data['Sex'] = data['Sex'].replace(x, sexmap[x])
    else:
        data['Sex'] = data['Sex'].replace(x, 0)

tissuemap = {
    'G' : 1, 
    'M': 2
}

for x in data['Tissue Type (Gonad or Muscle)']:
    if x in tissuemap:
        data['Tissue Type (Gonad or Muscle)'] = data['Tissue Type (Gonad or Muscle)'].replace(x, tissuemap[x])
    else:
        data['Tissue Type (Gonad or Muscle)'] = data['Tissue Type (Gonad or Muscle)'].replace(x, 0)

# for x in range(len(data['Sex'])):
#     if data['Sex'][x] == 0:
#         data.drop(x, inplace=True)

df = data.drop(columns = ['Analysis', 'Sample ID', 'Date Run'])
# print(df.head())



std_scaler = StandardScaler()
scaled_df = std_scaler.fit_transform(df)
pca = PCA(n_components = None)
pca.fit_transform(scaled_df)
# print(pca.components_)
print(pca.explained_variance_ratio_)







