'''
THIS IS OLDER CODE, DO NOT USE. SEE stats-analysis.py FOR THE MOST RECENT VERSION
'''

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


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
    'C/N (Molar)'])

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

df = data.drop(columns = ['Analysis', 'Sample ID', 'N (umoles)', 'C (umoles)'])
df_muscle = df[df['Tissue Type (Gonad or Muscle)'] == 2]
df_muscle = pd.DataFrame(df_muscle)

# print(df.isna().sum())
# print(df_muscle.columns)

new_df = df_muscle.drop(columns = ['Collection Date', 'Gear Type', 'Sex', 'Tissue Type (Gonad or Muscle)',
       'Number in gear type'])

std_scaler = StandardScaler()
scaled_df = std_scaler.fit_transform(df)
pca = PCA(n_components = None)
components = pca.fit_transform(scaled_df)

comp_df = pd.DataFrame(pca.components_, columns = list(df.columns))

# comp_df.to_csv('/Users/adelejordan/Downloads/Hurricane/Isotopes/pca_components2.csv')
# print(pca.explained_variance_)
# print(pca.explained_variance_ratio_)
print(pca.explained_variance_ratio_.cumsum())
# explained_variances = pca.explained_variance_ratio_
# loadings = pca.components_.T * np.sqrt(explained_variances)

colormap = {
    6 : 'r', 
    7 : 'y', 
    8 : 'g',
    9 : 'b',
    10 : 'm'
}


# for x in df_muscle['Collection Date']:
#     if x in colormap:
#         plt.scatter(components[df_muscle['Collection Date']==x][:, 0], 
#                     components[df_muscle['Collection Date']][:, 1],  
#                     c=colormap[x],
#                     label = colormap[x], 
#                     s=50)  # s is the size of the markers
# plt.legend()
# plt.show()

# for dates in df_muscle['Collection Date']:
#     plt.scatter(components[:,1], components[:,2], c = df_muscle['Gear Type'], cmap='viridis')
#     plt.xlim(-4,4)
#     plt.ylim(-4,4)


# plt.show()









