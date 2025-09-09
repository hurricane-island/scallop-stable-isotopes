import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

figures = Path(__file__).parent / "figures"
rawdata = Path(__file__).parent / "data" / "2023IsotopeDataReport-cleanedinexcel.csv"


df = pd.read_csv(rawdata, header=0, usecols = [
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


for i in range(len(df['Date Run'])):
    if df['Date Run'][i] == '9/6/23':
        df.drop(i, inplace=True)
    else:
        pd.to_datetime(df['Date Run'][i], format = '%m/%d/%y')
        pass
    # 9/6/23 samples were contaiminated 

df.dropna(subset=['Gear Type'], inplace=True) #only scallops and filters are being plotted

data_muscle = df.dropna(subset = ['Tissue Type (Gonad or Muscle)'])
data_muscle = data_muscle.drop(data_muscle[data_muscle['Tissue Type (Gonad or Muscle)'] == 'G'].index)

data_gonad = df.dropna(subset = ['Tissue Type (Gonad or Muscle)'])
data_gonad = data_gonad.drop(data_gonad[data_gonad['Tissue Type (Gonad or Muscle)'] == 'M'].index)

data_muscle_female = data_muscle.drop(data_muscle[data_muscle['Sex']=='M'].index)
data_muscle_male = data_muscle.drop(data_muscle[data_muscle['Sex']=='F'].index)

data_gonad_female = data_gonad.drop(data_gonad[data_gonad['Sex']=='M'].index)
data_gonad_male = data_gonad.drop(data_gonad[data_gonad['Sex']=='F'].index)

'''
Let's confirm that the data is normally distributed
'''

# data_muscle['% N'].hist()
# data_muscle['d13C'].hist()
# data_muscle['d15N'].hist()
# data_muscle['C/N (Molar)'].hist()
# plt.show()

'''
In the line below, change data_muscle to data_gonad depending on which tissue you want to analyze
'''

pd.DataFrame(data_muscle)
anova_data = data_muscle[['% N', 'Gear Type', 'Collection Date']].dropna()

#making columns categorical
anova_data['Gear Type'] = anova_data['Gear Type'].astype('category')
anova_data['Collection Date'] = anova_data['Collection Date'].astype('category')
pd.DataFrame(anova_data)

#renaming columns
anova_data = anova_data.rename(columns={'% N': 'N', 'Gear Type': 'Gear', 'Collection Date': 'Month'})
pd.DataFrame(anova_data)

monthly = data_muscle.groupby('Collection Date') #GONAD OR MUSCLE!!!
june = monthly.get_group(6)
july = monthly.get_group(7)
august = monthly.get_group(8)
sept = monthly.get_group(9)
october = monthly.get_group(10)

gear_types = data_muscle.groupby('Gear Type') 
cages = gear_types.get_group('C')
nets = gear_types.get_group('N')
wild = gear_types.get_group('W')



model = ols('N ~ C(Gear) + C(Month) + C(Gear):C(Month)', data=anova_data).fit()

# Generate the ANOVA table
anova_table = sm.stats.anova_lm(model, type=2) # typ=2 for Type II sum of squares

# print(anova_table)

'''
Let's confirm that the variance is homogeneous
'''
# Levene's test for homogeneity of variances for % N
# a = stats.levene(june['% N'], july['% N'], august['% N'], sept['% N'], october['% N'])
# b = stats.levene(cages['% N'], nets['% N'], wild['% N'])

# Levene's test for homogeneity of variances for d13C
# a = stats.levene(june['d13C'], july['d13C'], august['d13C'], sept['d13C'], october['d13C'])
# b = stats.levene(cages['d13C'], nets['d13C'], wild['d13C'])

# Levene's test for homogeneity of variances for d15N
# a = stats.levene(june['d15N'], july['d15N'], august['d15N'], sept['d15N'], october['d15N'])
# b = stats.levene(cages['d15N'], nets['d15N'], wild['d15N']) 

# Levene's test for homogeneity of variances for C/N (Molar)
a = stats.levene(june['C/N (Molar)'], july['C/N (Molar)'], august['C/N (Molar)'], sept['C/N (Molar)'], october['C/N (Molar)'])
b = stats.levene(cages['C/N (Molar)'], nets['C/N (Molar)'], wild['C/N (Molar)'])

# If p > 0.05, we can assume homogeneity of variances
print(a)
print(b)



'''
Since ANOVA assumptions are not met, let's try PCA
'''
typemap = {
    'C' : 1, 
    'N': 2, 
    'W': 3
}
for x in df['Gear Type']:
    if x in typemap:
        df['Gear Type'] = df['Gear Type'].replace(x, typemap[x])
    else:
        df['Gear Type'] = df['Gear Type'].replace(x, 0)

df = df[df['Gear Type'] != 0]

sexmap = {
    'F' : 1, 
    'M': 2
}

for x in df['Sex']:
    if x in sexmap:
        df['Sex'] = df['Sex'].replace(x, sexmap[x])
    else:
        df['Sex'] = df['Sex'].replace(x, 0)

df = df[df['Sex'] != 0]

tissuemap = {
    'G' : 1, 
    'M': 2
}

for x in df['Tissue Type (Gonad or Muscle)']:
    if x in tissuemap:
        df['Tissue Type (Gonad or Muscle)'] = df['Tissue Type (Gonad or Muscle)'].replace(x, tissuemap[x])
    else:
        df['Tissue Type (Gonad or Muscle)'] = df['Tissue Type (Gonad or Muscle)'].replace(x, 0)

df = df[df['Tissue Type (Gonad or Muscle)'] != 0]

df = df.drop(columns = ['Analysis', 'Sample ID', 'Date Run'])
# print(df.isna().sum())
df = pd.DataFrame(df)

std_scaler = StandardScaler()
scaled_df = std_scaler.fit_transform(df)
pca = PCA(n_components = None)
components = pca.fit_transform(scaled_df)

comp_df = pd.DataFrame(pca.components_, columns = list(df.columns))

# comp_df.to_csv('/Users/adelejordan/Downloads/Hurricane/Isotopes/pca_components2.csv')
explained_variance = pca.explained_variance_ratio_
# loadings = pca.components_.T * np.sqrt(explained_variances)

'''
Let's make a scree plot to visualize the proportion of variance explained by each principal component
'''

# pc_numbers = np.arange(len(explained_variance)) + 1

# plt.figure(figsize=(8, 6))
# plt.plot(pc_numbers, explained_variance, marker='o', linestyle='-')
# plt.title('Scree Plot')
# plt.xlabel('Principal Component Number')
# plt.ylabel('Proportion of Explained Variance')
# plt.grid(True)
# plt.show()

fig, ax = plt.subplots(figsize=(40, 10)) 
ax.axis('off')

'''
Let's make a table to summarize the PCA results
'''

pca.explained_variance_.tolist()
pca.explained_variance_ratio_.tolist()
pca.explained_variance_ratio_.cumsum().tolist()

# summary = [pca.explained_variance_.round(2), pca.explained_variance_ratio_.round(2), pca.explained_variance_ratio_.cumsum().round(2)
# ]

# table = ax.table(cellText=summary,
#                  colLabels=['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10', 'PC11', 'PC12', 'PC13'],
#                  rowLabels=['Explained Var', 'Explained Var Ratio', 'Cum Explained Var Ratio'],
#                  cellLoc = 'center',
#                  rowLoc='center',
#                  loc='center')
# plt.show()

'''
Let's look at score plots to visualize how samples relate to each other in the space defined by the principal components
'''

plt.figure(figsize=(10, 8))


for dates in df['Collection Date']:
    plt.scatter(components[:,0], components[:,1], c = df['Collection Date'], cmap='viridis')
    plt.title('PCA Score Plot: PC1 vs PC2')
    plt.xlim(-4,4)
    plt.ylim(-4,4)

# for dates in df['Gear Type']:
#     plt.scatter(components[:,0], components[:,1], c = df['Gear Type'], cmap='viridis')
#     plt.title('PCA Score Plot: PC1 vs PC2')
#     plt.xlim(-4,4)
#     plt.ylim(-4,4)

plt.show()



