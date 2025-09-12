import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA    
from scipy.stats import kruskal
import seaborn as sns

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


df.insert(4, 'Column Height', 0)
mapping = {'C': 1, 'N': 2, 'W': 1}
df['Column Height'] = df['Gear Type'].map(mapping) #C and W are at the same depth

df.insert(5, 'Farmed or Wild', 0)
mapping_farmvswild = {'C': '1', 'N': '1', 'W': '2'}
df['Farmed or Wild'] = df['Gear Type'].map(mapping_farmvswild) #C and N are farmed, W is wild

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
anova_data = data_muscle[['d13C', '% N', 'd15N', 'C/N (Molar)', 'Gear Type', 'Collection Date', 'Column Height', 'Farmed or Wild']].dropna() #'Farmed or Wild'

#making columns categorical
anova_data['Gear Type'] = anova_data['Gear Type'].astype('category')
anova_data['Collection Date'] = anova_data['Collection Date'].astype('category')
pd.DataFrame(anova_data)

#renaming columns
anova_data = anova_data.rename(columns={'% N': 'N', 'Gear Type': 'Gear', 'Collection Date': 'Month', 'Column Height': 'Depth', 'C/N (Molar)': 'CN', 'Farmed or Wild' : 'farmvwild'})
pd.DataFrame(anova_data)
#'Farmed or Wild' : 'farmvwild'

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



# model = ols('N ~ C(Gear) + C(Month) + C(Gear):C(Month)', data=anova_data).fit()
# model_oneway = ols('N ~ C(farmvwild)', data=anova_data).fit()
# # Generate the ANOVA table
# anova_table = sm.stats.anova_lm(model, type=2) # typ=2 for Type II sum of squares
# one_way_anova = sm.stats.anova_lm(model_oneway, type=2)

# print(one_way_anova)

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
# print(a)
# print(b)




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


'''
Based on tissue map, decide if you want only gonad or muscle
'''
df = df.drop(df[df['Tissue Type (Gonad or Muscle)'] == 1].index) 




'''
ADELE COME BACK TO THIS
'''
df_muscle_farm = df[df['Farmed or Wild'] != 2]
df_muscle_wild = df[df['Farmed or Wild'] != 1]

results = kruskal(df_muscle_farm['d15N'], df_muscle_wild['d15N'])
print(results)


df = df.drop(columns = ['Analysis', 'Sample ID', 'Date Run'])
# print(df.isna().sum())
df = pd.DataFrame(df)

std_scaler = StandardScaler()
scaled_df = std_scaler.fit_transform(df)
pca = PCA(n_components = None)
components = pca.fit_transform(scaled_df)
explained_variance = pca.explained_variance_ratio_
loadings = pca.components_.T * np.sqrt(explained_variance) #need to double check this


pairplot_df = df[['% N', 'N (umoles)', 'd15N', '%C', 'C (umoles)', 'd13C','C/N (Molar)']]
pairplot = sns.pairplot(df[['% N', 'N (umoles)', 'd15N', '%C', 'C (umoles)', 'd13C','C/N (Molar)', 'Sex']], hue='Sex')
sns.color_palette("hls", 8)
plt.savefig(figures / "pairplot-muscle-collectiondate.png")

'''
ADELE NEEDS TO MAKE THESE INTO TABLES
'''
comp_df = pd.DataFrame(pca.components_, columns = list(df.columns))
comp_df.to_csv('/Users/adelejordan/Downloads/Hurricane/Isotopes/pca_components4.csv')

loadings_df = pd.DataFrame(pca.components_.T * np.sqrt(explained_variance), columns = list(df.columns))
loadings_df.to_csv('/Users/adelejordan/Downloads/Hurricane/Isotopes/pca_loadings4.csv')

# summary = [pca.explained_variance_.round(2), pca.explained_variance_ratio_.round(2), pca.explained_variance_ratio_.cumsum().round(2)
# ]


fig, ax = plt.subplots(figsize=(8, 4)) 
ax.axis('off')

loadings_table = ax.table(cellText = loadings.round(2),
                 colLabels = df.columns,
                 rowLabels = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10', 'PC11', 'PC12', 'PC13', 'PC14', 'PC15'],
                 cellLoc = 'center',
                 rowLoc = 'center',
                 loc = 'center')
loadings_table.auto_set_font_size(False)
loadings_table.set_fontsize(8) 

# plt.show()

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



'''
Let's make a table to summarize the PCA results
'''

# pca.explained_variance_.tolist()
# pca.explained_variance_ratio_.tolist()
# pca.explained_variance_ratio_.cumsum().tolist()

# summary = [pca.explained_variance_.round(2), pca.explained_variance_ratio_.round(2), pca.explained_variance_ratio_.cumsum().round(2)
# ]

# fig, ax = plt.subplots(figsize=(40, 10)) 
# ax.axis('off')

# table = ax.table(cellText=summary,
#                  colLabels=['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10', 'PC11', 'PC12', 'PC13', 'PC14', 'PC15'],
#                  rowLabels=['Explained Var', 'Explained Var Ratio', 'Cum Explained Var Ratio'],
#                  cellLoc = 'center',
#                  rowLoc='center',
#                  loc='center')
# plt.show()

'''
Let's look at score plots to visualize how samples relate to each other in the space defined by the principal components
'''

plt.figure(figsize=(10, 8))


# for dates in df['Collection Date']:
#     plt.scatter(components[:,0], components[:,1], c = df['Collection Date'], cmap='viridis')
#     plt.title('PCA Score Plot: PC1 vs PC2')
#     plt.xlim(-4,4)
#     plt.ylim(-4,4)


for dates in df['Farmed or Wild']:
    plt.scatter(components[:,0], components[:,1], c = df['Farmed or Wild'].astype('category').cat.codes, cmap='viridis')
    plt.title('PCA Score Plot: PC1 vs PC2')
    plt.xlim(-4,4)
    plt.ylim(-4,4)

# for dates in df['Collection Date']:
#     plt.scatter(components[:,0], components[:,1], c = df['Collection Date'].astype('category').cat.codes, cmap='viridis')
#     plt.title('PCA Score Plot: PC1 vs PC2')
#     plt.xlim(-4,4)
#     plt.ylim(-4,4)


# plt.show()



