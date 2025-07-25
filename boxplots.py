import pandas as pd
from matplotlib import pyplot as plt


file_path = '/Users/adelejordan/Downloads/ScallopData.csv'
data = pd.read_csv(file_path, header=0, usecols = [
    'Analysis', 
    'Sample ID', 
    'Type',	
    'Location',	
    'Sex', 	
    'Gonad or Meat',	
    'Number',	
    'Mass (mg)',	
    '% N',	
    'N umoles',	
    'd15N',	
    '%C',	
    'C umoles',	
    'd13C',	
    'C/N (Molar)',	
    'Date Run'])

#print(data.head())



types = data.groupby('Location')
cages = types.get_group('C')
nets = types.get_group('N')
wild = types.get_group('W')



'''
box plots for d15N and d13C by location
'''

# plt.subplots(1, 3, figsize=(4, 3), sharex=False, sharey=True)
# plt.subplot(1, 3, 1)
# plt.boxplot(cages['d15N'])
# plt.title('Cages')

# plt.subplot(1, 3, 2)
# plt.boxplot(nets['d15N'])
# plt.title('Nets')

# plt.subplot(1, 3, 3)
# plt.boxplot(wild['d15N'])
# plt.title('Wild')

# plt.suptitle('d15N by Location')
# plt.show()

# plt.subplots(1, 3, figsize=(4, 3), sharex=False, sharey=True)
# plt.subplot(1, 3, 1)
# plt.boxplot(cages['d13C'])
# plt.title('Cages')

# plt.subplot(1, 3, 2)
# plt.boxplot(nets['d13C'])
# plt.title('Nets')

# plt.subplot(1, 3, 3)
# plt.boxplot(wild['d13C'])
# plt.title('Wild')

# plt.suptitle('d13C by Location')
# plt.show()

'''
box plots for %N and %C by location
'''
# plt.subplots(1, 3, figsize=(4, 3), sharex=False, sharey=True)
# plt.subplot(1, 3, 1)
# plt.boxplot(cages['% N'])
# plt.title('Cages')

# plt.subplot(1, 3, 2)
# plt.boxplot(nets['% N'])
# plt.title('Nets')

# plt.subplot(1, 3, 3)
# plt.boxplot(wild['% N'])
# plt.title('Wild')

# plt.suptitle('%N by Location')
# plt.show()

# plt.subplots(1, 3, figsize=(4, 3), sharex=False, sharey=True)
# plt.subplot(1, 3, 1)
# plt.boxplot(cages['%C'])
# plt.title('Cages')

# plt.subplot(1, 3, 2)
# plt.boxplot(nets['%C'])
# plt.title('Nets')

# plt.subplot(1, 3, 3)
# plt.boxplot(wild['%C'])
# plt.title('Wild')

# plt.suptitle('%C by Location')
# plt.show()

# sex = data.groupby('Sex')
# female = sex.get_group('F')
# male = sex.get_group('M')
#print(male.head())

# plt.subplots(1, 3, figsize=(4, 3), sharex=False, sharey=True)
# plt.subplot(1, 3, 1)
# plt.boxplot(male['d15N'])
# plt.title('Male')

# plt.subplot(1, 3, 2)
# plt.boxplot(female['d15N'])
# plt.title('Female')

# plt.suptitle('d15N by Sex')
# plt.show()

# plt.subplots(1, 3, figsize=(4, 3), sharex=False, sharey=True)
# plt.subplot(1, 3, 1)
# plt.boxplot(male['d13C'])
# plt.title('Male')

# plt.subplot(1, 3, 2)
# plt.boxplot(nets['d13C'])
# plt.title('Female')

# plt.suptitle('d13C by Sex')
# plt.show()

cages_df = pd.DataFrame(cages)
nets_df = pd.DataFrame(nets)
wild_df = pd.DataFrame(wild)

cage_sex = cages_df.groupby(('Sex'))
female_cages = cage_sex.get_group(('F'))
male_cages = cage_sex.get_group('M')

net_sex = nets_df.groupby('Sex')
female_nets = net_sex.get_group('F')
male_nets = net_sex.get_group('M')

wild_sex = wild_df.groupby('Sex')
female_wild = wild_sex.get_group('F')
male_wild = wild_sex.get_group('M')

plt.subplots(1, 6, figsize=(4, 3), sharex=False, sharey=True)
plt.subplot(1, 6, 1)
plt.boxplot(female_cages['d15N'])
plt.title('Cages Female')

plt.subplot(1, 6, 2)
plt.boxplot(male_cages['d15N'])
plt.title('Cages Male')

plt.subplot(1, 6, 3)
plt.boxplot(female_nets['d15N'])
plt.title('Nets Female')

plt.subplot(1, 6, 4)
plt.boxplot(male_nets['d15N'])
plt.title('Nets Male')

plt.subplot(1, 6, 5)
plt.boxplot(female_wild['d15N'])
plt.title('Wild Female')

plt.subplot(1, 6, 6)
plt.boxplot(male_wild['d15N'])
plt.title('Wild Male')

plt.suptitle('d15N by Sex')
plt.show()

plt.subplots(1, 6, figsize=(4, 3), sharex=False, sharey=True)
plt.subplot(1, 6, 1)
plt.boxplot(female_cages['d13C'])
plt.title('Cages Female')

plt.subplot(1, 6, 2)
plt.boxplot(male_cages['d13C'])
plt.title('Cages Male')

plt.subplot(1, 6, 3)
plt.boxplot(female_nets['d13C'])
plt.title('Nets Female')

plt.subplot(1, 6, 4)
plt.boxplot(male_nets['d13C'])
plt.title('Nets Male')

plt.subplot(1, 6, 5)
plt.boxplot(female_wild['d13C'])
plt.title('Wild Female')

plt.subplot(1, 6, 6)
plt.boxplot(male_wild['d13C'])
plt.title('Wild Male')

plt.suptitle('d13C by Sex')
plt.show()

# plt.subplots(1, 3, figsize=(4, 3), sharex=False, sharey=True)
# plt.subplot(1, 3, 1)
# plt.boxplot(male['d13C'])
# plt.title('Male')

# plt.subplot(1, 3, 2)
# plt.boxplot(nets['d13C'])
# plt.title('Female')

# plt.suptitle('d13C by Sex')
# plt.show()

