import pandas as pd
import matplotlib.pyplot as plt

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


for i in range(len(data['Date Run'])):
    if data['Date Run'][i] == '9/6/23':
        data.drop(i, inplace=True)
    else:
        pd.to_datetime(data['Date Run'][i], format = '%m/%d/%y')
        pass

monthly = data.groupby('Collection Date')
june = monthly.get_group(6)
july = monthly.get_group(7)
august = monthly.get_group(8)
sept = monthly.get_group(9)
october = monthly.get_group(10)

# plt.subplots(1, 5, figsize=(4, 3), sharex=False, sharey=True)
# plt.subplot(1, 5, 1)
# plt.boxplot(june['d15N'])
# plt.title('June')

# plt.subplot(1, 5, 2)
# plt.boxplot(july['d15N'])
# plt.title('July')

# plt.subplot(1, 5, 3)
# plt.boxplot(august['d15N'])
# plt.title('August')

# plt.subplot(1, 5, 4)
# plt.boxplot(sept['d15N'])
# plt.title('September')

# plt.subplot(1, 5, 5)
# plt.boxplot(october['d15N'])
# plt.title('October')

# plt.suptitle('d15N by Month')
# plt.show()

# plt.subplots(1, 5, figsize=(4, 3), sharex=False, sharey=True)
# plt.subplot(1, 5, 1)
# plt.boxplot(june['d13C'])
# plt.title('June')

# plt.subplot(1, 5, 2)
# plt.boxplot(july['d13C'])
# plt.title('July')

# plt.subplot(1, 5, 3)
# plt.boxplot(august['d13C'])
# plt.title('August')

# plt.subplot(1, 5, 4)
# plt.boxplot(sept['d13C'])
# plt.title('August')

# plt.subplot(1, 5, 5)
# plt.boxplot(october['d13C'])
# plt.title('October')

# plt.suptitle('d13C by Month')
# plt.show()

june_df = pd.DataFrame(june)
july_df = pd.DataFrame(july)
august_df = pd.DataFrame(august)
sept_df = pd.DataFrame(sept)
oct_df = pd.DataFrame(october)

jgear = june_df.groupby('Gear Type')
j_cage = jgear.get_group('C')
j_net = jgear.get_group('N')
j_wild = jgear.get_group('W')

jugear = july_df.groupby('Gear Type')
ju_cage = jugear.get_group('C')
ju_net = jugear.get_group('N')
ju_wild = jugear.get_group('W')

agear = august_df.groupby('Gear Type')
a_cage = agear.get_group('C')
a_net = agear.get_group('N')
a_wild = agear.get_group('W')

sgear = sept_df.groupby('Gear Type')
s_cage = sgear.get_group('C')
s_net = sgear.get_group('N')
s_wild = sgear.get_group('W')

ogear = oct_df.groupby('Gear Type')
o_cage = ogear.get_group('C')
o_net = ogear.get_group('N')
o_wild = ogear.get_group('W')

# plt.subplots(1, 6, figsize=(4, 3), sharex=False, sharey=True)
# plt.subplot(1, 6, 1)
# plt.boxplot(j_cage['d13C'])
# plt.title('June Cage')

# plt.subplot(1, 6, 2)
# plt.boxplot(j_net['d13C'])
# plt.title('June Net')

# plt.subplot(1, 6, 3)
# plt.boxplot(j_wild['d13C'])
# plt.title('June Wild')

# plt.subplot(1, 6, 4)
# plt.boxplot(o_cage['d13C'])
# plt.title('Oct Cage')

# plt.subplot(1, 6, 5)
# plt.boxplot(o_net['d13C'])
# plt.title('Oct Net')

# plt.subplot(1, 6, 6)
# plt.boxplot(o_wild['d13C'])
# plt.title('Oct wild')

# plt.suptitle('d13C by Month')
# plt.show()

# plt.subplots(1, 6, figsize=(4, 3), sharex=False, sharey=True)
# plt.subplot(1, 6, 1)
# plt.boxplot(j_cage['d15N'])
# plt.title('June Cage')

# plt.subplot(1, 6, 2)
# plt.boxplot(j_net['d15N'])
# plt.title('June Net')

# plt.subplot(1, 6, 3)
# plt.boxplot(j_wild['d15N'])
# plt.title('June Wild')

# plt.subplot(1, 6, 4)
# plt.boxplot(o_cage['d15N'])
# plt.title('Oct Cage')

# plt.subplot(1, 6, 5)
# plt.boxplot(o_net['d15N'])
# plt.title('Oct Net')

# plt.subplot(1, 6, 6)
# plt.boxplot(o_wild['d15N'])
# plt.title('Oct wild')

# plt.suptitle('d15N by Month')
# plt.show()

# plt.subplots(1, 5, figsize=(4, 3), sharex=False, sharey=True)
# plt.subplot(1, 5, 1)
# plt.boxplot(j_cage['d13C'])
# plt.title('June Cage')

# plt.subplot(1, 5, 2)
# plt.boxplot(ju_cage['d13C'])
# plt.title('July Cage')

# plt.subplot(1, 5, 3)
# plt.boxplot(a_cage['d13C'])
# plt.title('Aug Cage')

# plt.subplot(1, 5, 4)
# plt.boxplot(s_cage['d13C'])
# plt.title('Sept Cage')

# plt.subplot(1, 5, 5)
# plt.boxplot(o_cage['d13C'])
# plt.title('Oct Cage')

# plt.suptitle('d13C by Month')
# plt.show()

# plt.subplots(1, 5, figsize=(4, 3), sharex=False, sharey=True)
# plt.subplot(1, 5, 1)
# plt.boxplot(j_wild['d13C'])
# plt.title('June Wild')

# plt.subplot(1, 5, 2)
# plt.boxplot(ju_wild['d13C'])
# plt.title('July Wild')

# plt.subplot(1, 5, 3)
# plt.boxplot(a_wild['d13C'])
# plt.title('Aug Wild')

# plt.subplot(1, 5, 4)
# plt.boxplot(s_wild['d13C'])
# plt.title('Sept Wild')

# plt.subplot(1, 5, 5)
# plt.boxplot(o_wild['d13C'])
# plt.title('Oct Wild')

# plt.suptitle('d13C by Month')
# plt.show()

# plt.subplots(1, 5, figsize=(4, 3), sharex=False, sharey=True)
# plt.subplot(1, 5, 1)
# plt.boxplot(j_net['d13C'])
# plt.title('June Net')

# plt.subplot(1, 5, 2)
# plt.boxplot(ju_net['d13C'])
# plt.title('July Net')

# plt.subplot(1, 5, 3)
# plt.boxplot(a_net['d13C'])
# plt.title('Aug Net')

# plt.subplot(1, 5, 4)
# plt.boxplot(s_net['d13C'])
# plt.title('Sept Net')

# plt.subplot(1, 5, 5)
# plt.boxplot(o_net['d13C'])
# plt.title('Oct Net')

# plt.suptitle('d13C by Month')
# plt.show()

# plt.subplots(1, 5, figsize=(4, 3), sharex=False, sharey=True)
# plt.subplot(1, 5, 1)
# plt.boxplot(j_cage['d15N'])
# plt.title('June Cage')

# plt.subplot(1, 5, 2)
# plt.boxplot(ju_cage['d15N'])
# plt.title('July Cage')

# plt.subplot(1, 5, 3)
# plt.boxplot(a_cage['d15N'])
# plt.title('Aug Cage')

# plt.subplot(1, 5, 4)
# plt.boxplot(s_cage['d15N'])
# plt.title('Sept Cage')

# plt.subplot(1, 5, 5)
# plt.boxplot(o_cage['d15N'])
# plt.title('Oct Cage')

# plt.suptitle('d15N by Month')
# plt.show()

# plt.subplots(1, 5, figsize=(4, 3), sharex=False, sharey=True)
# plt.subplot(1, 5, 1)
# plt.boxplot(j_wild['d15N'])
# plt.title('June Wild')

# plt.subplot(1, 5, 2)
# plt.boxplot(ju_wild['d15N'])
# plt.title('July Wild')

# plt.subplot(1, 5, 3)
# plt.boxplot(a_wild['d15N'])
# plt.title('Aug Wild')

# plt.subplot(1, 5, 4)
# plt.boxplot(s_wild['d15N'])
# plt.title('Sept Wild')

# plt.subplot(1, 5, 5)
# plt.boxplot(o_wild['d15N'])
# plt.title('Oct Wild')

# plt.suptitle('d15N by Month')
# plt.show()

# plt.subplots(1, 5, figsize=(4, 3), sharex=False, sharey=True)
# plt.subplot(1, 5, 1)
# plt.boxplot(j_net['d15N'])
# plt.title('June Net')

# plt.subplot(1, 5, 2)
# plt.boxplot(ju_net['d15N'])
# plt.title('July Net')

# plt.subplot(1, 5, 3)
# plt.boxplot(a_net['d15N'])
# plt.title('Aug Net')

# plt.subplot(1, 5, 4)
# plt.boxplot(s_net['d15N'])
# plt.title('Sept Net')

# plt.subplot(1, 5, 5)
# plt.boxplot(o_net['d15N'])
# plt.title('Oct Net')

# plt.suptitle('d15N by Month')
# plt.show()

gear = data.groupby('Gear Type')
cages = gear.get_group('C')
nets = gear.get_group('N')
wild = gear.get_group('W')

cages_df = pd.DataFrame(cages)
nets_df = pd.DataFrame(nets)
wild_df = pd.DataFrame(wild)

# plt.scatter(cages_df['d15N'], cages_df['d13C'], c=cages_df['Collection Date'], cmap='viridis')
# plt.title('Cages: d15N vs d13C')
# plt.legend(loc='upper right')
# plt.show()

# plt.scatter(nets_df['d15N'], nets_df['d13C'], c=nets_df['Collection Date'], cmap='viridis')
# plt.title('Nets: d15N vs d13C')
# plt.show()

# plt.scatter(wild_df['d15N'], wild_df['d13C'], c=wild_df['Collection Date'], cmap='viridis')
# plt.title('Wild: d15N vs d13C')
# plt.show()

data.dropna(subset=['Gear Type'], inplace=True)

colormap = {
    'C': 'b', 
    'N': 'g', 
    'W': 'r'
}

# for x in data['Gear Type']:
#     if x in colormap:
#         plt.scatter(data[data['Gear Type'] == x]['d15N'], 
#                     data[data['Gear Type'] == x]['d13C'], 
#                     c=colormap[x], 
#                     s=50)  # s is the size of the markers

# # plt.scatter(data['d15N'], data['d13C'], s = ,  color = data['Gear Type']) #doesnt work because c, n, and w are also colors and markers?
# plt.xlabel('d15N')
# plt.ylabel('d13C')
# plt.legend(loc='upper right')
# plt.show()

plt.subplots(2, 2, figsize =(10,8), sharex=False, sharey = False)

for x in june_df['Gear Type']:
    if x in colormap:
        plt.subplot(2, 2, 1)
        plt.scatter(june_df[june_df['Gear Type'] == x]['d15N'], 
                    june_df[june_df['Gear Type'] == x]['d13C'], 
                    c=colormap[x], 
                    s=50)  # s is the size of the markers

# plt.scatter(data['d15N'], data['d13C'], s = ,  color = data['Gear Type']) #doesnt work because c, n, and w are also colors and markers?
plt.xlabel('d15N')
plt.ylabel('d13C')
plt.legend(loc='upper right')

for x in oct_df['Gear Type']:
    if x in colormap:
        plt.subplot(2, 2, 2)
        plt.scatter(oct_df[oct_df['Gear Type'] == x]['d15N'], 
                    oct_df[oct_df['Gear Type'] == x]['d13C'], 
                    c=colormap[x], 
                    s=50)  # s is the size of the markers

# plt.scatter(data['d15N'], data['d13C'], s = ,  color = data['Gear Type']) #doesnt work because c, n, and w are also colors and markers?
plt.xlabel('d15N')
plt.ylabel('d13C')
plt.legend(loc='upper right')

for x in august_df['Gear Type']:
    if x in colormap:
        plt.subplot(2, 2, 3)
        plt.scatter(august_df[august_df['Gear Type'] == x]['d15N'], 
                    august_df[august_df['Gear Type'] == x]['d13C'], 
                    c=colormap[x], 
                    s=50)  # s is the size of the markers

# plt.scatter(data['d15N'], data['d13C'], s = ,  color = data['Gear Type']) #doesnt work because c, n, and w are also colors and markers?
plt.xlabel('d15N')
plt.ylabel('d13C')
plt.legend(loc='upper right')
plt.show()