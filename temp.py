from pathlib import Path
import pandas as pd
import datetime as dt


figures = Path(__file__).parent / "figures"
rawdata = Path(__file__).parent / "data" / "2023_2022_GSI_Environmental_Data_2023_Temperature_and_Light.csv"

data = pd.read_csv(rawdata, header = 0)

'''
THIS IS WRONG BECAUSE IT COUNTS HOURS, NOT DAYS
FIND THE MAX FOR EACH DAY, THEN COUNT THE DAYS
'''

data['Date-Time (EDT)'] = pd.to_datetime(data['Date-Time (EDT)'], format = '%m/%d/%y %H:%M')

'''
For this data:
Groups:
    6/15/23 [0:13]
    6/16/23 - 7/13/23 [14:687]
    7/14/23 - 8/14/23 [689:1455]
    8/15/23 - 9/15/23 [1456:2223]
    9/16/23 - 10/12/23 [2224:2864]
'''

degree_day = 53.6

june_surface_hour_count = 0
for x in range(len(data['Date-Time (EDT)'][0:13])):
    if data['Net Top, Temperature (°F)'][x] > degree_day:
        june_surface_hour_count += 1
    else: 
        pass

june_bottom_hour_count = 0
for x in range(len(data['Date-Time (EDT)'][0:13])):
    if data['Cage, Temperature (°F)'][x] > degree_day:
        june_bottom_hour_count += 1
    else: 
        pass

july_surface_hour_count = 0
for x in range(len(data['Date-Time (EDT)'][14:687])):
    if data['Net Top, Temperature (°F)'][x] > degree_day:
        july_surface_hour_count += 1
    else: 
        pass

july_bottom_hour_count = 0
for x in range(len(data['Date-Time (EDT)'][14:687])):
    if data['Cage, Temperature (°F)'][x] > degree_day:
        july_bottom_hour_count += 1
    else: 
        pass

aug_surface_hour_count = 0
for x in range(len(data['Date-Time (EDT)'][689:1455])):
    if data['Net Top, Temperature (°F)'][x] > degree_day:
        aug_surface_hour_count += 1
    else: 
        pass

aug_bottom_hour_count = 0
for x in range(len(data['Date-Time (EDT)'][689:1455])):
    if data['Cage, Temperature (°F)'][x] > degree_day:
        aug_bottom_hour_count += 1
    else: 
        pass

sept_surface_hour_count = 0
for x in range(len(data['Date-Time (EDT)'][1456:2223])):
    if data['Net Top, Temperature (°F)'][x] > degree_day:
        sept_surface_hour_count += 1
    else: 
        pass

sept_bottom_hour_count = 0
for x in range(len(data['Date-Time (EDT)'][1456:2223])):
    if data['Cage, Temperature (°F)'][x] > degree_day:
        sept_bottom_hour_count += 1
    else: 
        pass

oct_surface_hour_count = 0
for x in range(len(data['Date-Time (EDT)'][2224:2864])):
    if data['Net Top, Temperature (°F)'][x] > degree_day:
        oct_surface_hour_count += 1
    else: 
        pass

oct_bottom_hour_count = 0
for x in range(len(data['Date-Time (EDT)'][2224:2864])):
    if data['Cage, Temperature (°F)'][x] > degree_day:
        oct_bottom_hour_count += 1
    else: 
        pass


temp_data = {
    'Month': ['June', 'July', 'August', 'September', 'October'],
    'Surface Degree Hours': [june_surface_hour_count, july_surface_hour_count, aug_surface_hour_count, sept_surface_hour_count, oct_surface_hour_count],
    'Bottom Degree Hours': [june_bottom_hour_count, july_bottom_hour_count, aug_bottom_hour_count, sept_bottom_hour_count, oct_bottom_hour_count]
}

print(temp_data)