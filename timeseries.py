from pandas import pandas as pd

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

dates = pd.read_csv(file_path, header=0, usecols = ['BCID', 'Date Run'])