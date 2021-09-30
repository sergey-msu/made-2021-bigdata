import os
import subprocess
import pandas as pd


prices = pd.read_csv('AB_NYC_2019.csv')['price'].dropna().values
mean_check = prices.mean()
var_check = ((prices - prices.mean())**2).mean()

print('Mean from Map/Reduce:')
os.system('python mean_mapper.py < AB_NYC_2019.csv | python mean_reducer.py')

print('Var from Map/Reduce:')
os.system('python var_mapper.py < AB_NYC_2019.csv | python var_reducer.py')

print('Mean from formula:', mean_check)
print('Var from formula: ', var_check)

