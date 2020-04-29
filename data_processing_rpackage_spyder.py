import pandas as pd

path_r_data = r'C:\Users\sesig\Documents\master data science\tfm\r_dataset\r_data.csv'
data = pd.read_csv(filepath_or_buffer=path_r_data, sep=',')
print(data.head())
    
cj_split = str.split(data.iloc[0,0],'>')

for j in cj_split:
    print(j)