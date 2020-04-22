import pandas as pd

path_r_data = r'C:\Users\sesig\Documents\master data science\tfm\r_dataset\r_data.csv'
data = pd.read_csv(filepath_or_buffer=path_r_data, sep=',')
print(data.head())
    
data_processed = pd.DataFrame(columns=['uid','campaign','total_conversion','total_null'])

for i in data.index:
    print(i)
    cj = data.iloc[i,0]
    cj_split = str.split(cj, '>')
    for j in cj_split:
        data_to_append = {'uid':i, 'campaign':j, 'total_conversions':data.iloc[i,1], 'total_null':data.iloc[i,3]}
        data_processed = pd.DataFrame.append(data_processed, data_to_append, ignore_index=True)

path_processed = r'C:\Users\sesig\Documents\master data science\tfm\criteo_cleaned_data\r_dataset_cleaned.csv'
pd.DataFrame.to_csv(data_processed,path_or_buf=path_processed,sep=',',index=False)
