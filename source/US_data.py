import pandas as pd
import numpy as np

#Read data
Canadian_data = pd.read_excel('data/Employee vs. Independent Contractor (August 4, 2021).xlsx',1)
US_data = pd.read_excel('data/length of service US Scales (Borello) - July 21.xlsx')

Canadian_data = Canadian_data.copy(deep=True).drop(columns=['Length of service']).rename(columns={'NEW LENGTH OF SERVICE':'Length of service'}).replace({0.:np.nan})
Canadian_data=Canadian_data[~np.isnan(Canadian_data.Outcome)]

#Clean US datasheet
for var in US_data.keys():
    if var[-1]==' ':
        US_data = US_data.rename(columns={var:var[:-1]})
US_data = US_data.copy(deep=True).drop(columns=['LENGTH OF SERVICE (months)']).rename(columns={'New LENGTH OF SERVICE (months)':'LENGTH OF SERVICE (months)','RESULTS':'Outcome'}).replace({0.:np.nan})
US_data=US_data[~US_data.Outcome.isna()]
US_data=US_data[~US_data['CASE NO.'].isna()]
US_data['RISK OF LOSS'] = US_data['RISK OF LOSS'].str.strip(' \n')


#Create a mapping of values/keys using a comparison chart.
##Value map to the new numeric values consistent across US/Canada.
Mapping_df = pd.read_excel('data/US-Canada-Comparison-Chart_July-26.xlsx')
Mapping_df = Mapping_df.rename(columns={'Unnamed: 5':'OLD US VARIABLE','Unnamed: 8':'OLD CANADA VARIABLE'})
for col in Mapping_df.keys():
    Mapping_df = Mapping_df.rename(columns={ col: col.strip(' \n') })
for var in Mapping_df.US:
    Mapping_df.loc[Mapping_df.US==var,'US'] = str(var).strip(' \n')
for var in Mapping_df.CANADA:
    Mapping_df.loc[Mapping_df.CANADA==var,'CANADA'] = str(var).strip(' \n')
for var in Mapping_df['OLD US VARIABLE']:
    Mapping_df.loc[Mapping_df['OLD US VARIABLE']==var,'OLD US VARIABLE'] = str(var).strip(' \n')
##Variable map
varname_map = Mapping_df[['US','CANADA']].dropna()

##Confirm all the variables in the map are in the original dataset.
print('Variables in US:')
for var in varname_map.US:
    if var in US_data.keys():
        print(var,':in')
    else:
        print(var,':out')
print('Variables in Canada:')
for var in varname_map.CANADA:
    if var in Canadian_data.keys():
        print(var,':in')
    else:
        print(var,':out')

#Convert values
Canada_map = Mapping_df[['CANADA','NEW CANADA VARIABLE','OLD CANADA VARIABLE']].rename(columns={'CANADA':'varname'})
Canada_map = Canada_map[~Canada_map['NEW CANADA VARIABLE'].isna()]
Canada_map = Canada_map.iloc[:-3]
Canada_map = Canada_map.fillna(method='pad')
Canada_variables = list(Canada_map['varname'].unique() )+ ['Outcome', 'Length of service']

US_map = Mapping_df[['US','NEW US VARIABLE','OLD US VARIABLE']].rename(columns={'US':'varname'})
US_map = US_map[~US_map['NEW US VARIABLE'].isna()]
US_map = US_map.iloc[:-3]
US_map = US_map.fillna(method='pad')
US_variables = list( US_map['varname'].unique() ) + ['Outcome', 'LENGTH OF SERVICE (months)']

##Convert values in Canadian data
for var in Canadian_data.keys():
    var_map = Canada_map[Canada_map['varname']==var][['OLD CANADA VARIABLE','NEW CANADA VARIABLE']].set_index('OLD CANADA VARIABLE').to_dict()
    Canadian_data[var] = Canadian_data[[var]].replace(var_map)
Canadian_data2 = Canadian_data[Canada_variables]
Canadian_data2 =  Canadian_data2.replace('\xa0',np.nan)

##Check if all the values are numeric
unconverted_Canada = []
for var in Canadian_data2.keys():
    for value in Canadian_data2[var].unique():
        if (not isinstance(value,float)) and (not isinstance(value,int)):
            unconverted_Canada.append([var,value])
unconverted_Canada_df = pd.DataFrame(unconverted_Canada,columns=['variable','value'])


##Convert values in US data
for var in US_data.keys():
    var_map = US_map[US_map['varname']==var][['OLD US VARIABLE','NEW US VARIABLE']].set_index('OLD US VARIABLE').to_dict()['NEW US VARIABLE']
    US_data[var] = US_data[[var]].replace(var_map)
US_data2 = US_data[US_variables]
US_data2['Length of service'] = round(US_data2['LENGTH OF SERVICE (months)']/12)
US_data2 = US_data2.drop(columns=['LENGTH OF SERVICE (months)'])
US_data2 = US_data2.replace({'Independent Contractor':2, 'Employee':1})



##Check if all the values are numeric
for var in US_data2.keys():
    print(var,US_data2[var].unique())
unconverted_US = []
for var in US_data2.keys():
    for value in US_data2[var].unique():
        if not isinstance(value,float):
            unconverted_US.append([var,value])
unconverted_US_df = pd.DataFrame(unconverted_US,columns=['variable','value'])

#Convert the variable names in US data to match to the Canadian data
US_data3 = US_data2.rename(columns=varname_map.set_index('US').to_dict()['CANADA'])
consistent_vars = set(US_data3.keys()).intersection( set(Canadian_data2.keys()) )


#Merge data
US_data4 = US_data3[consistent_vars]
US_data4 = US_data4.loc[:,~US_data4.columns.duplicated()]
Canadian_data3  = Canadian_data2[consistent_vars]

#Merge the two datasets.
merged_data = pd.concat((Canadian_data3,US_data4),axis=0)
merged_data['source'] = ['Canada'] * len(Canadian_data3) + ['US'] * len(US_data4)
merged_data.to_csv('data/US_Canada_merged.csv')
