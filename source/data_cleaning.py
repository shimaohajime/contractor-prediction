'''
Code to prepare the Canadian contractor-employee dataset for the predictive analysis.
This code should be run before predictions.py.
'''

import pandas as pd
import numpy  as np
import pickle
from source.utils import get_y_from_string, process_length, fill_database, impute_continuous
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
np.random.seed(0)


#df_raw = pd.read_excel('data/length of service CA Sagaz July 21.xlsx',1)
df_raw = pd.read_excel('data/Employee vs. Independent Contractor (August 4, 2021).xlsx',1)
df_raw.index = df_raw.index+2 #Match to the row number in excel sheet.


## Method for filling the missing values by imputation.
##FillType = 0, 1, 2 are the old implementation (not tested with the current version). The default is FillType=3.
FillType = 3
IndustryDummy = True #If True, use Indutry variable in one-hot coding.
ScaleX = False #If True, apply standard scaler to the data.
drop_if_missing = False #If integer, drop the samples with non-missing variables less than this number.

##Clean and keep only the relevant columns.
col = list(df_raw.columns[2:-2])
##Note: 'NEW LENGTH OF SERVICE' is the cleaned data for service length. The original 'Length of service' contains human notes and should not be used.
df1 = df_raw.copy(deep=True).drop(columns=['Length of service']).rename(columns={'NEW LENGTH OF SERVICE':'Length of service'}).replace({0.:np.nan})
##Correct a mistake in Barton v. 615208 N.B. Inc
df1.loc[df1['Risk of loss']==5,'Risk of loss']=1

df1=df1.replace('nan',np.nan)
df1=df1.replace('\xa0', 0)
df1=df1[~np.isnan(df1.Outcome)]

'''
Read the newly audited data.
'''
df_raw2 = pd.read_excel('data/Sagaz Update - August 27, 2021.xlsx',0)
df_raw2.index = df_raw2.index+3 #Match to the row number in excel sheet.
df_raw2 = df_raw2.iloc[1:]
for col in df_raw2.keys():
    df_raw2 = df_raw2.rename(columns={ col: col.strip(' \n').capitalize() })
df_raw2 = df_raw2.rename(columns={'Chance of profit refers to how worker is paid':'Chance of profit','Type of job':'Type of Job','Who set the work hours':'Who sets the work hours','Case name and case citation':'Case name and (case citation)'})

##Clean the new data.
df2 = df_raw2.copy(deep=True)
df2['Length of service'] = df2['Length of service'].replace({'3+1+3+1+3+1/6 = 2':2})
for col in df2.keys():
    if col not in ['Length of service', 'Name', 'Case name and (case citation)', 'Date', 'Court','Judge name','Notes','Column1']:
        df2[col] = pd.to_numeric( df2[col].str.extract('(^\d*)')[0] )
df2 = df2.replace({0:np.nan, 'N/a':np.nan})


#Merge the new data with the base
##Merge the pre-cleaned datasets and save it once.
df = pd.concat((df1,df2),axis=0).rename(columns={'Case name and (case citation)':'case_identifier'})
df.to_csv('data/Canada_merged_raw.csv')



'''
Construt the dataset for analysis.
'''
RelevantFeatures=['Industry',
       'Length of service', 'Supervision/review of work',
       'Ability to hire employees', 'Delegation of tasks',
       'Ownership of tools', 'Chance of profit', 'Risk of loss',
       'Exclusivity of services', 'Who sets the work hours',
       'Where the work is performed',
       'Is the worker required to wear a uniform?']
CategoricalFeatures = ['Industry',#'Type of Job',#'Occupation',#'Province',
       'Supervision/review of work',
       'Delegation of tasks',
       'Where the work is performed',
       'Is the worker required to wear a uniform?']
ContinuousFeatures = ['Length of service', 'Ownership of tools', 'Ability to hire employees', 'Chance of profit','Risk of loss','Exclusivity of services','Who sets the work hours']

df = df[df.Outcome.isin([1, 2])]
case_identifier = df['case_identifier']
df = df[CategoricalFeatures+ContinuousFeatures+['Outcome']]
df = df.astype(float)
df = pd.concat( (df, case_identifier), axis=1 )
df.loc[df['case_identifier'].isna(),'case_identifier'] = 'none'

df.to_csv('data/unprocessed_data.csv')


'''
Summary statistics before imputation
'''
df_summary = pd.DataFrame(index=df.drop(columns=['case_identifier']).keys())
df_summary['Type'] = ['Categorical']*len(CategoricalFeatures) + ['Continuous'] + ['Ordinal']*(len(ContinuousFeatures)-1) + ['Categorical']
df_summary['Min'] = df.drop(columns=['case_identifier']).min().astype(int)
df_summary['Max'] = df.drop(columns=['case_identifier']).max().astype(int)
df_summary['Median'] = df.drop(columns=['case_identifier']).median().astype(int)
df_summary['Observation'] = df.drop(columns=['case_identifier']).notna().sum()
df_summary['Description'] = ''
df_summary.to_latex('result/summary_stat.tex',index=True,caption='Summary of the observed variables.\label{tab:sumstat}')


if FillType<3:
    '''
    Imputation based on the previous implementation. Not maintained.
    '''
    df=df.fillna(0)
    Imputed_Features=['Supervision/review of work',
           'Ability to hire employees',
           'Ownership of tools', 'Chance of profit', 'Risk of loss',
           'Exclusivity of services', 'Who sets the work hours',
           'Is the worker required to wear a uniform?',
           'Where the work is performed',
           'Delegation of tasks']
    All_Features=['Year', 'Industry', 'Length of service', 'Supervision/review of work',
           'Ability to hire employees',
           'Ownership of tools', 'Chance of profit', 'Risk of loss',
           'Exclusivity of services', 'Who sets the work hours',
           'Is the worker required to wear a uniform?',
           'Where the work is performed',
           'Delegation of tasks']
    df_filled=fill_database(df,Imputed_Features,All_Features,FillType)
    df_filled['Length of service'] = df_filled['Length of service'].apply(process_length)
    df_filled['Supervision/review of work'] = df_filled['Supervision/review of work'].apply(np.round)
    df_filled_relevant = df_filled[RelevantFeatures]
    if IndustryDummy:
        df_with_dummy = pd.concat( (df_filled_relevant.drop(columns=['Industry']), pd.get_dummies(df_filled_relevant, columns=['Industry']), df['Outcome']),axis=1 )
        df_final = df_with_dummy[df_with_dummy.Outcome.isin([1, 2])]
    else:
        df_final = pd.concat( (df_filled_relevant, df['Outcome']),axis=1 )
    df_final.to_csv('data/processed_data.csv',index=False)
    X = df_final.drop(columns=['Outcome'])
    y = df_final['Outcome']

elif FillType==3:
    '''
    Imputation newly implemented. Uses iterative imputer.
    '''
    # df_temp =  pd.concat( (pd.get_dummies(df[CategoricalFeatures].astype(object),dummy_na=True,drop_first=True),df[ContinuousFeatures]),axis=1 )
    ##Drop entries with too many missings
    if drop_if_missing!=False:
        df_ = df.dropna(thresh=drop_if_missing)
        print('Dropped {} samples due to missing values.'.format( len(df)-len(df_) ))
        df = df_
    df = impute_continuous(df, CategoricalFeatures, ContinuousFeatures)





    # imp = IterativeImputer(max_iter=5, random_state=0)
    # df_temp =  pd.concat( (pd.get_dummies(df[CategoricalFeatures].fillna(-1).astype(int).astype(object),drop_first=False),df[ContinuousFeatures]),axis=1 )
    # for var in CategoricalFeatures:
    #     df_temp = df_temp.drop(columns=[var+'_-1'] )
    # filled_array = imp.fit_transform(df_temp)
    # df_filled = pd.DataFrame(data=filled_array, columns=df_temp.keys(), index=df_temp.index)
    # df = pd.concat( (df_filled, df['Outcome']),axis=1 )

    df.to_csv('data/processed_data_fill{}_dropmissing_{}.csv'.format(FillType,drop_if_missing),index=False)
    X = df.drop(columns=['Outcome','case_identifier'])#pd.concat( (pd.get_dummies(df[CategoricalFeatures]),df[ContinuousFeatures]),axis=1 )#df[RelevantFeatures]
    y = df['Outcome']

assert df.isna().sum().sum()==0

##Scale the X variables.
if ScaleX:
    scaler = StandardScaler()
    scaler.fit(X)
    with open('data/scaler_fill{}_dropmissing_{}.pickle'.format(FillType,drop_if_missing), 'wb') as f:
        pickle.dump(scaler,f)
    X_scaled = pd.DataFrame(data = scaler.transform(X), columns = X.keys(), index=X.index)
else:
    X_scaled = X

##We set (0,1) is for (Employee, Contractor).
y = y-1

##Save the split data.
X_train, X_val, Y_train, Y_val = train_test_split(X_scaled, y)
X_scaled.to_csv('data/X_scaled_fill{}_dropmissing_{}.csv'.format(FillType,drop_if_missing))
X_train.to_csv('data/X_train_fill{}_dropmissing_{}.csv'.format(FillType,drop_if_missing))
X_val.to_csv('data/X_val_fill{}_dropmissing_{}.csv'.format(FillType,drop_if_missing))
y.to_csv('data/y_fill{}_dropmissing_{}.csv'.format(FillType,drop_if_missing))
Y_train.to_csv('data/Y_train_fill{}_dropmissing_{}.csv'.format(FillType,drop_if_missing))
Y_val.to_csv('data/Y_val_fill{}_dropmissing_{}.csv'.format(FillType,drop_if_missing))
