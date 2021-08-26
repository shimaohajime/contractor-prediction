'''
Code to prepare the Canadian contractor-employee dataset for the predictive analysis.
This code should be run before predictions.py.
'''

import pandas as pd
import numpy  as np
import pickle
from source.utils import get_y_from_string, process_length, fill_database
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
ScaleX = True #If True, apply standard scaler to the data.


##Clean and keep only the relevant columns.
col = list(df_raw.columns[2:-2])
df = df_raw.copy(deep=True).drop(columns=['Length of service']).rename(columns={'NEW LENGTH OF SERVICE':'Length of service'}).replace({0.:np.nan})
##Note: 'NEW LENGTH OF SERVICE' is the cleaned data for service length. The original 'Length of service' contains human notes and should not be used.
df=df.replace('nan',np.nan)
df=df.replace('\xa0', 0)
df=df[~np.isnan(df.Outcome)]

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

df = df[CategoricalFeatures+ContinuousFeatures+['Outcome']]
df = df[df.Outcome.isin([1, 2])]
df = df.astype(float)

df.to_csv('data/unprocessed_data.csv')


'''
Summary statistics before imputation
'''
df_summary = pd.DataFrame(index=df.keys())
df_summary['Type'] = ['Categorical']*len(CategoricalFeatures) + ['Continuous']*len(ContinuousFeatures) + ['Categorical']
df_summary['Min'] = df.min().astype(int)
df_summary['Max'] = df.max().astype(int)
df_summary['Median'] = df.median().astype(int)
df_summary['Observation'] = df.notna().sum()
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
    df_temp =  pd.concat( (pd.get_dummies(df[CategoricalFeatures].fillna(-1).astype(int).astype(object),drop_first=False),df[ContinuousFeatures]),axis=1 )
    for var in CategoricalFeatures:
        df_temp = df_temp.drop(columns=[var+'_-1'] )
    imp = IterativeImputer(max_iter=5, random_state=0)
    filled_array = imp.fit_transform(df_temp)
    df_filled = pd.DataFrame(data=filled_array, columns=df_temp.keys(), index=df_temp.index)
    df = pd.concat( (df_filled, df['Outcome']),axis=1 )

    df.to_csv('data/processed_data.csv',index=False)
    X = df.drop(columns=['Outcome'])#pd.concat( (pd.get_dummies(df[CategoricalFeatures]),df[ContinuousFeatures]),axis=1 )#df[RelevantFeatures]
    y = df['Outcome']

assert df.isna().sum().sum()==0

##Scale the X variables.
if ScaleX:
    scaler = StandardScaler()
    scaler.fit(X)
    with open('data/scaler_fill{}.pickle'.format(FillType), 'wb') as f:
        pickle.dump(scaler,f)
    X_scaled = pd.DataFrame(data = scaler.transform(X), columns = X.keys(), index=X.index)
else:
    X_scaled = X

##We set (0,1) is for (Employee, Contractor).
y = y-1

##Save the split data.
X_train, X_val, Y_train, Y_val = train_test_split(X_scaled, y)
X_scaled.to_csv('data/X_scaled_fill{}.csv'.format(FillType))
X_train.to_csv('data/X_train_fill{}.csv'.format(FillType))
X_val.to_csv('data/X_val_fill{}.csv'.format(FillType))
y.to_csv('data/y_fill{}.csv'.format(FillType))
Y_train.to_csv('data/Y_train_fill{}.csv'.format(FillType))
Y_val.to_csv('data/Y_val_fill{}.csv'.format(FillType))
