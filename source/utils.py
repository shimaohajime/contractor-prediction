import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

np.random.seed(0)

# Correcting some typos on the database
def get_y_from_string(s):
    res = None
    letters = ['y','Y', 'm', 'M']
    for i in range(len(s)):
        #print(s[i])
        if s[i] in letters[:2]:
            #print(i)
            try:
                try:
                    y = float(s[i-2:i])
                except:
                    y = float(s[i-1])
            except:
                y = float(s[i-2])

            if res is None:
                res = y
            else:
                res +=y

        if s[i] in letters[2:]:
            try:
                try:
                    y = float(s[i-2:i])/12
                except:
                    y = float(s[i-1])/12
            except:
                y = float(s[i-2])/12

            if res is None:
                res = y
            else:
                res +=y
    return res

def process_length(x):
    if isinstance(x, str):
        res = get_y_from_string(x)
        return res
    return x

def fill_database(df,Imputed_Features,All_Features,method):

    df_imputed=df.copy()
    df_imputed=df_imputed.fillna(0)

    df2=df.copy()
    df2=df2.fillna(0)
    df2=df2[All_Features]

    if method==0:
        1+1
    elif method==1:
        for featnum in range(len(Imputed_Features)):
            Feature=Imputed_Features[featnum]
            print('** Imputing '+Feature+'... **')
            DD=df2[df2[Feature]>0]
            med_val=DD[Feature].median()
            print(med_val)
            df_imputed[df2[Feature]==0]=med_val
    else:
        for featnum in range(len(Imputed_Features)):
            Feature=Imputed_Features[featnum]
            print('** Imputing '+Feature+'... **')


            df3=df2[df2[Feature]>0.]
            y_del=df3[Feature]
            df3=df3.drop(Feature,axis=1)
            X_train, X_val, Y_train, Y_val = train_test_split(df3, y_del)
            print(f'Train size = {len(X_train)}')
            print(f'Test size = {len(X_val)}')

            m = RandomForestClassifier(n_estimators=5, n_jobs=-1, max_depth = 3, max_features=12)
            m.fit(X_train, Y_train)

            X=df.copy()
            X=X[All_Features]
            X_impute=X[X[Feature]==0]
            X_impute=X_impute.drop(Feature,axis=1)
            df_imputed.loc[df_imputed[Feature]==0,Feature]=m.predict(X_impute)
    return df_imputed


def print_score(m,d=1):
    if d==1:
        print(f'Accuracy train : {m.score(X_train, Y_train)}')
        print(f'Accuracy val : {m.score(X_val, Y_val)}')
        if hasattr(m, 'oob_score_'): print(f'Oob score : {m.oob_score_}')
    else:
        print(f'Accuracy train : {m.score(X_train[d], Y_train)}')
        print(f'Accuracy val : {m.score(X_val[d], Y_val)}')
        if hasattr(m, 'oob_score_'): print(f'Oob score : {m.oob_score_}')
