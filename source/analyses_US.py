import pandas as pd
import numpy  as np
import pickle
from joblib import dump, load

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_val_predict, KFold
from sklearn.metrics import accuracy_score
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
np.random.seed(0)

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from sklearn.ensemble import  RandomForestClassifier

import matplotlib.pyplot as plt

df_merged = pd.read_csv('data/US_Canada_merged.csv',index_col=0).dropna(thresh=3)
df_merged = df_merged[df_merged.Outcome.isin([1,2])]
CategoricalFeatures = [#'Industry',#'Type of Job',#'Occupation',#'Province',
       'Supervision/review of work',
       'Delegation of tasks',
       'Where the work is performed',
       'Is the worker required to wear a uniform?']
ContinuousFeatures = ['Length of service', 'Ownership of tools', 'Ability to hire employees', 'Chance of profit','Risk of loss','Exclusivity of services','Who sets the work hours']
df_temp =  pd.concat( (pd.get_dummies(df_merged[CategoricalFeatures].fillna(-1).astype(int).astype(object),drop_first=False),df_merged[ContinuousFeatures]),axis=1 )
for var in CategoricalFeatures:
    df_temp = df_temp.drop(columns=[var+'_-1'] )
df = pd.concat( (df_temp, df_merged[['source','Outcome']]),axis=1)


'''
1. Impute Canadian and US data separately. Allowing 2 columns to be missing.
'''
df_Canada = df[df['source']=='Canada'].drop(columns=['source'])
df_US = df[df['source']=='US'].drop(columns=['source'])

imp = IterativeImputer(max_iter=5, random_state=0)
filled_array_Canada = imp.fit_transform(df_Canada)
filled_array_US = imp.fit_transform(df_US)
df_filled_Canada = pd.DataFrame(data=filled_array_Canada, columns=df_Canada.keys(), index=df_Canada.index)
df_filled_US = pd.DataFrame(data=filled_array_US, columns=df_US.keys(), index=df_US.index)

'''
2. Scale the data based on Canadian data
'''
X_Canada = df_filled_Canada.drop(columns = ['Outcome'])
y_Canada = df_filled_Canada['Outcome']
X_US = df_filled_US.drop(columns = ['Outcome'])
y_US = df_filled_US['Outcome']
scaler = StandardScaler()
scaler.fit(X_Canada)
X_Canada_scaled = pd.DataFrame(data = scaler.transform(X_Canada), columns = X_Canada.keys(), index=X_Canada.index)
X_US_scaled = pd.DataFrame(data = scaler.transform(X_US), columns = X_US.keys(), index=X_US.index)

'''
3. Train a model on Canadian data, predict US data.
'''
rf_model= RandomForestClassifier()#(n_estimators=15, n_jobs=-1, max_depth = 6, oob_score=True)
rf_model.fit(X_Canada_scaled.values, y_Canada.values.flatten())
score = accuracy_score(rf_model.predict(X_US_scaled), y_US)
print('US accuracy based on Canadian data: {}'.format(score))

'''
4. Train a model on US only
'''
rf_model= RandomForestClassifier()#(n_estimators=15, n_jobs=-1, max_depth = 6, oob_score=True)
scores = cross_val_score(rf_model, X_US_scaled, y_US, cv=3)
print('US CV score based on US data: {}'.format(scores.mean()))

'''
5. Jointly train a model.
'''
class joint_model():
    def __init__(self, model, Canadian_X, Canadian_y):
        self.model = model
        self.Canadian_X =Canadian_X
        self.Canadian_y = Canadian_y
    def get_params(self, deep=True):
        return {"model": self.model,"Canadian_X": self.Canadian_X,"Canadian_y": self.Canadian_y}
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    def fit(self, US_X,US_y):
        X = np.concatenate( (self.Canadian_X, US_X), axis=0 )
        y = np.concatenate( (self.Canadian_y, US_y), axis=0 ).flatten()
        self.model.fit(X, y)
    def predict(self, X):
        return self.model.predict(X)
    def score(self, X, y, sample_weight=None):
        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)


rf_model= RandomForestClassifier()#(n_estimators=15, n_jobs=-1, max_depth = 6, oob_score=True)
model_mixed = joint_model(rf_model, X_Canada_scaled, y_Canada)
scores_mixed = cross_val_score(model_mixed, X_US_scaled, y_US, cv=3)


model_mixed2 = joint_model(rf_model, X_US_scaled, y_US)
scores_mixed2 = cross_val_score(model_mixed2, X_Canada_scaled, y_Canada, cv=3)
