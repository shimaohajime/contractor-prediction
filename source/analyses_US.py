import sys
import copy
import pandas as pd
import numpy  as np
import pickle
from joblib import dump, load

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_val_predict, KFold
from sklearn.metrics import accuracy_score
from source.utils import fill_database, impute_continuous
from scipy.stats import ttest_ind

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from sklearn.ensemble import  RandomForestClassifier

import shap

import matplotlib.pyplot as plt

np.random.seed(0)

drop_if_missing = False #If integer, drop the samples with missing variables more than this number.


df_merged = pd.read_csv('data/US_Canada_merged.csv',index_col=0)
if drop_if_missing!=False:
    df_merged = df_merged.dropna(thresh=drop_if_missing)


df_merged = df_merged[df_merged.Outcome.isin([1,2])]
CategoricalFeatures = [#'Industry',#'Type of Job',#'Occupation',#'Province',
       'Supervision/review of work',
       'Delegation of tasks',
       'Where the work is performed',
       'Is the worker required to wear a uniform?']
ContinuousFeatures = ['Length of service', 'Ownership of tools', 'Ability to hire employees', 'Chance of profit','Risk of loss','Exclusivity of services','Who sets the work hours']
# df_temp =  pd.concat( (pd.get_dummies(df_merged[CategoricalFeatures].fillna(-1).astype(int).astype(object),drop_first=False),df_merged[ContinuousFeatures]),axis=1 )
# for var in CategoricalFeatures:
#     df_temp = df_temp.drop(columns=[var+'_-1'] )
# df = pd.concat( (df_temp, df_merged[['source','Outcome','case_identifier']]),axis=1)
##Make sure Canadian cases come first
df = df_merged
assert pd.Index(df['source']).is_monotonic
assert df['source'].iloc[0]=='Canada'

'''
0. Summary stat of US data before imputation
'''
df_US_only = df_merged[df_merged.source=='US'].drop(columns=['source', 'Type of Job','case_identifier'])

df_summary = pd.DataFrame(index=df_US_only.keys())
df_summary['Type'] = ['Categorical']*len(CategoricalFeatures) + ['Continuous']*len(ContinuousFeatures) + ['Categorical']
df_summary['Min'] = df_US_only.min().astype(int)
df_summary['Max'] = df_US_only.max().astype(int)
df_summary['Median'] = df_US_only.median().astype(int)
df_summary['Observation'] = df_US_only.notna().sum()
df_summary['Description'] = ''
df_summary.to_latex('result/summary_stat_us.tex',index=True,caption='Summary of the observed variables in US cases.\label{tab:sumstat_us}')


'''
1. Impute Canadian and US data separately.
'''
assert(df.Outcome.isna().sum()==0)
df_Canada = df[df['source']=='Canada'].drop(columns=['source'])
df_US = df[df['source']=='US'].drop(columns=['source'])

df_Canada_ = df_Canada.drop(columns=['case_identifier'])
df_US_ = df_US.drop(columns=['case_identifier'])

# imp = IterativeImputer(max_iter=5, random_state=0)
# filled_array_Canada = imp.fit_transform(df_Canada_)
# filled_array_US = imp.fit_transform(df_US_)
# df_filled_Canada = pd.DataFrame(data=filled_array_Canada, columns=df_Canada_.keys(), index=df_Canada_.index)
# df_filled_US = pd.DataFrame(data=filled_array_US, columns=df_US_.keys(), index=df_US_.index)

df_filled_Canada = impute_continuous(df_Canada_, CategoricalFeatures, ContinuousFeatures)
df_filled_US = impute_continuous(df_US_, CategoricalFeatures, ContinuousFeatures)
# df_filled_Canada.to_csv('data/Canadian_data_filled.csv')

##Compare the mean
pval_df = pd.DataFrame(df_filled_Canada.mean(), index = df_filled_Canada.keys(), columns=['Canada'])
pval_df['US'] = df_filled_US.mean()
pval_df['p-value'] = ttest_ind(df_filled_Canada, df_filled_US)[1]
pval_df.to_csv('result/Canada_US_mean_difference.csv', float_format='%.4f')


'''
2. No scaling as it turned out to be better.
'''
X_Canada = df_filled_Canada.drop(columns = ['Outcome'])
y_Canada = df_filled_Canada['Outcome']
X_US = df_filled_US.drop(columns = ['Outcome'])
y_US = df_filled_US['Outcome']

# scaler = StandardScaler()
# scaler.fit(X_Canada)
# X_Canada_scaled = pd.DataFrame(data = scaler.transform(X_Canada), columns = X_Canada.keys(), index=X_Canada.index)
# X_US_scaled = pd.DataFrame(data = scaler.transform(X_US), columns = X_US.keys(), index=X_US.index)
X_Canada_scaled = X_Canada
X_US_scaled = X_US

X_scaled = pd.concat( (X_Canada_scaled,X_US_scaled),axis=0 )

'''
3. Train a model on Canada only.
'''
##CV score within Canada
rf_model= RandomForestClassifier()#(n_estimators=15, n_jobs=-1, max_depth = 6, oob_score=True)
scores = cross_val_score(rf_model, X_Canada_scaled, y_Canada, cv=3)
print('Canada CV score based on Canada data: {}'.format(scores.mean()))
with open('result/US_result.txt', 'w') as f:
    f.write('Canada CV score based on Canada data: {}\n'.format(scores.mean()))

##CV score within Canada, with sample size matched to US
scores = []
for i in range(10):
    idx  = np.random.randint(0,len(X_Canada_scaled),len(X_US_scaled)  )
    rf_model= RandomForestClassifier()#(n_estimators=15, n_jobs=-1, max_depth = 6, oob_score=True)
    scores.append(cross_val_score(rf_model, X_Canada_scaled.iloc[idx], y_Canada.iloc[idx], cv=3) )
print('Canada CV score based on Canada data with sample size matched to US: {}'.format(np.array(scores).mean()))
with open('result/US_result.txt', 'a') as f:
    f.write('Canada CV score based on Canada data with sample size matched to US: {}\n'.format(np.array(scores).mean()))
scores_mean = np.array(scores).mean()

##CV score on US data
##rf_model_canada  is trained on full-sample of Canadian data
rf_model_canada= RandomForestClassifier()#(n_estimators=15, n_jobs=-1, max_depth = 6, oob_score=True)
rf_model_canada.fit(X_Canada_scaled.values, y_Canada.values.flatten())
score = accuracy_score(rf_model_canada.predict(X_US_scaled), y_US)
print('US accuracy based on Canadian data: {}'.format(score))
with open('result/US_result.txt', 'a') as f:
    f.write('US accuracy based on Canadian data: {}\n'.format(score))

'''
4. Train a model on US only
'''
##CV score within US
rf_model= RandomForestClassifier()#(n_estimators=15, n_jobs=-1, max_depth = 6, oob_score=True)
scores = cross_val_score(rf_model, X_US_scaled, y_US, cv=3)
print('US CV score based on US data: {}'.format(scores.mean()))
with open('result/US_result.txt', 'a') as f:
    f.write('US CV score based on US data: {}\n'.format(scores.mean()))

##CV score on Canadian data
##rf_model_US is trained on full-sample of US data
rf_model_US= RandomForestClassifier()
rf_model_US.fit(X_US_scaled.values, y_US.values.flatten())
score = accuracy_score(rf_model_US.predict(X_Canada), y_Canada)
print('Canada accuracy based on US data: {}'.format(score))
with open('result/US_result.txt', 'a') as f:
    f.write('Canada accuracy based on US data: {}\n'.format(score))


'''
5. Difference in prediction
#df_ = copy.deepcopy(df_filled_Canada)
df_ = copy.deepcopy(df)
df_['Prediction_from_Canada_model'] = rf_model_canada.predict(X_scaled.values)
df_['Prediction_from_US_model'] = rf_model_US.predict(X_scaled.values)

df_diff = df_[df_['Prediction_from_Canada_model']!=df_['Prediction_from_US_model']].drop(columns=['Prediction_from_Canada_model', 'Prediction_from_US_model', 'case_identifier', 'source'])
df_same = df_[df_['Prediction_from_Canada_model']==df_['Prediction_from_US_model']].drop(columns=['Prediction_from_Canada_model', 'Prediction_from_US_model', 'case_identifier', 'source'])

pval_df = pd.DataFrame(df_same.mean(), index = df_diff.keys(), columns=['Same'])
pval_df['Different'] = df_diff.mean()
pval_df['p-value'] = ttest_ind(df_diff, df_same, nan_policy='omit')[1]
pval_df.to_csv('result/US_Canada_different_pred_pvalue.csv', float_format='%.4f')

# labels = df_diff.keys()
# x = np.arange(len(labels))  # the label locations
# width = 0.35  # the width of the barsfig, ax = plt.subplots()
# fig, ax = plt.subplots()
# rects1 = ax.bar(x - width/2, df_diff.mean(), width, label='Different', yerr=df_diff.std()/len(df_diff))
# rects2 = ax.bar(x + width/2, df_same.mean(), width, label='Same', yerr=df_same.std()/len(df_same))
# ax.set_xticklabels(labels)

df_.to_csv('result/prediction_by_US_Canada_models.csv')
'''


'''
6. SHAP
'''
explainer_canada = shap.Explainer(rf_model_canada)
explainer_us = shap.Explainer(rf_model_US)

shap_values_canada = explainer_canada(X_scaled)
shap_values_us = explainer_us(X_scaled)

shap.plots.bar(shap_values_canada[:,:,0],show=False)
plt.gcf().set_size_inches(plt.gcf().get_size_inches()*np.array([2.9,1.7]))
plt.savefig('result/rf_shap_importance_canada.png')
plt.close()

shap.plots.bar(shap_values_us[:,:,0],show=False)
plt.gcf().set_size_inches(plt.gcf().get_size_inches()*np.array([2.9,1.7]))
plt.savefig('result/rf_shap_importance_us.png')
plt.close()



'''
7. Jointly train a model.
'''
class joint_model():
    def __init__(self, model, other_X, other_y):
        self.model = model
        self.other_X =other_X
        self.other_y = other_y
    def get_params(self, deep=True):
        return {"model": self.model,"other_X": self.other_X,"other_y": self.other_y}
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    def fit(self, main_X, main_y):
        X = np.concatenate( (self.other_X, main_X), axis=0 )
        y = np.concatenate( (self.other_y, main_y), axis=0 ).flatten()
        self.model.fit(X, y)
    def predict(self, X):
        return self.model.predict(X)
    def score(self, X, y, sample_weight=None):
        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)


rf_model= RandomForestClassifier()#(n_estimators=15, n_jobs=-1, max_depth = 6, oob_score=True)

##Combine all Canadian data with 2/3 of US data to predict the 1/3 of US data.
model_mixed = joint_model(rf_model, X_Canada_scaled, y_Canada)
scores_mixed = cross_val_score(model_mixed, X_US_scaled, y_US, cv=3)
print('CV score on US data with entire Canadian data added to training: {}'.format(scores_mixed.mean()))
with open('result/US_result.txt', 'a') as f:
    f.write('CV score on US data with entire Canadian data added to training: {}\n'.format(scores_mixed.mean()))

##Combine all US data with 2/3 of Canadian data to predict the 1/3 of Canadian data.
model_mixed2 = joint_model(rf_model, X_US_scaled, y_US)
scores_mixed2 = cross_val_score(model_mixed2, X_Canada_scaled, y_Canada, cv=3)
print('CV score on Canadian data with entire US data added to training: {}'.format(scores_mixed2.mean()))
with open('result/US_result.txt', 'a') as f:
    f.write('CV score on Canadian data with entire US data added to training: {}\n'.format(scores_mixed2.mean()))
