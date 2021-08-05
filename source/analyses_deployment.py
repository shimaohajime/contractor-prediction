import pandas as pd
import numpy  as np
import pickle
from joblib import dump, load

from source.utils import get_y_from_string, process_length, fill_database
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_val_predict, KFold
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
np.random.seed(0)

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from sklearn.ensemble import  RandomForestClassifier

import matplotlib.pyplot as plt

col = [ 'datetime','years_of_service', 'industry', 'was_supervised','could_hire','who_assigned_tasks','supplied_equipment','how_was_paid','risk',\
'exclusivity_of_services','set_work_hours','where_to_work','dress_restrictions','entry_id','workplace_location','age','gender','immigration_status','level_of_education','past_or_present','user_type']

df_deployed_raw = pd.read_csv('data/WorkerClassification20210622.csv',header=None,skiprows=[0], names=col)
df_canadian_cases = pd.read_csv('data/unprocessed_data.csv')

col_match ={'Industry':'industry',
'Length of service':'years_of_service',
'Supervision/review of work':'was_supervised',
'Ability to hire employees':'could_hire',
'Delegation of tasks':'who_assigned_tasks',
'Ownership of tools':'supplied_equipment',
'Chance of profit':'how_was_paid',
'Risk of loss':'risk',
'Exclusivity of services':'exclusivity_of_services',
'Who sets the work hours':'set_work_hours',
'Where the work is performed':'where_to_work',
'Is the worker required to wear a uniform?':'dress_restrictions'}


CategoricalFeatures = ['industry',#'Type of Job',#'Occupation',#'Province',
'was_supervised',
'who_assigned_tasks',
'where_to_work',
'dress_restrictions']
ContinuousFeatures = ['years_of_service',
'supplied_equipment',
'could_hire',
'how_was_paid',
'risk',
'exclusivity_of_services',
'set_work_hours']


'''
1. Predict if a sample is from the app or the court
'''
df = pd.concat((df_deployed_raw,df_canadian_cases.rename(columns=col_match)),axis=0)[list(col_match.values())]

df_temp =  pd.concat( (pd.get_dummies(df[CategoricalFeatures].fillna(-1).astype(int).astype(object),drop_first=False),df[ContinuousFeatures]),axis=1 )

for var in CategoricalFeatures:
    df_temp = df_temp.drop(columns=[var+'_-1'] )
# imp = IterativeImputer(max_iter=5, random_state=0)
# filled_array = imp.fit_transform(df_temp)
# df_filled = pd.DataFrame(data=filled_array, columns=df_temp.keys(), index=df_temp.index)
df_temp['source'] = np.array([1]*len(df_deployed_raw) +[0]*len(df_canadian_cases) )
df_filled = df_temp.dropna()

X = df_filled.drop(columns=['source'])
y = df_filled['source'].values.flatten()

scaler = StandardScaler()
scaler.fit(X)
X_scaled = pd.DataFrame(data = scaler.transform(X), columns = X.keys(), index=X.index)

rf_model= RandomForestClassifier()#(n_estimators=15, n_jobs=-1, max_depth = 6, oob_score=True)
scores = cross_val_score(rf_model, X_scaled, y, cv=3)
print('CV score - predict samples from deployment: ',scores.mean())


'''
2. Prediction of the app samples.
'''
with open('data/scaler_fill{}.pickle'.format(3), 'rb') as f:
    scaler = pickle.load(f)

models = [
'LogisticRegression',
'RandomForestClassifier',
'KNeighborsClassifier',
'SVC',
'GaussianProcessClassifier',
'AdaBoostClassifier',
'XGBClassifier'
]

# X_app = scaler.transform(df_filled[df_filled['source']==1].drop(columns=['source']))
# X_court =  scaler.transform(df_filled[df_filled['source']==0].drop(columns=['source']))

pred_proba_df = df_filled[['source']] #pd.DataFrame( index=df_filled[df_filled['source']==1].index )
X  =  scaler.transform(df_filled.drop(columns=['source']))
for model_name in models:
    pred_proba_df[model_name+ '_pred'] = -1
    pred_proba_df[model_name+ '_proba0'] = -1
    pred_proba_df[model_name+ '_proba1'] = -1
    pred_proba_df[model_name+ '_confidence'] = -1

for model_name in models:
    model = load('result/{}.joblib'.format(model_name))
    model.predict(X)
    pred_proba_df.loc[:,model_name + '_pred'] = model.predict(X)
    pred_proba_df.loc[:,model_name+ '_proba0'] = model.predict_proba(X)[:,0]
    pred_proba_df.loc[:,model_name+ '_proba1'] = model.predict_proba(X)[:,1]
    pred_proba_df.loc[:,model_name+ '_confidence'] = pred_proba_df[[model_name+ '_proba0',model_name+ '_proba1']].max(axis=1)

# pred_proba_df.to_csv('result/prediction_app_sample.csv')

#pred_proba_df[pred_proba_df['source']==0]['']
fig,ax = plt.subplots(1,1)
conf_court = pred_proba_df[pred_proba_df['source']==0]['RandomForestClassifier_confidence']
conf_app = pred_proba_df[pred_proba_df['source']==1]['RandomForestClassifier_confidence']
ax.hist([conf_court,conf_app],label=['training','deployment'],weights=[ np.ones(len(conf_court)) / len(conf_court),np.ones(len(conf_app)) / len(conf_app) ],bins=10,alpha=.8,color=['navy','darkred'])
ax.legend()
fig.savefig('result/6-deployment-confidence-hist.png')

'''
3. Distribution of samples
'''
X  =  scaler.transform(df_filled.drop(columns=['source']))
emb_method = PCA(n_components=2)#TSNE(n_components=2)
pca = emb_method.fit(X[df_filled.reset_index()['source']==0])
X_embedded = pd.DataFrame( pca.transform(X))

fig,ax = plt.subplots(1,1)
emb_court = X_embedded[df_filled.reset_index()['source']==0]
emb_app = X_embedded[df_filled.reset_index()['source']==1]
ax.scatter(emb_court.loc[:,0],emb_court.loc[:,1],c='navy',alpha=.4,s=25,label='training')
ax.scatter(emb_app.loc[:,0],emb_app.loc[:,1],c='darkred',alpha=.4,s=25,label='deployment')
ax.legend(prop={'size': 6})
fig.savefig('result/6-deployment-pca.png')
