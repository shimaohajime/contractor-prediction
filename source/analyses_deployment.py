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

cols = [ 'datetime','years_of_service', 'industry', 'was_supervised','could_hire','who_assigned_tasks','supplied_equipment','how_was_paid','risk',\
'exclusivity_of_services','set_work_hours','where_to_work','dress_restrictions','entry_id','workplace_location','age','gender','immigration_status','level_of_education','past_or_present','user_type']

df_deployed_raw = pd.read_csv('data/WorkerClassification20210622.csv',header=None,skiprows=[0], names=cols)

##Drop the first and last two months
#df_deployed_raw = df_deployed_raw[(df_deployed_raw.datetime>='2020-11-01') & (df_deployed_raw.datetime<'2021-05-01')]
#df_deployed_raw = df_deployed_raw[(df_deployed_raw.datetime<'2020-11-01') | (df_deployed_raw.datetime<'2021-05-01')]

for col in ['industry', 'was_supervised','could_hire', 'who_assigned_tasks', 'supplied_equipment','how_was_paid', 'risk', 'exclusivity_of_services', 'set_work_hours','where_to_work', 'dress_restrictions', 'workplace_location']:
    df_deployed_raw[col] = df_deployed_raw[col].replace(0,np.nan)

'''
deployment data contains 0 but not None. Need to know the matching.
'''

df_canadian_cases = pd.read_csv('data/unprocessed_data.csv',index_col=0)

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

df_raw = pd.concat((df_deployed_raw,df_canadian_cases.rename(columns=col_match)),axis=0)[list(col_match.values())+['Outcome']]
df_temp =  pd.concat( (pd.get_dummies(df_raw[CategoricalFeatures].fillna(-1).astype(int).astype(object),drop_first=False),df_raw[ContinuousFeatures+['Outcome']]),axis=1 )
for var in CategoricalFeatures:
    try:
        df_temp = df_temp.drop(columns=[var+'_-1'] )
    except:
        pass
df = df_temp.copy(deep=True)
df['source'] = np.array([1]*len(df_deployed_raw) +[0]*len(df_canadian_cases) )



'''
1. Predict if a sample is from the app or the court
'''
df_pred_source = df.drop(columns=['Outcome']).dropna()
X = df_pred_source.drop(columns=['source'])
y = df_pred_source['source'].values.flatten()

scaler = StandardScaler()
scaler.fit(X)
X_scaled = pd.DataFrame(data = scaler.transform(X), columns = X.keys(), index=X.index)

rf_model= RandomForestClassifier()#(n_estimators=15, n_jobs=-1, max_depth = 6, oob_score=True)
scores = cross_val_score(rf_model, X_scaled, y, cv=3)
out_file = open('result/cv_predict_deployment.txt', 'w')
print('CV score - predict samples from deployment: {}'.format(scores.mean() ))
print('CV score - predict samples from deployment: {}'.format(scores.mean() ), file=out_file)


'''
2. Prediction of the app samples.
'''
##Prepare data
df_c = df[df.source==0].dropna(thresh=3)
df_d = df[df.source==1].dropna(thresh=3)
imp = IterativeImputer(max_iter=5, random_state=0)

filled_array_c = imp.fit_transform(df_c.drop(columns=['Outcome','source']))
df_c_filled = pd.DataFrame(data=filled_array_c, columns=df_c.drop(columns=['Outcome','source']).keys(), index=df_c.index)
df_c_filled['Outcome'] = df_c['Outcome']

filled_array_d = imp.fit_transform(df_d.drop(columns=['Outcome','source']))
df_d_filled = pd.DataFrame(data=filled_array_d, columns=df_d.drop(columns=['Outcome','source']).keys(), index=df_d.index)
df_d_filled['Outcome'] = df_d['Outcome']

##Train model with court samples
model = RandomForestClassifier()
X = df_c_filled.drop(columns=['Outcome'])
y = df_c_filled['Outcome']
scaler = StandardScaler()
scaler.fit(X)
X_scaled = pd.DataFrame(data = scaler.transform(X), columns = X.keys(), index=X.index)

X_dep = df_d_filled.drop(columns=['Outcome'])
X_dep_scaled = pd.DataFrame(data = scaler.transform(X_dep), columns = X_dep.keys(), index=X_dep.index)

kf = KFold(n_splits=3)
pred_proba_c = pd.DataFrame(data={'confidence':np.nan}, index=X_scaled.index )
pred_proba_d = pd.DataFrame(data={'confidence':0.},  index=X_dep_scaled.index )
for i, (train, test) in enumerate(kf.split(X_scaled) ):
    X_train = X_scaled.iloc[train]
    X_test = X_scaled.iloc[test]
    y_train = y.iloc[train]
    y_test = y.iloc[test]

    model.fit(X_train.values, y_train.values.flatten())
    pred_proba_c.loc[:,'confidence'].iloc[test] = np.max( (model.predict_proba(X_test)[:,0], model.predict_proba(X_test)[:,1] ),axis=0)
    pred_proba_d.loc[:,'confidence'] = pred_proba_d.loc[:,'confidence'] + np.max( (model.predict_proba(X_dep_scaled)[:,0], model.predict_proba(X_dep_scaled)[:,1] ),axis=0)
pred_proba_d.loc[:,'confidence'] = pred_proba_d.loc[:,'confidence']/3

fig,ax = plt.subplots(1,1)
ax.hist([pred_proba_c.values.flatten(),pred_proba_d.values.flatten()],label=['court','deployment'],weights=[ np.ones(len(pred_proba_c)) / len(pred_proba_c),np.ones(len(pred_proba_d)) / len(pred_proba_d) ],bins=10,alpha=.8,color=['navy','darkred'])
ax.legend()
fig.savefig('result/6-deployment-confidence-hist.png')

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

pred_proba_df = df[['source']] #pd.DataFrame( index=df_filled[df_filled['source']==1].index )
X  =  scaler.transform(df.drop(columns=['source']))
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

fig,ax = plt.subplots(1,1)
conf_court = pred_proba_df[pred_proba_df['source']==0]['RandomForestClassifier_confidence']
conf_app = pred_proba_df[pred_proba_df['source']==1]['RandomForestClassifier_confidence']
ax.hist([conf_court,conf_app],label=['training','deployment'],weights=[ np.ones(len(conf_court)) / len(conf_court),np.ones(len(conf_app)) / len(conf_app) ],bins=10,alpha=.8,color=['navy','darkred'])
ax.legend()
fig.savefig('result/6-deployment-confidence-hist.png')

# conf_court = pred_proba_df[pred_proba_df['source']==0]['RandomForestClassifier_confidence']
# conf_app = pred_proba_df[pred_proba_df['source']==1]['RandomForestClassifier_confidence']
# ax.hist(conf_app,weights=np.ones(len(conf_app)) / len(conf_app),bins=10,alpha=.8,color='darkred')
# fig.savefig('result/6-deployment-confidence-hist.png')
'''


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
