'''
Code to produce the main results of contractor-employee prediction.
One should run data_cleaning.py first to produce the necessary data.
'''

import sys
import pandas as pd
import numpy as np
import pickle
from joblib import dump, load
import copy

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import graphviz

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, cross_val_predict, KFold
from sklearn.cluster import KMeans, SpectralClustering, AffinityPropagation, MeanShift
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, silhouette_score, accuracy_score, f1_score, make_scorer, roc_auc_score

from sklearn import tree
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import AdaBoostClassifier
import xgboost as xgb

import shap
from fairlearn.reductions import ExponentiatedGradient, DemographicParity, EqualizedOdds
from sklearn.base import BaseEstimator

np.random.seed(0) #0

'''
0. Read data
'''
# FillType = 3
# X = pd.read_csv('data/X_scaled_fill{}.csv'.format(FillType),index_col=0)
# X_train = pd.read_csv('data/X_train_fill{}.csv'.format(FillType),index_col=0)
# X_val = pd.read_csv('data/X_val_fill{}.csv'.format(FillType),index_col=0)
# y = pd.read_csv('data/y_fill{}.csv'.format(FillType),index_col=0).astype(int)
# Y_train = pd.read_csv('data/Y_train_fill{}.csv'.format(FillType),index_col=0).astype(int)
# Y_val = pd.read_csv('data/Y_val_fill{}.csv'.format(FillType),index_col=0).astype(int)
# with open('data/scaler_fill{}.pickle'.format(FillType), 'rb') as f:
#     scaler = pickle.load(f)
'''
NOTE 1: We eventually used unscaled X since it lead to better accuracy.
NOTE 2: We use "Industry" variable only for fairness
'''
FillType = 3
drop_if_missing = False #If integer, drop the samples with missing variables more than this number.

#df_filled_Canada = pd.read_csv('data/Canadian_data_filled.csv',index_col=0)
df_filled_Canada = pd.read_csv('data/processed_data_fill{}_dropmissing_{}.csv'.format(FillType,drop_if_missing))

case_identifier = df_filled_Canada['case_identifier']
X = df_filled_Canada.drop(columns = ['Outcome','case_identifier']).drop(columns=list(df_filled_Canada.filter(regex='Industry')))
y = df_filled_Canada['Outcome']


'''
1. Clustering analysis.
'''
n_clusters = 2
for emb_method in [TSNE(n_components=2), PCA(n_components=2)]:
#emb_method = TSNE(n_components=2)#PCA(n_components=2)#
    X_embedded = pd.DataFrame( emb_method.fit_transform(X), index=X.index)
    clustering_model = KMeans(n_clusters=n_clusters, random_state=0)#SpectralClustering(n_clusters=n_clusters, random_state=0)  #
    clustering = clustering_model.fit(X)#
    label = clustering.labels_


    fig1,ax1 = plt.subplots(1,1)
    color_iter=iter(cm.rainbow(np.linspace(0,1,n_clusters)))
    for i in range(n_clusters):
        emb = X_embedded[label==i]
        color = next(color_iter)
        ax1.scatter(emb.loc[:,0],emb.loc[:,1],c=color,alpha=.4,s=25,label='cluster {}'.format(i))
    ax1.legend(prop={'size': 6})
    fig1.savefig('result/{}-{}-{}clusters_fill{}.png'.format(emb_method.__class__.__name__,clustering.__class__.__name__,n_clusters, FillType))
    plt.close()
    print('Adjusted Rand score: {}'.format(adjusted_rand_score(label,y.values.flatten()) ) )
    print('Silhouette score: {}'.format(silhouette_score(X, label) ) )

kmeans2 = KMeans(n_clusters=2)
kmeans3 = KMeans(n_clusters=3)
affinity_prop = AffinityPropagation()
mean_shift = MeanShift()
spectral2 = SpectralClustering(n_clusters=2)
spectral3 = SpectralClustering(n_clusters=3)
clustering_models = [kmeans2,kmeans3,affinity_prop,mean_shift,spectral2,spectral3]
ari_scores = []
for model in clustering_models:
    clustering = model.fit(X)#
    label = clustering.labels_
    ari = adjusted_rand_score(label,y.values.flatten())
    silhouette = silhouette_score(X, label)
    ari_scores.append( [model.__class__.__name__, ari,silhouette] )
ari_result_df = pd.DataFrame(ari_scores, columns=['model', 'adjusted rand index', 'silhouette score'])
ari_result_df.to_latex('result/clustering-ari.tex',index=False,float_format="%.3f",caption='Adjusted rand index between clusters and outcome and silhouette score. \label{tab:clustering-ari}')


'''
2. Predictive analysis
'''
lr_model = LogisticRegression()
rf_model= RandomForestClassifier()#(n_estimators=15, n_jobs=-1, max_depth = 6, oob_score=True)
kn_model = KNeighborsClassifier()#(n_neighbors=10)
svc_model =SVC(kernel='linear',probability=True)#(gamma=10.,kernel='linear')
gp_model = GaussianProcessClassifier(kernel=1. * RBF(1.),random_state=0)#GaussianProcessClassifier(kernel=1. * RBF(1.),random_state=0).fit(X_train, Y_train)
ab_model = AdaBoostClassifier()#AdaBoostClassifier(n_estimators=10)
xgb_model = xgb.XGBClassifier()#xgb.XGBClassifier(max_depth=4,reg_alpha=1,reg_lambda=10,learning_rate=0.1,n_estimators=100)

models = [lr_model, rf_model, kn_model, svc_model, gp_model, ab_model, xgb_model ]

'''
2.1 Simple out-of-sample-prediction

result_df = pd.DataFrame(scaler.inverse_transform(X_val), columns=X_val.keys(), index=X_val.index )
result_df['Outcome'] = Y_val
for model in models:
    model_name = model.__class__.__name__
    model.fit(X_train.values, Y_train.values.flatten())
    print('Model: {} Accuracy: {}'.format(model.__class__.__name__, model.score(X_val.values, Y_val.values.flatten())))
    result_df['Pred_{}'.format(model_name)] = model.predict(X_val)
    result_df['Correct_{}'.format(model_name)] = (result_df['Pred_{}'.format(model_name)]==result_df['Outcome'])

result_df.to_csv('result/sklearn-model-predictions_fill{}.csv'.format(FillType))

'''


'''
2.2 CV score
'''
print('------------CV--------------')
cv_result = []
cv_pred = pd.DataFrame(data = y,columns=['Outcome'], index = X.index)
for model in models:
    scores = cross_val_score(model, X, y.values.flatten(), cv=3)
    f1 = cross_val_score(model, X, y.values.flatten(), cv=3, scoring=make_scorer(f1_score))
    auc = cross_val_score(model, X, y.values.flatten(), cv=3, scoring=make_scorer(roc_auc_score))
    cv_pred[model.__class__.__name__] = cross_val_predict(model, X, y.values.flatten(), cv=3)
    cv_result.append([model.__class__.__name__, scores.mean(), f1.mean(), auc.mean()])
    print('Model: {} CV score: {}'.format(model.__class__.__name__, scores.mean()))
cv_df = pd.DataFrame(cv_result, columns = ['model', 'accuracy', 'F1', 'ROC AUC'])
cv_df.to_csv('result/sklearn-model-cvresult_fill{}.csv'.format(FillType))
cv_df.to_latex('result/sklearn-model-cvresult.tex',index=False,float_format="%.3f",caption='Predictive accuracy of supervised models. The accuracy is evaluated by average of 3-fold cross validation. \label{tab:cv-result}')

# pred_col = [col for col in result_df.keys() if col.startswith('Pred')]
# correct_col = [col for col in result_df.keys() if col.startswith('Correct')]

##Scatter plot with correct/mis coloring
mis_sample =  X[cv_pred.Outcome!=cv_pred.XGBClassifier].index
fp_sample =  X[(cv_pred.Outcome==1) & (cv_pred.XGBClassifier==0)].index
fn_sample =  X[(cv_pred.Outcome==0) & (cv_pred.XGBClassifier==1)].index
correct_sample = X[cv_pred.Outcome==cv_pred.XGBClassifier].index
for emb_method in [TSNE(n_components=2), PCA(n_components=2)]:
    #emb_method = TSNE(n_components=2)#PCA(n_components=2)#
    X_embedded = pd.DataFrame( emb_method.fit_transform(X), index=X.index)
    fig1,ax1 = plt.subplots(1,1)
    color_iter=iter(cm.rainbow(np.linspace(0,1,3)))
    color = next(color_iter)
    ax1.scatter(X_embedded.loc[correct_sample,0],X_embedded.loc[correct_sample,1],c=color,alpha=.4,s=25,label='TP, TN')
    # color = next(color_iter)
    # ax1.scatter(X_embedded.loc[mis_sample,0],X_embedded.loc[mis_sample,1],c=color,alpha=.4,s=25,label='Mis-classified')
    color = next(color_iter)
    ax1.scatter(X_embedded.loc[fp_sample,0],X_embedded.loc[fp_sample,1],c=color,alpha=.4,s=25,label='FP')
    color = next(color_iter)
    ax1.scatter(X_embedded.loc[fn_sample,0],X_embedded.loc[fn_sample,1],c=color,alpha=.4,s=25,label='FN')
    ax1.legend(prop={'size': 6})
    fig1.savefig('result/{}-missclassified_fill{}.png'.format(emb_method.__class__.__name__,FillType))
    plt.close()


##See how sample size matters by varying the CV folds.
cv_score_all = []
sample_size = []
for folds in [2,3,5,10]:
    cv_score = []
    for model in models:
        scores = cross_val_score(model, X, y.values.flatten(), cv=folds)
        cv_score.append( scores.mean())
    cv_score_all.append(cv_score)
    sample_size.append( int(len(X)/folds) )
cv_fold_df = pd.DataFrame(np.array(cv_score_all), columns = [model.__class__.__name__ for model in models])
cv_fold_df['folds'] = [2,3,5,10]
cv_fold_df['training sample'] = [ int(len(X)* (folds-1)/folds) for folds in [2,3,5,10] ]

cv_fold_df = cv_fold_df[list(cv_fold_df.keys())[-2:] + list(cv_fold_df.keys())[:-2]]
cv_fold_df = cv_fold_df.transpose()
cv_fold_df.to_latex('result/cv-vary-fold.tex',float_format="%.3f",header=False,caption='CV score when varying the number of folds. \label{tab:cv-vary-fold}')


'''
2.3 Re-train models with full sample.
'''
for model in models:
    model_name = model.__class__.__name__
    model.fit(X.values, y.values.flatten())
    dump(model, 'result/{}.joblib'.format(model_name))

##Coefficient of logistic regression
lr_coef = pd.DataFrame(lr_model.coef_.transpose(), index=X.keys(),columns=['Coefficient']  )
lr_coef.to_latex('result/lr-coef.tex',float_format="%.3f",caption='Coefficients of logistic regression. \label{tab:lr-coef}')



'''
2.4 Probability estimation.
'''
kf = KFold(n_splits=3)
pred_proba_df = pd.DataFrame( index=X.index )
for model in models:
    pred_proba_df[model.__class__.__name__ + '_pred'] = -1
for model in models:
    pred_proba_df[model.__class__.__name__+ '_proba0'] = -1
    pred_proba_df[model.__class__.__name__+ '_proba1'] = -1

for train, test in kf.split(X):
    X_train = X.iloc[train]
    X_test = X.iloc[test]
    y_train = y.iloc[train]
    y_test = y.iloc[test]
    for model in models:
        model.fit(X_train.values, y_train.values.flatten())
        model_name = model.__class__.__name__
        pred_proba_df.loc[:,model_name + '_pred'].iloc[test] = model.predict(X_test)
        pred_proba_df.loc[:,model_name+ '_proba0'].iloc[test] = model.predict_proba(X_test)[:,0]
        pred_proba_df.loc[:,model_name+ '_proba1'].iloc[test] = model.predict_proba(X_test)[:,1]

pred_proba_df['Outcome'] = y.values#y['Outcome']
pred_proba_df['case_identifier'] = case_identifier
models_ = [rf_model,xgb_model, svc_model, kn_model,  gp_model, ab_model,lr_model  ]
pred_proba_df.sort_values(by = ['Outcome'] + [model.__class__.__name__+'_proba0' for model in models_ ], ascending=False,inplace=True)
pred_proba_df.to_csv('result/sklearn-model-proba.csv')

mis_df = copy.deepcopy(pred_proba_df)
for model in models:
    mis_df = mis_df[mis_df[model.__class__.__name__ + '_pred']!=mis_df['Outcome']]
mis_df=mis_df[['Outcome','case_identifier'] + [model.__class__.__name__+'_proba1' for model in models ]]
for model in models:
    mis_df = mis_df.rename(columns = {model.__class__.__name__+'_proba1':model.__class__.__name__})
pred_proba_df.sort_values(by = [model.__class__.__name__+'_proba1' for model in models_ ], ascending=True,inplace=True)
pred_proba_df.sort_values(by = ['Outcome'] , ascending=False,inplace=True)

mis_df.to_latex('result/misclassified_cases_all.tex',index=True,float_format="%.3f",caption='Predicted probability of the consistently misclassfied cases. \label{tab:misclassified-all}')

mis_df_ = mis_df.copy(deep=True)
mis_df_['Outcome'] = mis_df_['Outcome']+1
mis_df_.to_csv('result/misclassified_cases_all.csv')

temp = pd.concat( (pred_proba_df[:6],pred_proba_df[-6:]) ,axis=0 )
temp2=temp[['Outcome'] + [model.__class__.__name__+'_proba0' for model in models ]]
for model in models:
    temp2 = temp2.rename(columns = {model.__class__.__name__+'_proba0':model.__class__.__name__})
temp2.to_latex('result/misclassified_cases.tex',index=True,float_format="%.3f",caption='Predicted probability of the misclassfied cases with high confidence. \label{tab:misclassified}')




'''
3. SHAP values
'''
##Overall feature importance
feature_importance = pd.DataFrame(data=rf_model.feature_importances_, index=X_train.keys())

explainer_xgb = shap.Explainer(xgb_model)
shap_values = explainer_xgb(X)
xgb_shap_df = pd.DataFrame(data=shap_values.values, columns=X.keys(), index=X.index)
xgb_shap_df.to_csv('result/xgb_shap_fill{}.csv'.format(FillType))

shap.plots.bar(shap_values,show=False)
plt.gcf().set_size_inches(plt.gcf().get_size_inches()*np.array([2.9,1.7]))
plt.savefig('result/xgb_shap_importance_fill{}.png'.format(FillType))
plt.close()

mis_ = ["Misclassified" if i in mis_sample else "Correctly classified" for i in X.index]
shap.plots.bar(shap_values.cohorts(mis_).abs.mean(0),show=False)
plt.gcf().set_size_inches(plt.gcf().get_size_inches()*np.array([2.9,1.7]))
plt.savefig('result/xgb_shap_with_mis_importance_fill{}.png'.format(FillType))
plt.close()


'''
4. Transparency/Fairness analyses with a simple tree model
NOTE: We use data with Industry variable.
'''
X_with_industry = pd.read_csv('data/X_scaled_fill{}_dropmissing_{}.csv'.format(FillType,drop_if_missing),index_col=0)
y = pd.read_csv('data/y_fill{}_dropmissing_{}.csv'.format(FillType,drop_if_missing),index_col=0).astype(int)


dt_model = tree.DecisionTreeClassifier(max_depth=4)
scores = cross_val_score(dt_model, X_with_industry, y.values.flatten(), cv=3)
print('CV score - predict using a simple decision tree: {}'.format(scores.mean() ))

##Fairness constraint on industry
industry = ['Industry_1', 'Industry_2', 'Industry_3', 'Industry_4']#'Industry_0',

scores_cb = cross_val_score(dt_model, X_with_industry.drop(columns=industry), y.values.flatten(), cv=3)
print('CV score with CB using a simple decision tree: {}'.format(scores_cb.mean() ))

class fair_model():
    def __init__(self, model, constraint, sensitive_features):
        self.model = model
        self.constraint =constraint
        self.sensitive_features = sensitive_features
    def get_params(self, deep=True):
        # suppose this estimator has parameters "alpha" and "recursive"
        return {"model": self.model, "constraint": self.constraint,"sensitive_features":self.sensitive_features}
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    def fit(self, X,y):
        self.mitigator = ExponentiatedGradient(self.model, self.constraint() )
        self.mitigator.fit(X, y, sensitive_features=X[self.sensitive_features])
    def predict(self, X):
        return self.mitigator.predict(X)
    def score(self, X, y, sample_weight=None):
        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)
dt_dp = fair_model(dt_model, DemographicParity, industry)
scores_dp = cross_val_score(dt_dp, X_with_industry, y.values.flatten(), cv=3)
dt_eo = fair_model(dt_model, EqualizedOdds, industry)
scores_eo = cross_val_score(dt_eo, X_with_industry, y.values.flatten(), cv=3)
print('CV score with DP using a simple decision tree: {}'.format(scores_dp.mean() ))
print('CV score with EO using a simple decision tree: {}'.format(scores_eo.mean() ))


original_stdout = sys.stdout
with open('result/fairness_cv.txt', 'w') as f:
    sys.stdout = f
    print('CV score - predict using a simple decision tree: {}'.format(scores.mean() ))
    print('CV score with CB using a simple decision tree: {}'.format(scores_cb.mean() ))
    print('CV score with DP using a simple decision tree: {}'.format(scores_dp.mean() ))
    print('CV score with EO using a simple decision tree: {}'.format(scores_eo.mean() ))
    # Reset the standard output
    sys.stdout = original_stdout


##Train on full model to draw the tree diagram
dot_data = tree.export_graphviz(dt_model.fit(X_with_industry, y), out_file=None,feature_names=X_with_industry.keys(),class_names=True,filled=True, rounded=True,special_characters=True)
graph = graphviz.Source(dot_data)
graph.render('result/decision-tree',format='png')

dot_data_no_industry = tree.export_graphviz(dt_model.fit(X_with_industry.drop(columns=industry), y), out_file=None,feature_names=X_with_industry.drop(columns=industry).keys(),class_names=True,filled=True, rounded=True,special_characters=True)
graph = graphviz.Source(dot_data_no_industry)
graph.render('result/decision-tree-no-industry',format='png')
