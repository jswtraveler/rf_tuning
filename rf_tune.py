from sklearn.model_selection import cross_val_score, KFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score, precision_score, recall_score,roc_auc_score
from sklearn.ensemble import RandomForestRegressor
import numpy as np

X_train, X_test, y_train, y_test = \
    sklearn.model_selection.train_test_split(X, y, random_state=1985)

test=df_test

feats = set(test.columns)-set([])#at least target if not others

x1=test[list(feats)]
y1=test['target']

param_grid= {
'n_estimators': range(100,300,50)
,'max_features': range(.2,.6,.1)#Also try none
,'min_samples_leaf':[50,100]
,'bootstrap': ['True','False']
#,'class_weight': [] #USE IF MULTI-TARGETS

}

rf = RandomForestRegressor(random_state=1985, eval_metric=, n_jobs=-1)

grid_rf = GridSearchCV(rf, param_grid = param_grid, cv=3)

grid_xgb.fit(X1, y1)

grid_rf.cv_results_
grid_rf.best_estimator_
grid_rf.best_score_
grid_rf.best_params_

n_iter_search = 50
ran_rf = RandomizedSearchCV(rf, param_distributions=param_dist, cv=3, n_iter=n_iter_search)

ran_rf.fit(X1, y1)

ran_rf.cv_results_
ran_rf.best_estimator_
ran_rf.best_score_
ran_rf.best_params_
