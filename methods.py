import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
import joblib

X = np.array(pd.read_csv('dataset_poly_train.csv'))
y = np.array(pd.read_csv('dataset_poly_train_labels.csv'))


kf = KFold(n_splits=5, shuffle=True, random_state=42)



# clf0 = LogisticRegression(penalty='l1', max_iter=100000, solver='liblinear', class_weight=None)

# clf0 = SVC(class_weight='balanced', probability=True)


# clf0 = GaussianNB()


# clf0 = BaggingClassifier(estimator=DecisionTreeClassifier(criterion='gini',
#                                                           max_depth=50,
#                                                           max_leaf_nodes=200,
#                                                           splitter='best',
#                                                           class_weight=None),
#                          n_estimators=1000, n_jobs=11, random_state=1)

# clf0 = AdaBoostClassifier(n_estimators=100, learning_rate=0.5, random_state=1)





clf0 = BaggingClassifier(estimator=AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=10,
                                                                                       max_leaf_nodes=20,
                                                                                       class_weight={0: 1, 1: 1.3}),
                                                      n_estimators=35,
                                                      learning_rate=0.3),
                         n_estimators=37)


for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    print(X_train.shape, y_train.shape)
    clf0.fit(X_train, y_train)

joblib.dump(clf0, 'best_BaggingClassifier_(AdaB).pkl')

# base_estimator = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=10, max_leaf_nodes=20, class_weight={0: 1, 1: 1.3}))
#
# clf = BaggingClassifier(base_estimator=base_estimator, n_estimators=37)
#
# param_grid = {'base_estimator__learning_rate': uniform(0.1, 1.0)}
#
# random_search = RandomizedSearchCV(clf, param_distributions=param_grid, n_iter=10, cv=5)
#
# random_search.fit(x_train, y_train)
#
# print("Best Parameters: ", random_search.best_params_)
# print("Best Score: ", random_search.best_score_)