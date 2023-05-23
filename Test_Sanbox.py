import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import joblib

from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, roc_curve

from custom_metrics import TPR_FPR

import warnings
from warnings import filterwarnings


warnings.filterwarnings("ignore", category=DeprecationWarning)
filterwarnings('ignore')

x = np.array(pd.read_csv('dataset_poly_test.csv'))
y = np.array(pd.read_csv('dataset_poly_test_labels.csv'))

# clf0 = 'best_GNB.pkl'
clf1 = 'best_SVC.pkl'
clf2 = 'best_AdaB.pkl'
clf3 = 'best_BaggingClassifier(DT).pkl'
clf4 = 'best_BaggingClassifier_(AdaB).pkl'
clf5 = 'best_LogisticRegression.pkl'

print("xyxyx", x.shape, y.shape)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# clf0 = joblib.load(clf0)
clf1 = joblib.load(clf1)
clf2 = joblib.load(clf2)
clf3 = joblib.load(clf3)
clf4 = joblib.load(clf4)
clf5 = joblib.load(clf5)


estimators = [
    # ('GNB', clf0),
    ('SVC', clf1),
    ('ADB', clf2),
    ('Bag(DT)', clf3),
    ('Bag(ad)', clf4),
    ('L1', clf5)
]
clf = StackingClassifier(estimators=estimators,
                         final_estimator=LogisticRegression(penalty='l2', class_weight='balanced'),
                         passthrough=False)


kf = KFold(n_splits=5, shuffle=True, random_state=1)
train_scores, train_cms, scores, cms = [], [], [], []

print(x.shape)
df_tests = 0
df_thresholds = 0
for train_index, test_index in kf.split(x):
    for est in estimators:
        clf = est[1]
        clf.fit(x[train_index], y[train_index])
        train_probs = clf.predict_proba(x[train_index])
        train_preds = clf.predict(x[train_index])
        preds = clf.predict(x[test_index])
        probs = clf.predict_proba(x[test_index])

        probs = np.mean(probs, axis=1)
        print('train_preds', train_preds, preds, probs)

        train_acc = accuracy_score(y[train_index], train_preds)
        train_auc = roc_auc_score(y[train_index], train_probs, multi_class='ovr')
        train_cm = confusion_matrix(y[train_index], train_preds)
        print('train_preds', train_acc, train_auc, train_cm)

        print("shape", y[test_index].ravel().shape, preds.shape, probs.shape)

        unique_classes = np.unique(y[test_index].ravel())
        num_classes = unique_classes.shape[0]
        print('num_classes', num_classes)

        acc = accuracy_score(y[test_index], preds)
        cm = confusion_matrix(y[test_index], preds)

        print(acc)
        print(cm)
        train_scores.append(np.array([train_acc, train_auc]))
        train_cms.append(train_cm)
        scores.append(np.array([acc]))
        cms.append(cm)
        roc = roc_curve(y[test_index], probs)
        plt.plot(roc[0], roc[1], label=est[0])

        df_tst, df_thr = TPR_FPR(roc, est[0], True)
        if type(df_tests) == int:
            df_tests = df_tst
            df_thresholds = df_thr
        else:
            df_tests = pd.concat([df_tests, df_tst])
            df_thresholds = pd.concat([df_thresholds, df_thr])
        print(clf.classes_)
        print(df_thresholds)

plt.legend()
plt.show()

print('\ntrain scores')
print(np.array(train_scores).mean(axis=0))
print(np.array(train_cms).mean(axis=0), '\ntest scores')
print(np.array(scores).mean(axis=0))
print(np.array(cms).mean(axis=0))
print('\naverage thresholds')
print(df_thresholds.groupby('classifier').mean())
#df_thresholds.groupby('classifier').mean().to_csv('thresholds.csv')

sns_plot = sns.heatmap(df_tests.groupby('classifier').mean(), annot=True, cmap="crest")
plt.show()