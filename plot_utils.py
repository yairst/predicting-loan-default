import numpy as np
import matplotlib as mpl
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

def plot_categorical_data_hist(df_col, custom_label = None, figsize = None, logy = False):
    if figsize:
        plt.figure(figsize=figsize)
    df_col.value_counts().plot(kind = 'bar', logy = logy)
    if custom_label:
        label = custom_label
    else:
        label = df_col.name
    plt.title(label + ' Frequency')
    plt.xlabel(label)
    plt.ylabel('Frequency')
    

def plot_numerical_data_hist(df_col, bins = None, custom_label = None, figsize = None,
                             logx = False, logy = False):
    if figsize:
        plt.figure(figsize=figsize)
    if logx == True:
        min_val = df_col.min()
        max_val = df_col.max()
        delta = 0
        if min_val <= 0:
            delta = 1 - min_val
        if bins == None:
            bins = mpl.rcParams['hist.bins']
        bins = np.logspace(np.log10(min_val + delta), np.log10(max_val + delta), bins)  - delta        
    df_col.plot(kind = 'hist', bins = bins, logx = logx, logy = logy)
    if custom_label:
        label = custom_label
    else:
        label = df_col.name
        plt.title(label + ' Frequency')
        plt.xlabel(label)
    plt.ylabel('Frequency')
    
def plot_pr_curve(clf, X, y):
    precision, recall, thresholds = precision_recall_curve(y, clf.predict_proba(X)[:,1])
    closest_zero = np.argmin(np.abs(thresholds - 0.5))
    closest_zero_p = precision[closest_zero]
    closest_zero_r = recall[closest_zero]

    plt.figure()
    plt.xlim([0.0, 1.01])
    plt.ylim([0.0, 1.01])
    plt.plot(recall, precision, label='Precision-Recall Curve')
    plt.plot(closest_zero_r, closest_zero_p, 'o', markersize = 12, fillstyle = 'none', c='r', mew=3)
    plt.ylabel('Precision', fontsize=16)
    plt.xlabel('Recall', fontsize=16)
    plt.axes().set_aspect('equal')
    plt.show()
    
def plot_roc_curve(clf, X, y):
    
    try:
        y_score = clf.predict_proba(X)[:,1]
    except AttributeError:
        y_score = clf.decision_function(X)
        
    fpr, tpr, _ = roc_curve(y, y_score)
    roc_auc = auc(fpr, tpr, reorder = True)

    plt.figure()
    plt.xlim([-0.01, 1.00])
    plt.ylim([-0.01, 1.01])
    plt.plot(fpr, tpr, lw=3, label='ROC curve (area = {:0.2f})'.format(roc_auc))
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.title('ROC curve', fontsize=16)
    plt.legend(loc='lower right', fontsize=13)
    plt.plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--')
    plt.axes().set_aspect('equal')
    plt.show()
   
  