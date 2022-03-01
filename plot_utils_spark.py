import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

   
def plot_roc_curve(roc,roc_auc,rawPrediction):
    
    plt.figure()
    plt.xlim([-0.01, 1.00])
    plt.ylim([-0.01, 1.01])
    plt.plot(roc.FPR, roc.TPR, lw=3, label='ROC curve (area = {:0.2f})'.format(roc_auc))
    
    closest_zero_idx = np.argmin(np.abs(sorted(rawPrediction, reverse=True)))
    closest_zero_fpr = roc.loc[closest_zero_idx,'FPR']
    closest_zero_tpr = roc.loc[closest_zero_idx,'TPR']
    plt.plot(closest_zero_fpr, closest_zero_tpr, 'o', markersize = 12, fillstyle = 'none', c='r', mew=3)
    
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.title('ROC curve', fontsize=16)
    plt.legend(loc='lower right', fontsize=13)
    plt.plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--')
    plt.axes().set_aspect('equal')
    plt.show()
   
   
  