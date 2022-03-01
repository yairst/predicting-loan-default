

def print_confusion_matrix(df):
    tp = df[(df.label == 1) & (df.prediction == 1)].count()
    tn = df[(df.label == 0) & (df.prediction == 0)].count()
    fp = df[(df.label == 0) & (df.prediction == 1)].count()
    fn = df[(df.label == 1) & (df.prediction == 0)].count()
    print("True Negatives:", tn)
    print("False Positives:", fp)
    print("False Negatives:", fn)
    print("True Positives:", tp)
    print("Total", df.count())
    
def model_summary(train_summary, test_summary, train_pred_df, test_pred_df):
    print('train auc score: ', train_summary.areaUnderROC)
    print('train f1-score: ', train_summary.fMeasureByLabel()[1])
    print('train precision: ', train_summary.precisionByLabel[1])
    print('train recall: ', train_summary.recallByLabel[1])
    print_confusion_matrix(train_pred_df)
    print('test auc score: ', test_summary.areaUnderROC)
    print('test f1-score: ', test_summary.fMeasureByLabel()[1])
    print('test precision: ', test_summary.precisionByLabel[1])
    print('test recall: ', test_summary.recallByLabel[1])
    print_confusion_matrix(test_pred_df)



