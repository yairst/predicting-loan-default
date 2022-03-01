
# coding: utf-8

# In[1]:

# for auto-reload user code (imported py file) on each execution of code from it
# need to be executed only once
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')


# In[2]:

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plot_utils
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel, RFECV

sns.set()
get_ipython().magic('matplotlib inline')


# first glance at the dataset:

# In[3]:

df = pd.read_csv('dataset.csv')
df.head()


# # Reports

# ### Credit amount as function of purpose

# graph:

# In[4]:

plt.figure(figsize=(15,10))
sns.boxplot(x="purpose", y="credit_amount", data=df)


# In general the credits are higher for used car or buisness loans.

# table:

# In[5]:

grouped = df.groupby('purpose')
grouped['credit_amount'].describe().sort_values('mean', ascending = False)


# ### Total amount of loans greater than 10,000 that were given to unemployed males over the age of 30 

# In[6]:

print(df.personal_status.unique())


# In[7]:

(df[(df['personal_status'] != "'female div/dep/mar'") &
          (df['employment'] == "unemployed") &
          (df['age'] > 30) &
          (df['credit_amount'] > 10000)].credit_amount.sum())
 


# ### Credit Amount distribution as function of personal status

# graph:

# In[8]:

plt.figure(figsize=(15,10))
sns.violinplot(x="personal_status", y="credit_amount", data=df)


# table:

# In[9]:

df.groupby('personal_status').credit_amount.describe().sort_values(by='mean', ascending = False)


# ### Number of Customers who are employed over 4 years

# In[10]:

df['employment'].unique()


# In[11]:

((df['employment'] == "'>=7'") | (df['employment'] == "'4<=X<7'")).sum()


# ## Train a Model, Test and Evaluate

# split to train and test:

# In[12]:

TRAIN_FRAC = 0.8
df_train, df_test, y_train, y_test = train_test_split(df.drop(['class'], axis = 1), df['class'], test_size=0.2, random_state=42)
df_train.head()


# check the types of the columns:

# In[13]:

df_train.dtypes


# statistics of the numerical columns:

# In[14]:

df_train.describe()


# 1. From the numeric columns, only duration, credit amount and age can be treated as numbers and the rest as categoricals.
# 2. There are no missing values.

# statistics of the categorical columns:

# In[15]:

df_train.describe(include = ['O'])


# 1. It seems that all the categoricals features have small amount of unique values - no need in aggregation
# 2. No missing values

# Distribution of the numericals and their log:

# In[16]:

plt.figure(figsize=(18,20))
ax1 = plt.subplot(3,2,1)
sns.distplot(df_train.age, ax = ax1)
ax2 = plt.subplot(3,2,2)
sns.distplot(np.log(df_train.age), ax = ax2)
plt.xlabel('log age')

ax3 = plt.subplot(3,2,3)
sns.distplot(df_train.duration, ax = ax3)
ax4 = plt.subplot(3,2,4)
sns.distplot(np.log(df_train.duration), ax = ax4)
plt.xlabel('log duration')

ax5 = plt.subplot(3,2,5)
sns.distplot(df_train.credit_amount, ax = ax5)
ax6 = plt.subplot(3,2,6)
sns.distplot(np.log(df_train.credit_amount), ax = ax6)
plt.xlabel('log credit_amount')


# Since the distributions of the log of the numerics are more close to a normal dist. it make sense to add them as features.
# Also, first the numeric features will be used as is. After, based on the first results, maybe some of them will be categorized in order to improve performance.

# Define categoricals and numericals column names lists:

# In[17]:

df_train.columns


# In[18]:

categorical_features = ['checking_status', 'credit_history', 'purpose', 'savings_status', 'employment',
       'installment_commitment', 'personal_status', 'other_parties', 'residence_since', 'property_magnitude',
        'other_payment_plans', 'housing', 'existing_credits', 'job', 'num_dependents', 'own_telephone', 'foreign_worker']
numerical_features = ['duration', 'credit_amount', 'age']


# In[19]:

numerical_features


# add the log of the numericals:

# In[20]:

log_numericals = []
for numerical in numerical_features:
    df_train['log_' + numerical] = np.log(df_train[numerical])
    df_test['log_' + numerical] = np.log(df_test[numerical])
    log_numericals += ['log_' + numerical]
numerical_features += log_numericals
numerical_features


# In[21]:

# check that the log of the numericals added correctly:
df_train[numerical_features].head()


# vectorize categoricals and scale numericals of the train data:

# In[22]:

cat_dict = df_train[categorical_features].to_dict('records')
dict_vect = DictVectorizer()
X_train = dict_vect.fit_transform(cat_dict).todense()

scaler = StandardScaler()
numerical_train_scaled = scaler.fit_transform(df_train[numerical_features])

X_train = np.hstack((X_train, numerical_train_scaled))
# df_train = pd.concat((pd.get_dummies(df_train[categorical_features]), df_train[numerical_features]), axis = 1)

X_train.shape


# In[23]:

print(dict_vect.feature_names_)


# In[24]:

# vectorize categoricals and scale numericals for the test data:

cat_dict_test = df_test[categorical_features].to_dict('records')
X_test = dict_vect.transform(cat_dict_test).todense()

numerical_test_scaled = scaler.transform(df_test[numerical_features])
X_test = np.hstack((X_test, numerical_test_scaled))

X_test.shape


# transform the target to 0/1:

# In[25]:

le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)
le.classes_


# In[26]:

y_train[:10]


# In[27]:

le.inverse_transform(y_train[:10])


# In[28]:

y_train.sum()/len(y_train)


# 70% of the loans are good - the data is naturally imbalanced (most of the pepole return the loan...) hence we can't use accuracy as performance measure (just predict all as good loans and we get accuracy of 70%...).
# 
# Under the general purpose of 80% "success" we need to refine the goal. In our case of loans, we want to detect as much as bad loans but we also want to ensure not to reject to much potenially good loans. Now if we define bad loans as the positive class (1) and good loans as negative (0) that's mean we want to maximaize recall and minimize false positive rate (FPR). Thats exactly what The ROC curve gives along with the AUC-score which give us a single number of the goodness of the ROC.
# 
# Nonetheless due to the imbalance of the data it should be noted that it is quite "easy" to get good/low FPR and even if the recall is not so high, still get good auc score. To take care of it we will also look at the f1-score which concentrates on the performance of detecting the positive class (harmonic mean of the precision and recall).

# In[30]:

# since the label encoder define bad as 0 and good as 1 and we are focused on the performance of the bad class
# we will invert the target values:
y_train = 1 - y_train
y_test = 1 - y_test


# In[31]:

y_train.sum()/len(y_train)


# start with logistic regression with default parameters:

# In[32]:

from sklearn.linear_model import LogisticRegression

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)


# In[33]:

# Predict
pred_train = model.predict(X_train)
pred_test = model.predict(X_test)

y_train_score = model.decision_function(X_train)
print('train auc score: ', roc_auc_score(y_train, y_train_score))
print('train f1-score: ', f1_score(y_train, pred_train))
print(classification_report(y_train, pred_train))
print(confusion_matrix(y_train, pred_train))

y_test_score = model.decision_function(X_test)
print('test auc score: ', roc_auc_score(y_test, y_test_score))
print('test f1-score: ', f1_score(y_test, pred_test))
print(classification_report(y_test, pred_test))
print(confusion_matrix(y_test, pred_test))


# In[34]:

plot_utils.plot_roc_curve(model, X_test, y_test)


# First try and we got satisfing results! the test auc score is 0.82 - larger than 0.8!
# 
# Anyway, if we dig deeper we find that while the model keeps FPR rate low (10%)as needed, the recall is (very) low - only 0.56, and it is reflected in test f1-score of only 0.62.
# 
# So, we can srop here and be content with the current results or the seek for better models that treat the positive class better (not on the expense of low FPR).
# 
# Train vs test error:
# We can see that the test score is even higher than the training, so there is no overfitting but maybe there is too much regularization (maybe we need higher C value). Let's try with no regularization (very high C value):

# In[35]:

from sklearn.linear_model import LogisticRegression

# Train the model
model = LogisticRegression(C = 10000)
model.fit(X_train, y_train)


# In[36]:

# Predict
pred_train = model.predict(X_train)
pred_test = model.predict(X_test)

y_train_score = model.decision_function(X_train)
print('train auc score: ', roc_auc_score(y_train, y_train_score))
print('train f1-score: ', f1_score(y_train, pred_train))
print(classification_report(y_train, pred_train))
print(confusion_matrix(y_train, pred_train))

y_test_score = model.decision_function(X_test)
print('test auc score: ', roc_auc_score(y_test, y_test_score))
print('test f1-score: ', f1_score(y_test, pred_test))
print(classification_report(y_test, pred_test))
print(confusion_matrix(y_test, pred_test))

plot_utils.plot_roc_curve(model, X_test, y_test)


# A very little improvement with no regularization. And since with no regularization both train and test results are similiar, we are not in high variance situation. Does it mean that the model has high bias? depends... formally yes, because we already meet the goal. But if we look only on the f1-score: 0.64 is still low. we want better. so from this prespective the model still have high bias and we need to refine it / to increase its complexity which mean: more parameters, eliminate possible bugs in our feature extraction step and/or other algorithm...

# ### feature importance/ selection:

# Altough we are not in high variance situation (so we don't need to reduce the number of features) it is interesting to see which are the more important predictors:

# In[37]:

# first glance at the coefs:
model.coef_


# In[38]:

# let sort them w.r.t thier abs value:
importance_idx = np.argsort(abs(model.coef_[0]))[::-1]
importance_idx


# In[39]:

# extract feature name list:
features_names = dict_vect.feature_names_ + numerical_features


# In[40]:

# feature sorted by importance:
[features_names[idx] for idx in importance_idx]


# plot the most N important features:

# In[41]:

plt.figure(figsize=(18,14))
# ax = plt.bar(np.arange(10), model.coef_[0][importance_idx][:10],
#              tick_label = [features_names[idx] for idx in importance_idx][:10])
sns.barplot([features_names[idx] for idx in importance_idx][:10], model.coef_[0][importance_idx][:10])
plt.xticks(rotation = 45, size = 12)


# From the top 10 we can see that there aren't features with significant higher importance (the coefs value decrease slowly from the begining).
# 
# Some insights:
# 1. loan purposes of education or new car are good predictors of bad loans..
# 2. log age (and not age... as we concluded earlier) is good predictor. since its coef is negative (and good loan = 0) it can be concluded that as the customer is older the model will tend to classify 0

# an overall look on feature importance:

# In[42]:

plt.figure(figsize=(18,10))
sns.barplot([features_names[idx] for idx in importance_idx], model.coef_[0][importance_idx])


# predictors of bad loans:

# In[43]:

print([features_names[idx] for idx in importance_idx if model.coef_[0,idx] > 0 ])


# predictors of good loans:

# In[44]:

print([features_names[idx] for idx in importance_idx if model.coef_[0,idx] < 0 ])


# From inspection of the predictors above there are some which make sense ,like high credit amount -> bad loan, but not a few which not make sense, like job='unemp/unskilled non res'-> good loan. Thats correspond with the low f1-score of the current model.

# A function to inspect samples and their prediction, compared to the true value:

# In[45]:

def examine_predictions(sample_num):
    for i in range(10):
        print([features_names[idx] for idx in importance_idx][i].ljust(50) + str(X_train[sample_num,importance_idx[i]]))
    print()
    print('true class value: {}'.format('bad' if y_train[sample_num] == 1 else 'good'))
    print('predicted class value: {}'.format('bad' if pred_train[sample_num] == 1 else 'good'))


# In[46]:

examine_predictions(200)


# To further validate that we not having high variance situation and hence we don't need to reduce the number of features, recursive feature selection tool is activated:

# In[47]:

# Create the RFE object and compute a cross-validated score.
svc = LinearSVC()

rfecv = RFECV(estimator=svc, scoring='roc_auc')
rfecv.fit(X_train, y_train)

print("Optimal number of features : %d" % rfecv.n_features_)

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (f1 score)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()


# Indeed, from the 64 features, the algorithm point out most of them, 52, as the optimal number and even so there is a flato and not decline in the performance as we add more features (in compliance with that the the model doesn't have high variance). let's check what happen if we choose only those 52 features (we expect to get the same performance or little worse):

# In[48]:

X_train_new = X_train[:,rfecv.get_support()]
X_test_new = X_test[:,rfecv.get_support()]


# In[49]:

print([name for i, name in enumerate(features_names) if rfecv.get_support()[i]])


# In[50]:

X_train_new.shape


# In[51]:

model = LogisticRegression(C = 10000)
model.fit(X_train_new, y_train)

# Predict
pred_train = model.predict(X_train_new)
pred_test = model.predict(X_test_new)

y_train_score = model.decision_function(X_train_new)
print('train auc score: ', roc_auc_score(y_train, y_train_score))
print('train f1-score: ', f1_score(y_train, pred_train))
print(classification_report(y_train, pred_train))
print(confusion_matrix(y_train, pred_train))

y_test_score = model.decision_function(X_test_new)
print('test auc score: ', roc_auc_score(y_test, y_test_score))
print('test f1-score: ', f1_score(y_test, pred_test))
print(classification_report(y_test, pred_test))
print(confusion_matrix(y_test, pred_test))

plot_utils.plot_roc_curve(model, X_test_new, y_test)


# Indeed, using only those 52 features are enough to get the same results as with all the 64.

# ## Tries to improve results

# let add another feature: The credit amount divided by the duration:

# In[52]:

df_train['principal_month'] = df_train['credit_amount'] / df_train['duration']
df_test['principal_month'] = df_test['credit_amount'] / df_test['duration']


# In[53]:

plt.figure(figsize=(18,6))
ax1 = plt.subplot(1,2,1)
sns.distplot(df_train.principal_month, ax = ax1)
ax2 = plt.subplot(1,2,2)
sns.distplot(np.log(df_train.principal_month), ax = ax2)
plt.xlabel('log principal_month')


# From the graph it seems that the log of the principal may be a good/ better feature. need to add it:

# In[54]:

df_train['log_principal_month'] = np.log(df_train['principal_month'])
df_test['log_principal_month'] = np.log(df_test['principal_month'])


# In[55]:

numerical_features = numerical_features + ['principal_month', 'log_principal_month']
features_names += ['principal_month', 'log_principal_month']
numerical_features


# update X_train and X_test with the new 2 features:

# In[56]:

X_train = X_train[:,:-6]

scaler = StandardScaler()
numerical_train_scaled = scaler.fit_transform(df_train[numerical_features])

X_train = np.hstack((X_train, numerical_train_scaled))
# df_train = pd.concat((pd.get_dummies(df_train[categorical_features]), df_train[numerical_features]), axis = 1)

X_train.shape


# In[57]:

X_test = X_test[:,:-6]

numerical_test_scaled = scaler.transform(df_test[numerical_features])

X_test = np.hstack((X_test, numerical_test_scaled))
# df_test = pd.concat((pd.get_dummies(df_test[categorical_features]), df_test[numerical_features]), axis = 1)

X_test.shape


# let's check if it does something to the results:

# In[58]:

model = LogisticRegression(C = 10000) 
model.fit(X_train, y_train)

y_train_score = model.decision_function(X_train)
print('train auc score: ', roc_auc_score(y_train, y_train_score))
print('train f1-score: ', f1_score(y_train, pred_train))
print(classification_report(y_train, pred_train))
print(confusion_matrix(y_train, pred_train))

y_test_score = model.decision_function(X_test)
print('test auc score: ', roc_auc_score(y_test, y_test_score))
print('test f1-score: ', f1_score(y_test, pred_test))
print(classification_report(y_test, pred_test))
print(confusion_matrix(y_test, pred_test))

plot_utils.plot_roc_curve(model, X_test, y_test)


# Nice. The new 2 features add some little improvement to the results. Let's validate it:

# In[59]:

importance_idx = np.argsort(abs(model.coef_[0]))[::-1]
[features_names[idx] for idx in importance_idx[:20]]


# Indeed, the prinicapl is the 10th most important feature and the lof_principal is the 20th..

# After examine the features list it seems that some of the categoricals are not as they should be. Specifically, the categoricals that represented as integers, like 'existing_credits' and 'residence_since'. Let's check how DictVectorizer treated them:

# In[60]:

# check to numerical integers features - it seems that dictVectorizer doesn't work well for them:
check_dict = df_train[['existing_credits','residence_since']].to_dict('records')
dict_vect_check = DictVectorizer()
temp = dict_vect_check.fit_transform(check_dict).todense()
temp[:10,:]


# yes... it seems that dictVectorizer doesn't binarize integers... we need to do it with OneHotEncoder.

# In[61]:

df_train.dtypes


# In[62]:

print(df_train.columns)
print(len(df_train.columns))


# we need to seperate the integers categorical from the text categoricals:

# In[63]:

categorical_features = ['checking_status', 'credit_history', 'purpose', 'savings_status', 'employment',
        'personal_status', 'other_parties', 'property_magnitude',
        'other_payment_plans', 'housing', 'job', 'own_telephone', 'foreign_worker']
int_features = ['installment_commitment','residence_since','existing_credits', 'num_dependents']
numerical_features = ['duration', 'credit_amount', 'age', 'principal_month', 'log_duration', 'log_credit_amount', 'log_age',
                     'log_principal_month']
print(len(categorical_features+int_features+numerical_features))


# In[64]:

df_train[numerical_features].head()


# In[65]:

df_train[int_features].head()


# And now vectorize the categoricals sperately for text and integers:

# In[66]:

# vectorize categoricals and scale numericals for the train data:

cat_dict = df_train[categorical_features].to_dict('records')
dict_vect = DictVectorizer(sparse = False)
X_train = dict_vect.fit_transform(cat_dict)

one_hot = OneHotEncoder(sparse = False)
one_hot_train = one_hot.fit_transform(df_train[int_features])

scaler = StandardScaler()
numerical_train_scaled = scaler.fit_transform(df_train[numerical_features])

X_train = np.hstack((X_train, one_hot_train, numerical_train_scaled))
# df_train = pd.concat((pd.get_dummies(df_train[categorical_features]), df_train[numerical_features]), axis = 1)

X_train.shape


# Unfortunately oneHotEncoder doesn't preserve the original names of the features (as in dictVectorizer) so we need to do it manually.

# First let's extract the unique values of those features:

# In[67]:

for int_feat in int_features:
    print(df_train[int_feat].unique())


# Make a list of one hot encoding feature names (asumes that the oneHotEncoder sort the unique values before fitting):

# In[68]:

start_idx_of_one_hot = len(dict_vect.feature_names_)
end_idx_of_one_hot = start_idx_of_one_hot + len(one_hot.active_features_) - 1
one_hot_names = []
for feat_name in int_features:
    for feat_val in sorted(df_train[feat_name].unique()):
        one_hot_names += [feat_name + '_' + str(feat_val)]
one_hot_names


# sanity check: check that indeed the enoding is as it should be after sort the unique values and according to the order of one_hot_names:

# In[69]:

print(one_hot_train[:10,9]) # chose column num from 0 to 13 (3 features * 4 values for each + 1 feature * 2 values)
print(one_hot_names[9]) # choose the same index
print(df_train[int_features[2]].head(10)) 


# sanity check passed!

# In[70]:

# concatenating all the names
feature_names = dict_vect.feature_names_ + one_hot_names + numerical_features
print(len(feature_names))
print(feature_names)


# In[71]:

# vectorize categoricals and scale numericals for the test data:

cat_dict = df_test[categorical_features].to_dict('records')
X_test = dict_vect.transform(cat_dict)

one_hot_test = one_hot.transform(df_test[int_features])

numerical_test_scaled = scaler.transform(df_test[numerical_features])

X_test = np.hstack((X_test, one_hot_test, numerical_test_scaled))

X_test.shape


# In[72]:

model = LogisticRegression(C = 10000) 
model.fit(X_train, y_train)

# Predict
pred_train = model.predict(X_train)
pred_test = model.predict(X_test)

y_train_score = model.decision_function(X_train)
print('train auc score: ', roc_auc_score(y_train, y_train_score))
print('train f1-score: ', f1_score(y_train, pred_train))
print(classification_report(y_train, pred_train))
print(confusion_matrix(y_train, pred_train))

y_test_score = model.decision_function(X_test)
print('test auc score: ', roc_auc_score(y_test, y_test_score))
print('test f1-score: ', f1_score(y_test, pred_test))
print(classification_report(y_test, pred_test))
print(confusion_matrix(y_test, pred_test))

plot_utils.plot_roc_curve(model, X_test, y_test)


# Interestingly, adding the last categroical features seems to be the straw that broke the camel's back where while the train results improved a little, there is a small decrease in the auc score.

# ## Summary

# Linear regression with no regularization and default features seems to get the desired results with 83% success to predict the quality of the loan, or more precisely the model has probability of 0.83 to rank a randomly chosen positive example (bad loan) above a randomly chosen negative one (good loan).
# 
# While the results are good there is more to do in order to improve recall and precision of predicting bad loans.

# In[ ]:



