import sys
import pickle
import os
import numpy as np
import pandas as pd



sys.path.append("../final_project/")
sys.path.append("../tools/")
os.chdir('/Users/Alex/ud120-projects/tools')
os.chdir('/Users/Alex/ud120-projects/final_project')
import tester
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, SVC
from tester import dump_classifier_and_data
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.feature_selection import SelectKBest, f_classif
import pprint
pp = pprint.PrettyPrinter(indent=4)
### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi',
                'salary',
                'bonus', 
                'long_term_incentive', 
                'deferred_income', 
                'deferral_payments',
                'loan_advances', 
                'other',
                'expenses', 
                'director_fees',
                'total_payments',
                'exercised_stock_options',
                'restricted_stock',
                'restricted_stock_deferred',
                'total_stock_value',
                'email_address',
                'to_messages',
                'from_messages',
                'from_this_person_to_poi',
                'from_poi_to_this_person',
                'shared_receipt_with_poi']

financial_features = ['salary',
                'bonus', 
                'long_term_incentive', 
                'deferred_income', 
                'deferral_payments',
                'loan_advances', 
                'other',
                'expenses', 
                'director_fees',
                'total_payments',
                'exercised_stock_options',
                'restricted_stock',
                'restricted_stock_deferred',
                'total_stock_value']
                
email_features = ['to_messages',
                'from_messages',
                'from_this_person_to_poi',
                'from_poi_to_this_person',
                'shared_receipt_with_poi']

            
### Load the dictionary containing the dataset

with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


    
    
# Transform the dictionary to a DataFrame (personal preference)
df = pd.DataFrame.from_dict(data_dict, orient = 'index')

#Order columns in DataFrame, exclude email column
df = df[features_list]
df = df.replace('NaN', np.nan)
df = df.drop('email_address', 1)
df.info()

#Count NaNs
total_points = df.count().sum()+df.isnull().sum().sum()
share_nan = round(100*(df.isnull().sum().sum() / float(total_points)),2)
nans = df.isnull().sum().sum()
print "Number of data points: ", total_points
print "Out of them NaNs are %d (%2.2f%%)" % (nans, share_nan)
print "NaNs by feature:", df.isnull().sum()

#Focus on NaNs for email data 
df_email = df.loc[:,email_features]
df_email[df_email.isnull().any(axis=1)].info()

total_points_email = df_email.count().sum()+df_email.isnull().sum().sum()
share_nan_email = round(100*(df_email.isnull().sum().sum() / 
                             float(total_points_email)),2)
nans_email = df_email.isnull().sum().sum()
print "Number of email data points:", total_points_email
print "Out of them NaNs are %d (%2.2f%%)" % (nans_email, share_nan_email)
print "NaNs by feature:", df_email.isnull().sum()

#Replace NaNs with zeros in the financial data
#because nans are zeros for financial features
df.ix[:,financial_features] = df.ix[:,financial_features].fillna(0)

#Recompute NaNs shares
total_points = df.count().sum()+df.isnull().sum().sum()
share_nan = round(100*(df.isnull().sum().sum() / float(total_points)),2)
nans = df.isnull().sum().sum()
print "Number of data points: ", total_points
print "Out of them NaNs are %d (%2.2f%%)" % (nans, share_nan)
print "NaNs by feature:", df.isnull().sum()

#See how many POIs
poi_split = df.poi.value_counts()
print "Number of POIs: %d" % poi_split[1]
print "Number of non-POIs: %d" % poi_split[0]

#check data: summing payments and stock features and
#compare with total_payments and total_stock_value
payment_vals = ['salary',
            'bonus', 
            'long_term_incentive', 
            'deferred_income', 
            'deferral_payments',
            'loan_advances', 
            'other',
            'expenses', 
            'director_fees']
            
stock_vals = ['exercised_stock_options',
                'restricted_stock',
                'restricted_stock_deferred']
#display payment data for rows where totals are not consistent with data
df[df[payment_vals].sum(axis='columns') != df.total_payments].ix[:,1:11]

#display stock data for rows where totals are not consistent with data
df[df[stock_vals].sum(axis='columns') != df.total_stock_value].ix[:,11:15]

#Two entries are not correct (BELFER and BHATNAGAR). Double check with pdf -->
#columns were
#messed up --> fix manually

#Fix payment data
df.ix['BELFER ROBERT',1:10]=[0,0,0,-102500,0,0,0,3285,102500]
df.ix['BHATNAGAR SANJAY',1:10]=[0,0,0,0,0,0,0,137864,0]


#Fix stocks data
df.ix['BELFER ROBERT',11:14]=[0,44093,-44093]
df.ix['BHATNAGAR SANJAY',11:14]=[15456290, 2604490,-2604490]

#Sum up to get totals for payments and stocks
df.ix['BELFER ROBERT',10] = df.ix['BELFER ROBERT',1:10].sum() 
df.ix['BHATNAGAR SANJAY',10]=df.ix['BHATNAGAR SANJAY',1:10].sum()
df.ix['BELFER ROBERT',14] = df.ix['BELFER ROBERT',11:14].sum() 
df.ix['BHATNAGAR SANJAY',14]=df.ix['BHATNAGAR SANJAY',11:14].sum()


#Check correctness again
wrong_payment_totals = df[df[payment_vals].sum(axis='columns') != 
                          df.total_payments].ix[:,1:11]
wrong_stock_totals = df[df[stock_vals].sum(axis='columns') != 
                        df.total_stock_value].ix[:,11:15]
if wrong_payment_totals.empty:
    print "all good with payment totals"
else:
    print 'ERROR'
if wrong_stock_totals.empty:
    print "all good with stock totals"
else:
    print 'ERROR'

    
### Task 2: Remove outliers
#scatter plotting total_payments vs total_stock_value
plot_features = ['total_payments', 'total_stock_value']
df.ix[:,plot_features].plot(kind='scatter',x='total_payments',
    y='total_stock_value')
#noticing some outliers, checking who those are
df.loc[df['total_payments'] > 10000000].ix[:,['poi','total_payments']]

#same for salary vs bonus
plot_features = ['salary', 'bonus']
df.ix[:,plot_features].plot(kind='scatter',x='salary',
    y='bonus')
#noticing some outliers, checking who those are
df.loc[df['salary'] > 1000000].ix[:,['poi','salary']]



#Excluding the "TOTAL" row. LAY, SKILLING are POIs --> keep
#FREVERT not POIs but a top manager --> keep
df = df.drop(['TOTAL'])

#check if some rows contain only zeros
df[df.abs().sum(axis=1) == 0]
#kill the row
df = df.drop(['LOCKHART EUGENE E'])

#list of names left in the df
names = df.index.values
for record in names:
    print record
# drop THE TRAVEL AGENCY IN THE PARK 
df = df.drop(['THE TRAVEL AGENCY IN THE PARK'])



#Replotting with POIs in red:
ax = df.loc[df['poi'] == True].plot(kind='scatter', x='salary', y='bonus',
             color='Red', label='POI')
df.loc[df['poi'] == False].plot(kind='scatter', x='salary', y='bonus',
        color='Blue', label='non-POI', ax=ax);

### Task 3: Create new feature(s)
##sum of all interactions with POIs
df['total_poi_inter'] = df['shared_receipt_with_poi'] + \
                                      df['from_this_person_to_poi'] + \
                                      df['from_poi_to_this_person']

##to_poi/from_messages ratio - obtained by division
df['to_poi_ratio'] = \
 df['from_this_person_to_poi'] / df['from_messages']
#from_poi/to_messages ratio - - obtained by division
df['from_poi_ratio'] = \
 df['from_poi_to_this_person'] / df['to_messages']

df['total_pay'] = df['total_payments'] + \
                                   df['total_stock_value']

##Explore variables created 
def explore_data(feature_x, feature_y):
    ax = df.loc[df['poi'] == True].plot(kind='scatter', x=feature_x, \
    y=feature_y, color='Red', label='POI')
    df.loc[df['poi'] == False].plot(kind='scatter', x=feature_x, \
    y=feature_y, color='Blue', label='non-POI',ax=ax)
    return
    
explore_data('to_poi_ratio','from_poi_ratio')
#seems like POIs send more emails to other POIs compared to non-POIs

#fill zeros for missing email data
df = df.fillna(0)
df.info()


#split into features/labels
y_df = df.drop('poi',1)
x_df = df['poi']

##Impact of new features
#1 step: save initial dataframe before adding features and 
#           do the selection with k='all'
selector = SelectKBest(score_func=f_classif,k='all')
initial_df = y_df.ix[:,financial_features+email_features]
initial_df.info()
selector.fit(initial_df, x_df)
scores_initial = selector.scores_
features_initial = financial_features+email_features
combined_initial = zip(features_initial, scores_initial)
combined_initial.sort(reverse=True, key= lambda x: x[1])
#2 step: selection with new features included

selector.fit(y_df, x_df)
scores_new = selector.scores_
features_new = list(y_df.columns)
combined_new = zip(features_new, scores_new)
combined_new.sort(reverse=True, key= lambda x: x[1])

pp.pprint(combined_initial)
pp.pprint(combined_new)
#create list of features
X_features = list(y_df.columns)
















#Model selection
sk_fold = StratifiedShuffleSplit(x_df, n_iter=100, test_size=0.1, random_state=42)
scoring_metric = 'recall'
pipeline1 = Pipeline(steps=[('minmaxer', MinMaxScaler()),
                           ('selection', SelectKBest(score_func=f_classif)),
                           ('reducer', PCA()),
                           ('classifier', LogisticRegression())
                           ])

params1 = {'selection__k': [8, 10, 12, 14, 16, 18],
          'classifier__C': [1e-02, 1e-1, 1],
          'classifier__class_weight': ['balanced'],
          'classifier__tol': [1e-3, 1e-4],
          'classifier__random_state': [42],
          'reducer__n_components': [1, 2, 3],
          'reducer__whiten': [True]
          }


grid_searcher1 = GridSearchCV(pipeline1, param_grid=params1, cv=sk_fold,
                       n_jobs=-1, scoring=scoring_metric)

grid_searcher1.fit(y_df,x_df)
mask1 = grid_searcher1.best_estimator_.named_steps['selection'].get_support()
top_features1 = [x for (x, boolean) in zip(X_features, mask1) if boolean]
n_pca_components1 = grid_searcher1.best_estimator_.named_steps['reducer'].n_components_
best_params1 = grid_searcher1.best_params_
score1 = grid_searcher1.best_score_


##LinearSVC
pipeline2 = Pipeline(steps=[('minmaxer', MinMaxScaler()),
                           ('selection', SelectKBest(score_func=f_classif)),
                           ('reducer', PCA()),
                           ('classifier', LinearSVC())
                           ])

params2 = {'selection__k': [8, 10, 12, 14, 16, 18],
          'classifier__C': [1e-02, 1e-1, 1],
          'classifier__class_weight': ['balanced'],
          'classifier__tol': [1e-3, 1e-4],
          'classifier__random_state': [42],
          'reducer__n_components': [1, 2, 3],
          'reducer__whiten': [True]
          }

grid_searcher2 = GridSearchCV(pipeline2, param_grid=params2, cv=sk_fold,
                       n_jobs=-1, scoring=scoring_metric)
grid_searcher2.fit(y_df,x_df)

mask2 = grid_searcher2.best_estimator_.named_steps['selection'].get_support()
top_features2 = [x for (x, boolean) in zip(X_features, mask2) if boolean]
n_pca_components2 = grid_searcher2.best_estimator_.named_steps['reducer'].n_components_
best_params2 = grid_searcher2.best_params_
score2 = grid_searcher2.best_score_


##SVC with rbf-kernel
pipeline3 = Pipeline(steps=[('minmaxer', MinMaxScaler()),
                           ('selection', SelectKBest(score_func=f_classif)),
                           ('reducer', PCA()),
                           ('classifier', SVC())
                           ])

params3 = {'selection__k': [10, 12, 16],
          'classifier__C': [1e-02, 1e-1, 1],
          'classifier__gamma': [1e-1,0],
          'classifier__class_weight': ['balanced'],
          'classifier__tol': [1e-3, 1e-4],
          'classifier__random_state': [42],
          'classifier__kernel': ['rbf'],
          'reducer__n_components': [0.5,1,2],
          'reducer__whiten': [True]
          }


grid_searcher3 = GridSearchCV(pipeline3, param_grid=params3, cv=sk_fold,
                       n_jobs=-1, scoring=scoring_metric)
grid_searcher3.fit(y_df,x_df)

mask3 = grid_searcher3.best_estimator_.named_steps['selection'].get_support()
top_features3 = [x for (x, boolean) in zip(X_features, mask3) if boolean]
n_pca_components3 = grid_searcher3.best_estimator_.named_steps['reducer'].n_components_
best_params3 = grid_searcher3.best_params_
score3 = grid_searcher3.best_score_

##DecisionTreeCLF (no scaling needed)
pipeline4 = Pipeline(steps=[('selection', SelectKBest(score_func=f_classif)),
                           ('reducer', PCA()),
                           ('classifier', DecisionTreeClassifier())
                           ])

params4 = {'selection__k': [8, 10, 12, 14, 16, 18],
          'classifier__criterion': ['gini','entropy'],
          'classifier__splitter': ['best','random'],
          'classifier__class_weight': ['balanced'],
          'classifier__random_state': [42],
          'classifier__min_samples_split': [2,4,6,8,10],
          'reducer__n_components': [1, 2, 3],
          'reducer__whiten': [True]
          }




grid_searcher4 = GridSearchCV(pipeline4, param_grid=params4, cv=sk_fold,
                       n_jobs=-1, scoring=scoring_metric)
grid_searcher4.fit(y_df,x_df)

mask4 = grid_searcher4.best_estimator_.named_steps['selection'].get_support()
top_features4 = [x for (x, boolean) in zip(X_features, mask4) if boolean]
n_pca_components4 = grid_searcher4.best_estimator_.named_steps['reducer'].n_components_
best_params4 = grid_searcher4.best_params_
score4 = grid_searcher4.best_score_


##GaussianNB
pipeline5 = Pipeline(steps=[('minmaxer', MinMaxScaler()),
                            ('selection', SelectKBest(score_func=f_classif)),
                           ('reducer', PCA()),
                           ('classifier', GaussianNB())
                           ])

params5 = {'selection__k': [8, 10, 12, 14, 16, 18],
          'reducer__n_components': [1, 2, 3],
          'reducer__whiten': [True]
          }


grid_searcher5 = GridSearchCV(pipeline5, param_grid=params5, cv=sk_fold,
                       n_jobs=-1, scoring=scoring_metric)
grid_searcher5.fit(y_df,x_df)

mask5 = grid_searcher5.best_estimator_.named_steps['selection'].get_support()
top_features5 = [x for (x, boolean) in zip(X_features, mask4) if boolean]
n_pca_components5 = grid_searcher5.best_estimator_.named_steps['reducer'].n_components_
best_params5 = grid_searcher5.best_params_
score5 = grid_searcher5.best_score_

names = ['LinReg','LinearSVC','SVC','DecTree','NB']
scores_tmp = [score1,score2,score3,score4,score5]
scores = []
for record in scores_tmp:
    scores.append(round(record,5))
report = zip(names, scores)
pp.pprint(report)

##Save top features and F1-scores (ordered)
#Logistic Regression
scores_LR = grid_searcher1.best_estimator_.named_steps['selection'].scores_
combined_LR = zip(top_features1, scores_LR)
combined_LR.sort(reverse=True, key= lambda x: x[1])

#Linear SVC
scores_LSVC = grid_searcher2.best_estimator_.named_steps['selection'].scores_
combined_LSVC = zip(top_features2, scores_LSVC)
combined_LSVC.sort(reverse=True, key= lambda x: x[1])


#SVC
scores_SVC = grid_searcher3.best_estimator_.named_steps['selection'].scores_
combined_SVC = zip(top_features3, scores_SVC)
combined_SVC.sort(reverse=True, key= lambda x: x[1])


#Decision Tree
scores_TREE = grid_searcher4.best_estimator_.named_steps['selection'].scores_
combined_TREE = zip(top_features4, scores_TREE)
combined_TREE.sort(reverse=True, key= lambda x: x[1])

#Gaussian NB
scores_NB = grid_searcher5.best_estimator_.named_steps['selection'].scores_
combined_NB = zip(top_features5, scores_NB)
combined_NB.sort(reverse=True, key= lambda x: x[1])


y_df.insert(0, 'poi', x_df)
data_dict = y_df.T.to_dict()


### Store to my_dataset for easy export below.
my_dataset = data_dict
my_features_list = ['poi']+top_features1
pp.pprint(combined_LR)
clf = grid_searcher1.best_estimator_
dump_classifier_and_data(clf, my_dataset, my_features_list)
tester.main()