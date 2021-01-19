#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 12:46:17 2020

@author: PiyushChopra
"""
# =============================================================================
# TASKS TO BE COMPLETED. 
# #1.How well should the department expect your model to identify defaults?
# #2.How confident are you the model will work well in implementation (new data)?
# #3.Briefly explain the effect of each variable in your final model on credit risk.
# #4.Please provide the probabilities of default (probability of class = 1) in a csv file as a new column labeled PD.
# 
# =============================================================================

# =============================================================================
# k-Nearest Neighbors
# Naive Bayes classifiers
# Support Vector Machines
# Decision Trees
# Random Forests
# =============================================================================


import pandas as pd
import numpy as np
import seaborn as sb
from sklearn import preprocessing
import matplotlib.pyplot as plt
plt.rc("font",size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sb
sb.set(style="white")
sb.set(style="whitegrid",color_codes=True)
pd.set_option('display.max_columns', 500)
from pylab import rcParams
rcParams['figure.figsize'] = 10, 8
sb.set_style('whitegrid')



# =============================================================================
# Data
# =============================================================================

loan_application=pd.read_csv(r'/Users/PiyushChopra/Desktop/TCF Bank/interview-exercise/loan_application.csv',usecols = ['y','x1','x2','x3','x4','x5','x6'])
loan_application.head()

loan_application.info()
loan_application.describe()
df1=loan_application.copy()

# =============================================================================
# Data Analysis
# =============================================================================
df1['y'].value_counts()
sns.countplot(x='y',data=df1,palette='hls')


df1['x1'].value_counts()
sns.countplot(x='x2',data=df1,palette='hls')


## Check Means across Y variable -
df1.groupby('y').mean()
df1.groupby('x1').mean()


pd.crosstab(df1.y,df1.x2).plot(kind='bar')
plt.title('Purchase Frequency for Job Title')
plt.xlabel('Job')
plt.ylabel('Frequency of Purchase')



sb.boxplot(x='y', y='x2', data=df1, palette='hls')


sb.heatmap(df1.corr()) 


sb.pairplot(df1)
plt.hist(df1['x6'])

corr_df=df1.corr().round(2)
max_corr = 0.4
plt.figure(figsize=(8,5))
mask = np.zeros_like(corr_df)
mask[np.triu_indices_from(mask)] = True
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
    f, ax = plt.subplots(figsize=(7, 5))
    ax = sns.heatmap(corr_df, mask=mask,vmax=max_corr, square=True, annot=True, cmap="YlGnBu")



### ROC Curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

#Implement Model.
import statsmodels.api as sm
logit_model=sm.Logit(y_train,X_train)
result=logit_model.fit()
print(result.summary2())

y_pred=result.predict(X_test)
y_pred_train=result.predict(X_train)

fpr, tpr, thresholds = roc_curve(y_test, y_pred)
# To get roc stats in df
roc_df=pd.DataFrame({'thresholds': thresholds,'tpr':tpr,'fpr':fpr})

# Plotting the ROC curve
plt.figure(figsize=(12,8))
plt.plot(fpr, tpr,marker="o")
plt.plot([0,1],[0,1])
plt.xlim(0,1)
plt.ylim(0,1.05)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive rate")
plt.title("Receiver Operatinng Characteristics")
plt.legend(loc="lower right")
plt.show()

roc_auc_score(y_test, y_pred)
roc_auc_score(y_train, y_pred_train)

# =============================================================================
# Test Codes to try out
# =============================================================================

titanic.isnull().sum()

#Recursive Feature Elimination
data_final_vars=data_final.columns.values.tolist()
y=['y']
X=[i for i in data_final_vars if i not in y]
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
rfe = RFE(logreg, 20)
rfe = rfe.fit(os_data_X, os_data_y.values.ravel())
print(rfe.support_)
print(rfe.ranking_)


#Implement Model.
import statsmodels.api as sm
logit_model=sm.Logit(y,X)
result=logit_model.fit()
print(result.summary2())


# =============================================================================
# ### Sklearn - Logistic Regression Model Fitting
# 
# =============================================================================

#Logistic Regression Model Fitting
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X=df1.loc[:, df1.columns != "y"]
y=df1.loc[:, df1.columns == "y"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
logreg = LogisticRegression(solver='liblinear',C=10.0)
logreg.fit(X_train, y_train)
predictions = logreg.predict(X_test)
#Calculate performance metrics
from sklearn.metrics import classification_report
print(classification_report(y_test, predictions))
#Generate a confusion matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, predictions))



logreg.classes_
logreg.intercept_

logreg.coef_
logreg.score(X_train, y_train)

# Evaluate the model
p_pred = logreg.predict_proba(X_test)
y_pred = logreg.predict(X_test)
score_ = logreg.score(X_train, y_train)
conf_m = confusion_matrix(y_test, predictions)
report = classification_report(y_test, predictions)


### Confusion Matrix Plot
cm = confusion_matrix(y_test, predictions)
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(cm)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='red')
plt.show()

from statsmodels.stats.outliers_influence import variance_inflation_factor

def calculate_vif(X):
  vif = pd.DataFrame()
  vif["variables"] = X.columns
  vif["VIF"] = [variance_inflation_factor(X.values, i) for i in  range(X.shape[1])]
  return(vif)
X1 = df1.loc[:, df1.columns != "y"]
calculate_vif(X1)


# =============================================================================
# ### statsmodels - Logistic Regression Model Fitting
# 
# =============================================================================
import statsmodels.api as sm
from statsmodels.formula.api import logit

X = sm.add_constant(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

formula=()
model = sm.Logit(y_train, X_train)
result = model.fit(method='newton')
result.params
result.predict(X_train)
(result.predict(X_train) >= 0.5).astype(int)
result.pred_table()
result.summary()
result.summary2()

# Fetching the statistics
stat_df=pd.DataFrame({'coefficients':result.params, 'p-value': result.pvalues, 'odds_ratio': np.exp(result.params)})
# Condition for significant parameters
significant_params=stat_df[stat_df['p-value']<=0.05].index
#significant_params= significant_params.drop('const')
significant_params

print('Total number of parameters: %s '%len(X.keys()) )
print('Number of Significant Parameters: %s'%(len(significant_params)))
stat_df.loc[significant_params].sort_values('odds_ratio', ascending=False)['odds_ratio']


model.summary()
AME=result.get_margeff(at='overall',method='dydx')
AME.summary()



### Scaled data

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)



# =============================================================================
# Trying out Random Forest. 
#
# =============================================================================

from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics



X=df1.loc[:, df1.columns != "y"]
y=df1.loc[:, df1.columns == "y"]
# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) # 70% training and 30% test
#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=100,bootstrap=True,max_features='sqrt')
#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)


# Limit depth of tree to 3 levels
rf_small = RandomForestClassifier(n_estimators=10, max_depth = 3)
rf_small.fit(X_train, y_train)
# Extract the small tree
tree_small = rf_small.estimators_[5]
# Save the tree as a png image
export_graphviz(tree_small, out_file = 'small_tree.dot', feature_names = list(X_train.columns), rounded = True, precision = 1)
(graph, ) = pydot.graph_from_dot_file('small_tree.dot')
# Write graph to a png file
graph.write_png('tree.png')

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


clf=RandomForestClassifier(n_estimators=100,bootstrap=True,max_features='sqrt',max_depth = 3)
#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)

# Extract single tree
estimator = clf.estimators_[5]

from sklearn.tree import export_graphviz
# Export as dot file
export_graphviz(estimator, out_file='tree.dot', 
                feature_names = list(X_train.columns),
                class_names = list(["0","1"]),
                rounded = True, proportion = False, 
                precision = 2, filled = True)
from subprocess import call
call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])
from IPython.display import Image
Image(filename = 'tree.png')

(graph, ) = pydot.graph_from_dot_file('small_tree.dot')
# Write graph to a png file
graph.write_png('tree.png')



from treeinterpreter import treeinterpreter as ti
prediction, bias, contributions = ti.predict(estimator, X_test[6:7])
N = 7 # no of entries in plot , 4 ---> features & 1 ---- class label
class0 = []
class1 = []
virginica = []
for j in range(2):
    list_ =  [class0,class1]
    for i in range(6):
        val = contributions[0,i,j]
        list_[j].append(val)
class0.append(prediction[0,0]/5)
class1.append(prediction[0,1]/5)
fig, ax = plt.subplots()
ind = np.arange(N)   
width = 0.15        
p1 = ax.bar(ind, class0, width, color='red', bottom=0)
p2 = ax.bar(ind+width, class1, width, color='green', bottom=0)
ax.set_title('Contribution of all feature for a particular \n sample of flower ')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(col, rotation = 90)
ax.legend((p1[0], p2[0]), ('Class0', 'Class1') , bbox_to_anchor=(1.04), loc="upper left")
ax.autoscale_view()
plt.show()



# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# Actual class predictions
rf_predictions = clf.predict(X_test)
# Probabilities for each class
rf_probs = clf.predict_proba(X_test)[:, 1]
roc_value=roc_auc_score(y_test,rf_probs)


## Feature Importance
feature_imp = pd.Series(clf.feature_importances_,index=['x1', 'x2', 'x3', 'x4', 'x5', 'x6']).sort_values(ascending=False)
feature_imp


%matplotlib inline
# Creating a bar plot
sns.barplot(x=feature_imp, y=feature_imp.index)
# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.legend()
plt.show()


n_nodes = []
max_depths = []

# Stats about the trees in random forest
for ind_tree in clf.estimators_:
    n_nodes.append(ind_tree.tree_.node_count)
    max_depths.append(ind_tree.tree_.max_depth)
    
print(f'Average number of nodes {int(np.mean(n_nodes))}')
print(f'Average maximum depth {int(np.mean(max_depths))}')

# Training predictions (to demonstrate overfitting)
train_rf_predictions = clf.predict(X_train)
train_rf_probs = clf.predict_proba(X_train)[:, 1]

# Testing predictions (to determine performance)
rf_predictions = clf.predict(X_test)
rf_probs = clf.predict_proba(X_test)[:, 1]

from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Plot formatting
plt.style.use('fivethirtyeight')
plt.rcParams['font.size'] = 18

def evaluate_model(predictions, probs, train_predictions, train_probs):
    """Compare machine learning model to baseline performance.
    Computes statistics and shows ROC curve."""
    
    baseline = {}
    
    baseline['recall'] = recall_score(y_test, 
                                     [1 for _ in range(len(y_test))])
    baseline['precision'] = precision_score(y_test, 
                                      [1 for _ in range(len(y_test))])
    baseline['roc'] = 0.5
    
    results = {}
    
    results['recall'] = recall_score(y_test, predictions)
    results['precision'] = precision_score(y_test, predictions)
    results['roc'] = roc_auc_score(y_test, probs)
    
    train_results = {}
    train_results['recall'] = recall_score(y_train, train_predictions)
    train_results['precision'] = precision_score(y_train, train_predictions)
    train_results['roc'] = roc_auc_score(y_train, train_probs)
    
    for metric in ['recall', 'precision', 'roc']:
        print(f'{metric.capitalize()} Baseline: {round(baseline[metric], 2)} Test: {round(results[metric], 2)} Train: {round(train_results[metric], 2)}')
    
    # Calculate false positive rates and true positive rates
    base_fpr, base_tpr, _ = roc_curve(y_test, [1 for _ in range(len(y_test))])
    model_fpr, model_tpr, _ = roc_curve(y_test, probs)

    plt.figure(figsize = (8, 6))
    plt.rcParams['font.size'] = 16
    
    # Plot both curves
    plt.plot(base_fpr, base_tpr, 'b', label = 'baseline')
    plt.plot(model_fpr, model_tpr, 'r', label = 'model')
    plt.legend();
    plt.xlabel('False Positive Rate'); 
    plt.ylabel('True Positive Rate'); plt.title('ROC Curves');
    plt.show();

evaluate_model(rf_predictions, rf_probs, train_rf_predictions, train_rf_probs)
plt.savefig('roc_auc_curve.png')

from sklearn.metrics import confusion_matrix
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Oranges):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Source: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    # Plot the confusion matrix
    plt.figure(figsize = (10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, size = 24)
    plt.colorbar(aspect=4)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, size = 14)
    plt.yticks(tick_marks, classes, size = 14)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    
    # Labeling the plot
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), fontsize = 20,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
        
    plt.grid(None)
    plt.tight_layout()
    plt.ylabel('True label', size = 18)
    plt.xlabel('Predicted label', size = 18)

# Confusion matrix
cm = confusion_matrix(test_labels, rf_predictions)
plot_confusion_matrix(cm, classes = ['Poor Health', 'Good Health'],
                      title = 'Health Confusion Matrix')







# =============================================================================
# Feature Selection
# =============================================================================

from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier

X_new = SelectKBest(chi2, k=4).fit_transform(X, y)
X_new.shape

lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, y)
log_reg1 = LogisticRegression(C=0.1, penalty="l2").fit(X, y)

model = SelectFromModel(lsvc, prefit=True)
model = SelectFromModel(log_reg1, prefit=True)

X_new = model.transform(X)
X_new.shape



clf_1= ExtraTreesClassifier(n_estimators=50)
clf_1 = clf_1.fit(X, y)
clf_1.feature_importances_ 
model = SelectFromModel(clf_1, prefit=True)

X_new = model.transform(X)
X_new.shape




##################################################################################################################################################################################################################

cv_score = cross_val_score(LogisticRegression(), 
                            X_train_numerical, y_train_numerical,
                            scoring = 'accuracy',
                            cv = 3,
                            n_jobs = -1,
                            verbose = 1)
cv_score

##################################################################################################################################################################################################################


X_train, X_test, y_train, y_test = train_test_split(
    X_train_selected, y_train_original, test_size=0.2)
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
clf.score(X_test, y_test)

param_grid = {
    'n_estimators': [10, 20, 30],
    'max_depth': [6, 10, 20, 30]
}
gridsearch = GridSearchCV(RandomForestClassifier(n_jobs = -1), 
                          param_grid=param_grid, 
                          scoring='accuracy', cv=3, 
                          return_train_score=True, verbose=10)
gridsearch.fit(X_train, y_train)
RandomForestClassifier().get_params().keys()
pd.DataFrame(gridsearch.cv_results_).sort_values( \
                                         by='rank_test_score')

##################################################################################################################################################################################################################

f, axes = plt.subplots(2,2, figsize=(7, 7), sharex=True)
sb.distplot( df1["x2"] , color="skyblue", ax=axes[0, 0])
sb.distplot( df1["x3"] , color="olive", ax=axes[0, 1])
sb.distplot( df1["x4"] , color="gold", ax=axes[1, 0])
sb.distplot( df1["x5"] , color="teal", ax=axes[1, 1])






