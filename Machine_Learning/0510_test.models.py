# -*- coding: utf-8 -*-
"""
Created on Fri May 11 13:31:45 2018

@author: kimi
"""

# Import libraries & funtions -------------------------------------------------
from pandas.tools.plotting import scatter_matrix

from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Load dataset ----------------------------------------------------------------
from sklearn.datasets import load_iris


# Data exploration & analysis -------------------------------------------------
iris_dataset = load_iris()
iris_dataset.keys()
iris_dataset.values()
iris_dataset.items()

iris_dataset['data']
iris_dataset['feature_names']
iris_dataset['target_names']

print('feature_names : {}'.format(iris_dataset['feature_names']))
print('target_names : {}'.format(iris_dataset['target_names']))


# From sklearn.linear_model import LogisticRegression -------------------------
X_train, X_test, y_train, y_test = train_test_split(
        iris_dataset['data'],   # input features 
        iris_dataset['target'], # output features
        test_size=0.3,          # test dataset proportion
        train_size=0.7,         # train dataset proportion 
        random_state=0)

logistics = LogisticRegression().fit(X_train, y_train)
log_pred = logistics.predict(X_test)
print("Prediction: {}".format(log_pred))

print("Accuracy of the prediction1:{:.3f}".format(np.mean(log_pred==y_test)))
print("Accuracy of the prediction2:{:.3f}".format(logistics.score(X_test, y_test)))


# From sklearn.tree import DecisionTreeClassifier -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
        iris_dataset['data'],   # input features 
        iris_dataset['target'], # output features
        test_size=0.1,          # test dataset proportion
        train_size=0.9,         # train dataset proportion 
        random_state=0)

tree = DecisionTreeClassifier().fit(X_train, y_train)
tree_pred = tree.predict(X_test)
print("Prediction: {}".format(tree_pred))

print("Accuracy of the prediction1:{:.3f}".format(np.mean(tree_pred==y_test)))
print("Accuracy of the prediction2:{:.3f}".format(tree.score(X_test, y_test)))


# from sklearn.naive_bayes import GaussianNB ----------------------------------
X_train, X_test, y_train, y_test = train_test_split(
        iris_dataset['data'],   # input features 
        iris_dataset['target'], # output features
        test_size=0.3,          # test dataset proportion
        train_size=0.7,         # train dataset proportion 
        random_state=0)

nb = GaussianNB().fit(X_train, y_train)
nb_pred = nb.predict(X_test)
print("Prediction: {}".format(nb_pred))

print("Accuracy of the prediction1:{:.3f}".format(np.mean(nb_pred==y_test)))
print("Accuracy of the prediction2:{:.3f}".format(nb.score(X_test, y_test)))


# From sklearn.svm import SVC -------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
        iris_dataset['data'],   # input features 
        iris_dataset['target'], # output features
        test_size=0.1,          # test dataset proportion
        train_size=0.9,         # train dataset proportion 
        random_state=0)

svc = SVC().fit(X_train, y_train)
svc_pred = svc.predict(X_test)
print("Prediction: {}".format(svc_pred))

print("Accuracy of the prediction1:{:.3f}".format(np.mean(svc_pred==y_test)))
print("Accuracy of the prediction2:{:.3f}".format(svc.score(X_test, y_test)))



# -----------------------------------------------------------------------------

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pd.read_csv(url, names=names)

dataset
dataset.describe()
dataset.shape
dataset.groupby('class').size()

dataset.plot(kind='box', 
             subplots=True,
             layout=(2,2),
             sharex=False,    # sharing x axis 
             sharey=False)    # sharing y axis 
plt.show()

scatter_matrix(dataset)
plt.show()

array=dataset.values
array
type(array)
X=array[:, 0:4]
Y=array[:, 4]

validation_size = 0.10 # valid data partition
seed = 7
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, Y,
 test_size=validation_size, random_state=seed)

models = []
models.append(('LogisticReg', LogisticRegression())) # () tuple
models.append(('KNN', KNeighborsClassifier()))
models.append(('TREE_CART', DecisionTreeClassifier()))
models.append(('NaiveBayes', GaussianNB()))
models.append(('SVM', SVC()))
models


scoring = 'accuracy'
results = [] # cv results
names = [] # model names
seed = 7
for name, model in models:  # models = [name of model, model function]
   kfold = model_selection.KFold(n_splits=10, random_state=seed)
   cv_results = model_selection.cross_val_score(model, X_test, y_test, cv=kfold, scoring=scoring) # cross validation
   results.append(cv_results)
   names.append(name)
   msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
   print(msg)