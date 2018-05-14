# -*- coding: utf-8 -*-
"""
Created on Mon May 14 12:12:29 2018

@author: kimi
"""
# Import libraries & funtions
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline


# Load dataset
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
cancer.data


# Create a funtion 
def plot_feature_importances_cancer(model):
    n_features = cancer.data.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), cancer.feature_names)
    plt.xlabel("importance of features")
    plt.ylabel("features")
    plt.ylim(-1, n_features)


# Training/Testing data separation + modeling with DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
X_train, X_test, y_train, y_test = train_test_split(cancer.data, 
                                                    cancer.target, 
                                                    test_size = 0.3, 
                                                    random_state=0)


# modeling - basic model 
gbrt = GradientBoostingClassifier(random_state=0)
gbrt.fit(X_train, y_train)

print("accuracy with the train dataset: {:.3f}".format(gbrt.score(X_train, y_train))) # 1.000
print("accuracy with the test dataset: {:.3f}".format(gbrt.score(X_test, y_test)))    # 0.977

## we observe an overftting issue here. How can we handing that ? 


# Handing overfitting issue 1 - max_depth adjustment
gbrt1 = GradientBoostingClassifier(random_state=0, max_depth=1)
gbrt1.fit(X_train, y_train)

print("accuracy with the train dataset: {:.3f}".format(gbrt1.score(X_train, y_train)))  # 0.990
print("accuracy with the test dataset: {:.3f}".format(gbrt1.score(X_test, y_test)))     # 0.953


# Handing overfitting issue 2 - learning_rate adjustment
gbrt2 = GradientBoostingClassifier(random_state=0, learning_rate=0.01)
gbrt2.fit(X_train, y_train)

print("accuracy with the train dataset: {:.3f}".format(gbrt2.score(X_train, y_train)))  # 0.987
print("accuracy with the test dataset: {:.3f}".format(gbrt2.score(X_test, y_test)))     # 0.959


# check importance of features in each model
plot_feature_importances_cancer(gbrt) # using all the features
## top3 features : worst concave points, worst area, mean concave points

plot_feature_importances_cancer(gbrt1) # using some features
## top3 features: worst texture, area error, worst concave points

plot_feature_importances_cancer(gbrt2) # using only few features
## top3 features: worst concave points, mean concave points, worst perimeter

## With lower learning_rate, we can have better accuracy although we only use few features
 

# find out the most highest accuracy for test dataset
cnt = 0
l1 = []
lni = []
lnr = []
for i in range(1, 5):
    for j in range(1, 100):
        rate = j/100
        print("max_dept: ", i, ", learning_rate: ", rate)
        gbrt = GradientBoostingClassifier(random_state=0, max_depth=i, learning_rate=rate)
        gbrt.fit(X_train, y_train)
        acTest = gbrt.score(X_test, y_test)
        print("accuracy with the train dataset: {:.3f}".format(gbrt.score(X_train, y_train))) 
        print("accuracy with the test dataset: {:.3f}".format(gbrt.score(X_test, y_test)))
        lni.append(gbrt.score(X_train, y_train))
        lnr.append(gbrt.score(X_test, y_test))
        cnt += 1
        l1.append(cnt)

plt.plot(lni, "--", label="train set", color="blue")
plt.plot(lnr, "-", label="test set", color="red")
plt.legend()
plt.show()


max(lnr)  
## the most highest accuracy for test dataset is : 0.994152046784 






# draft----------------------------------
max(lni)
[val, idx] = max(lnr)       
lnr.index(max(lnr))

maxi, index = lnr[0], 0
for ind, i in enumerate(lnr):
    if i > maxi:
        maxi, index = i, ind

print("The maximum value that appears in the list is " + str(max) + ".")
print("It's index is " + str(index) + ".")

gbrt[237]
acTest[237]




max=0; numMax= 0; cnt= 0
cnt = 0
l1 = []
lni = []
lnr = []
for i in range(1, 5):
    for j in range(1, 100):
        rate = j/100
        print("max_dept: ", i, ", learning_rate: ", rate)
        gbrt = GradientBoostingClassifier(random_state=0, max_depth=i, learning_rate=rate)
        gbrt.fit(X_train, y_train)
        acTest = gbrt.score(X_test, y_test)
        print("accuracy with the train dataset: {:.3f}".format(gbrt.score(X_train, y_train))) 
        print("accuracy with the test dataset: {:.3f}".format(acTest))
        lni.append(gbrt.score(X_train, y_train))
        lnr.append(acTest)
        cnt += 1
        if max < acTest:
            max = acTest
            numMax = cnt

print(max, numMax)
plt.plot(lni, "--", label="train set", color="blue")
plt.plot(lnr, "-", label="test set", color="red")
plt.plot(numMax, max, "o")
ann = plt.annotate("is" % str(n))
plt.legend()
plt.show()




