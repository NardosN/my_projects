
# coding: utf-8



#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RandomizedLasso
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from time import time
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV
from sklearn import grid_search
from sklearn import tree
import numpy as np
from tester import test_classifier, dump_classifier_and_data
from feature_format import featureFormat
from feature_format import targetFeatureSplit


# features to be used (email address discarded)

features_list = ['poi','salary', 'deferral_payments',  \
               'total_payments', 'loan_advances', 'bonus',   \
              'restricted_stock_deferred', 'deferred_income', \
               'total_stock_value', 'expenses', 'exercised_stock_options', \
               'other', 'long_term_incentive', 'restricted_stock', 'director_fees', \
               'to_messages', 'from_poi_to_this_person', 'from_messages',  \
               'from_this_person_to_poi', 'shared_receipt_with_poi'] 


# Load the dictionary containing the dataset

data_dict = pickle.load(open("final_project_dataset.pkl", "r") )


# Data Overwiew 

total_people = data_dict.keys()
print('Total number of individuals: %d' % len(total_people))
POI_count = 0
for individual in total_people:
    POI_count += data_dict[individual]['poi']    
print('Number of Persons of Interest: %d' % POI_count)
print('Number of not Persons of Interest: %d' % (len(total_people) - POI_count))


# Feature Exploration

total_features = data_dict['CORDES WILLIAM R'].keys()
print('Every person has %d features ' %  len(total_features))

# Missing values exploration
missing_values = {}
for feature in total_features:
    missing_values[feature] = 0
for individual in total_people:
    records = 0
    for feature in total_features:
        if data_dict[individual][feature] == 'NaN':
            missing_values[feature] += 1
        else:
            records += 1
            
# Print results 
print(' Missing Values for Every Feature:')
for feature in total_features:
    print("%s: %d" % (feature, missing_values[feature])) 


# checking missing values for all financial features

financial_info_incomplete = []
for individual in data_dict.keys():  
    if data_dict[individual]['total_payments'] == 'NaN' and  \
    data_dict[individual]['total_stock_value'] == 'NaN':
        financial_info_incomplete.append(individual)
print
if len(financial_info_incomplete) > 0:
    print('Individuals with missing data for payments and stock value:')
    records = 0
    for individual in financial_info_incomplete:
        print individual        
        records += data_dict[individual]['poi']
    print('Out of these %d individual %d are POIs' % (len(financial_info_incomplete), 
          records))
else:
    print('No individual with missing data for payments and stock value.')
print


# Task 2: Remove outliers

data_dict.pop("CHAN RONNIE",0)
data_dict.pop("POWERS WILLIAM",0)
data_dict.pop("LOCKHART EUGENE E",0)
data_dict.pop("TOTAL",0)
#Remove "THE TRAVEL AGENCY IN THE PARK" as it has values only for the "Other" 
#and "Total Payments" fields.
data_dict.pop("THE TRAVEL AGENCY IN THE PARK",0) 


# Create new features

for key in data_dict:
    # a fraction of total messages received   
    fraction_shared_receipt_with_poi = float(data_dict[key]['shared_receipt_with_poi'])  \
    / float(data_dict[key]['to_messages'])    
    if np.isnan(fraction_shared_receipt_with_poi):     
        data_dict[key]['fraction_shared_receipt_with_poi'] =  0
    else:
        data_dict[key]['fraction_shared_receipt_with_poi'] =       \
        round(fraction_shared_receipt_with_poi,2)
        
    #  fraction of messages received from poi     
    fraction_from_poi = float(data_dict[key]['from_poi_to_this_person'])  \
    / float(data_dict[key]['to_messages']) 
    if np.isnan(fraction_from_poi):     
        data_dict[key]['fraction_from_poi'] =  0
    else:
        data_dict[key]['fraction_from_poi'] =  round(fraction_from_poi,2)
        
    # fraction of messages sent to poi
    fraction_to_poi = float(data_dict[key]['from_this_person_to_poi'])    \
    / float(data_dict[key]['from_messages'])
    if np.isnan(fraction_from_poi):     
        data_dict[key]['fraction_to_poi'] =  0
    else:
        data_dict[key]['fraction_to_poi'] =  round(fraction_to_poi,2)

my_dataset = data_dict


#print data_dict.keys()


#print data_dict["CORDES WILLIAM R"]


# Features Selection

features_list = ['poi','salary', 'deferral_payments',    \
              'total_payments', 'loan_advances', 'bonus',   \
              'restricted_stock_deferred', 'deferred_income', \
                 'total_stock_value', 'expenses', 'exercised_stock_options', \
                 'other', 'long_term_incentive', 'restricted_stock',   \
                 'director_fees', 'fraction_shared_receipt_with_poi',  \
                 'fraction_from_poi', 'fraction_to_poi']

data = featureFormat(my_dataset, features_list)
labels, features = targetFeatureSplit(data)

#Stability Selection

rlasso = RandomizedLasso(random_state=2)
rlasso.fit(features,labels)
scores = rlasso.scores_
print scores


for s in range(len(scores)):
    print features_list[s+1],": ",scores[s]
    
my_features_list = ['poi']
for s in np.where(scores > 0.30)[0]:
    my_features_list.append(features_list[s+1])



# features selected

print my_features_list

# Naive Bayes Classifier

#from sklearn.naive_bayes import GaussianNB
#from tester import test_classifier
#nclf= GaussianNB()
#t0 = time()
#test_classifier(nclf, data_dict, my_features_list)
#print("Naive Bayes fitting time: %rs" % round(time()-t0, 3))


# SVM
#from sklearn import svm
##svr = svm.SVC()
#clfs=SVC(kernel='linear')
#t0 = time()
#test_classifier(clfs, data_dict, my_features_list)
#print("SVM fitting time: %rs" % round(time()-t0, 3))

# KNeighbors Classifier
#from sklearn.neighbors import KNeighborsClassifier
#from tester import test_classifier
#knn = KNeighborsClassifier()
#t0 = time()
#test_classifier(knn, data_dict, my_features_list)
#print("KNeighbors fitting time: %rs" % round(time()-t0, 3))

# Random Forest Classifier

clf = RandomForestClassifier()
from time import time
from tester import test_classifier
t0 = time()
test_classifier(clf, data_dict, my_features_list)
print("Random forest fitting time: %rs" % round(time()-t0, 3))


# Decision Tree Classifier

dclf = DecisionTreeClassifier()
from time import time
from tester import test_classifier
t0 = time()
test_classifier(dclf, data_dict, my_features_list)
print("Decision Tree fitting time: %rs" % round(time()-t0, 3))


# Adaboost

abclf = AdaBoostClassifier()
t0 = time()
test_classifier(abclf, data_dict, my_features_list)
print("AdaBoost fitting time: %rs" % round(time()-t0, 3))


# Random Forest Tunning

stss = StratifiedShuffleSplit(labels, 10, test_size=0.3, random_state=42)
                                     
parameters = {'max_depth': [2,4],
              'min_samples_split':[2,3,4],
              'n_estimators':[10,20],
              'min_samples_leaf':[1,2,3],
              'criterion':('gini', 'entropy')}
t0 = time()
Rf_clf = RandomForestClassifier()
Rfgrd = GridSearchCV(Rf_clf, parameters, scoring = 'f1', cv=stss)
print("Random Forest tuning: %r" % round(time()-t0, 3))
Rfgrd.fit(features, labels)
print("Random forest fitting time: %rs" % round(time()-t0, 3))

rf = Rfgrd.best_estimator_
t0 = time()
test_classifier(rf, my_dataset, my_features_list, folds = 100)
print("Random Forest evaluation time: %rs" % round(time()-t0, 3))


# K-Nearest Neighbor Tuning

#from sklearn.grid_search import GridSearchCV
#sss = StratifiedShuffleSplit(labels, 10, test_size=0.3, random_state=42)
#k = np.arange(10)+1
#parameters = {'n_neighbors': k}       
#knn_clf = KNeighborsClassifier()
#knngrd = GridSearchCV(knn_clf, parameters, scoring = 'f1', cv=sss)
#knngrd.fit(features, labels)
#knn = knngrd.best_estimator_
#t0 = time()
#test_classifier(knn, my_dataset, my_features_list)
#print("KNeighbor evaluation time: %rs" % round(time()-t0, 3))



# Adaboost Tuning

stss = StratifiedShuffleSplit(labels, 10, test_size=0.3, random_state=42)
                                     
val = []
for i in range(5):
    val.append(DecisionTreeClassifier(max_depth=(i+1)))
parameters = {'base_estimator': val, 
             'n_estimators': [50, 100]}
t0 = time()
abclf= AdaBoostClassifier()                            
abgrd = GridSearchCV(abclf, parameters, scoring='f1', cv=stss)
print("AdaBoost tuning: %r" % round(time()-t0, 3))
t0 = time()
abgrd.fit(features, labels)
print("AdaBoost fitting time: %rs" % round(time()-t0, 3))
ab = abgrd.best_estimator_
t0 = time()
test_classifier(ab, my_dataset, my_features_list, folds = 100)
print("AdaBoost evaluation time: %rs" % round(time()-t0, 3))


# Dumping classifiers
# Adaboost selected as best classifier
clf=ab
test_classifier(clf, my_dataset, my_features_list)


dump_classifier_and_data(clf, my_dataset, my_features_list)





