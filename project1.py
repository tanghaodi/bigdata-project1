import numpy as np
import sklearn as sl
import pandas as pd
import time 
from sklearn import tree 
from sklearn.svm import SVC 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.neural_network import MLPClassifier 
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score 
time_start=time.time()
print('Start learning!!!')
col_names = ['gameId','creationTime','gameDuration','seasonId','winner','firstBlood','firstTower','firstInhibitor','firstBaron','firstDragon','firstRiftHerald','t1_towerKills','t1_inhibitorKills','t1_baronKills','t1_dragonKills','t1_riftHeraldKills','t2_towerKills','t2_inhibitorKills','t2_baronKills','t2_dragonKills','t2_riftHeraldKills'] 
lol = pd.read_csv("test_set.csv", header=None, names=col_names)
lol = lol.iloc[1:]
lol.head() 
feature_cols = ['firstBlood','firstTower','firstInhibitor','firstBaron','firstDragon','firstRiftHerald','t1_towerKills','t1_inhibitorKills','t1_baronKills','t1_dragonKills','t1_riftHeraldKills','t2_towerKills','t2_inhibitorKills','t2_baronKills','t2_dragonKills','t2_riftHeraldKills']
trainset = lol[feature_cols]
trainlabel= lol.winner

Col_names = ['gameId','creationTime','gameDuration','seasonId','winner','firstBlood','firstTower','firstInhibitor','firstBaron','firstDragon','firstRiftHerald','t1_towerKills','t1_inhibitorKills','t1_baronKills','t1_dragonKills','t1_riftHeraldKills','t2_towerKills','t2_inhibitorKills','t2_baronKills','t2_dragonKills','t2_riftHeraldKills'] 
LoL = pd.read_csv("new_data.csv", header=None, names=col_names)
LoL = LoL.iloc[1:]
LoL.head() 
Feature_cols = ['firstBlood','firstTower','firstInhibitor','firstBaron','firstDragon','firstRiftHerald','t1_towerKills','t1_inhibitorKills','t1_baronKills','t1_dragonKills','t1_riftHeraldKills','t2_towerKills','t2_inhibitorKills','t2_baronKills','t2_dragonKills','t2_riftHeraldKills']
testset = LoL[feature_cols]
testlabel = LoL.winner
clsDT = tree.DecisionTreeClassifier(criterion='gini', splitter='random', max_depth=15, min_samples_split=2)
clsSVM = SVC(kernel='rbf',  gamma='auto')
clsKNN = KNeighborsClassifier (n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=16, p=2) 
clsMLP = MLPClassifier(hidden_layer_sizes=(10,10),activation='relu',batch_size='auto', learning_rate_init=0.001, max_iter=1500,early_stopping =True, tol=0.00001, n_iter_no_change=20) 
clsEN = VotingClassifier (estimators=[('dt',clsDT),('svm',clsSVM) ,('knn',clsKNN) , ('mlp',clsMLP) ], voting='hard') 
a1=time.time()
clsDT.fit(trainset, trainlabel)
b1=time.time()
print('Training time for Decision Tree classifier is',b1-a1,'seconds')
a2=time.time()
clsSVM.fit(trainset, trainlabel)
b2=time.time()
print('Training time for Support Vector Machine classifier is',b2-a2,'seconds')
a3=time.time()
clsKNN.fit(trainset, trainlabel) 
b3=time.time()
print('Training time for K nearest neighbor classifier is',b3-a3,'seconds')
a4=time.time()
clsMLP.fit(trainset, trainlabel)
b4=time.time()
print('Training time for Multi-Layer Perceptron classifier is',b4-a4,'seconds')
a5=time.time()
clsEN.fit(trainset, trainlabel)
b5=time.time()
print('Training time for the Ensemble(voting) classifier is',b5-a5,'seconds')
pred1 = clsDT.predict(testset) 
pred2 = clsSVM.predict(testset) 
pred3 = clsKNN.predict(testset)
pred4 = clsMLP.predict(testset)
pred5 = clsEN.predict(testset)
A1=accuracy_score(testlabel, pred1) 
A2=accuracy_score(testlabel, pred2) 
A3=accuracy_score(testlabel, pred3) 
A4=accuracy_score(testlabel, pred4) 
A5=accuracy_score(testlabel, pred5) 
print('The accuracy of the Decision Tree classifier is',A1)
print('The accuracy of the Support Vector Machine classifier is',A2)
print('The accuracy of the K nearest neighbour classifier is',A3)
print('The accuracy of the Multi-Layer Perceptron classifier is',A4)
print('The accuracy of the Ensemble(voting) classifier is',A5)
time_end=time.time()
print('time cost',time_end-time_start,'seconds')


