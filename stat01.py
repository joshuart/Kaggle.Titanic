#######################################################################################################################
#Program: stat01.py
#Project: Titanic
#Author: Josh Taylor
#Last Edited: 7/1/15
#######################################################################################################################

import pandas as pd
import numpy as np
import statsmodels.formula.api as sm
from sklearn.ensemble import RandomForestClassifier
import math



#path to this file: "/Volumes/KINGSTON/Competitive Data Science/Kaggle/Titanic/stat01.py"

#Impliment a logit model

train = pd.read_csv("/Volumes/KINGSTON/Competitive Data Science/Kaggle/Titanic/train.csv")

test = pd.read_csv("/Volumes/KINGSTON/Competitive Data Science/Kaggle/Titanic/test.csv")


#TODO: fix this

medianAgeC1Train = np.median(train['Age'][train['Pclass'] == 1])

medianAgeC2Train = np.median(train['Age'][train['Pclass'] == 2])

medianAgeC3Train = np.median(train['Age'][train['Pclass'] == 3])

for i in range(train.shape[0]):
    if math.isnan(train.loc[i, 'Age']) and train.loc[i, 'Pclass'] == 1:
        train.loc[i,'Age'] = medianAgeC1Train
    elif math.isnan(train.loc[i, 'Age']) and train.loc[i, 'Pclass'] == 2:
        train.loc[i, 'Age'] = medianAgeC2Train
    elif math.isnan(train.loc[i, 'Age']) and train.loc[i, 'Pclass'] == 3:
        train.loc[i, 'Age'] = medianAgeC3Train


train['SibSpSq'] = train['SibSp']**2

logitTrain = sm.logit(formula="Survived ~ C(Sex) + Age + C(Pclass) + SibSp + SibSpSq", data = train).fit()

train['prob'] = np.exp(logitTrain.predict())/(1 + np.exp(logitTrain.predict()))
train['predSurvival'] = (train['prob'] > .6) + 0
tab = pd.crosstab(train['predSurvival'], train['Survived'])


logitBetas = logitTrain.params.as_matrix()

#Analyze the test data

medianAgeC1Test = test[test['Pclass'] == 1]['Age'].dropna().median()

medianAgeC2Test = test[test['Pclass'] == 2]['Age'].dropna().median()

medianAgeC3Test = test[test['Pclass'] == 3]['Age'].dropna().median()


for i in range(test.shape[0]):
    if math.isnan(test.loc[i, 'Age']) and test.loc[i, 'Pclass'] == 1:
        test.loc[i,'Age'] = medianAgeC1Test
    elif math.isnan(test.loc[i, 'Age']) and test.loc[i, 'Pclass'] == 2:
        test.loc[i, 'Age'] = medianAgeC2Test
    elif math.isnan(test.loc[i, 'Age']) and test.loc[i, 'Pclass'] == 3:
        test.loc[i, 'Age'] = medianAgeC3Test


test['SibSpSq'] = test['SibSp']**2


test['constant'] = 1
test['c2'] = (test['Pclass'] == 2) + 0
test['c3'] = (test['Pclass'] == 3) + 0
test['male'] = (test['Sex'] == 'male') + 0
test['pLogit'] = np.dot(test[['constant','male','c2', 'c3', 'Age', 'SibSp', 'SibSpSq']].as_matrix(), logitBetas)
test['prob'] = np.exp(test['pLogit']) / (1 + np.exp(test['pLogit']))
test['pred'] = (test['prob'] > .5) + 0


#Impliment a random forest

