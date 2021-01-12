import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# read data from csv file.
dataSet=pd.read_csv("train_set.csv")
testSet=pd.read_csv("test_set.csv")


# to find the no of missig data
dataSet.isna().sum()
testSet.isna().sum()


# Convert categorical data into numerical data

for i in ["LoanAmount","Loan_Amount_Term"]:    
    dataSet[i].fillna(int(dataSet[i].mean()),inplace=True)
    testSet[i].fillna(int(dataSet[i].mean()),inplace=True)
for i in ["Gender","Married","Dependents","Self_Employed","Credit_History"]:
    dataSet[i].fillna(dataSet[i].value_counts().index[0],inplace=True)
    testSet[i].fillna(dataSet[i].value_counts().index[0],inplace=True)


dataSet["Dependents"].replace({"3+":3},inplace=True)
testSet["Dependents"].replace({"3+":3},inplace=True)

# Seperate the data from results
X=dataSet.iloc[:,1:12].values
X_test=dataSet.iloc[:,1:12].values

Y=dataSet.iloc[:,-1].values
Y_test=dataSet.iloc[:,-1].values


# Encoding Categorical data
from sklearn.preprocessing import LabelEncoder ,OneHotEncoder
from sklearn.compose import ColumnTransformer


labalEncoder=LabelEncoder()
for i in [0,1,3,4,10]:
    X[:,i]=labalEncoder.fit_transform(X[:,i])
    X_test[:,i]=labalEncoder.fit_transform(X_test[:,i])
    
ct = ColumnTransformer([("Property_Area", OneHotEncoder(), [10])], remainder = 'passthrough')

X=ct.fit_transform(X)
X_test=ct.fit_transform(X_test)
X=X[:,1:]
X_test=X_test[:,1:]
Y=labalEncoder.fit_transform(Y)
Y_test=labalEncoder.fit_transform(Y_test)



# Performing BAckward Elimination
import statsmodels.api as sm
X=np.append(arr=np.ones((614,1)).astype(float),values=X,axis=1)
X_opt=np.array(X[:,[0,1,2,4,6,9,10,11]],dtype=float)
regression_OLS=sm.OLS(endog=Y,exog=X_opt,dtype=float).fit()
print(regression_OLS.summary())

# Performing RandomForest Cassification
from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=500,criterion="entropy",random_state=0)

classifier.fit(X_opt,Y)

Y_pred=classifier.predict(X_opt)

#Checking the precidicted values
from sklearn.metrics import confusion_matrix as cm
cm(Y,Y_pred)



