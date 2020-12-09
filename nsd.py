import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import seaborn as sns
phishing=pd.read_csv('phishingdataset.csv')
phishing.head()
print(phishing.nunique())
X=phishing.iloc[:,1:-1]
y=phishing.iloc[:,-1]
print(X.head())
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)

#SUPPORT VECTOR MACHINE
print('SUPPORT VECTOR MACINE:')
print('________________________')
print('\n')
svc=SVC()
svc.fit(X_train,y_train)
svcpredictions=svc.predict(X_test)
plt.figure()
sns.distplot(y_test-svcpredictions)
plt.show()
print('classification report:')
print(classification_report(y_test,svcpredictions))
print('\n')
print('confusion metrix:')
print(confusion_matrix(y_test,svcpredictions))
accuracy=metrics.accuracy_score(y_test,svcpredictions)
svmaccuracyscore=accuracy*100
print("accuracy score:",svmaccuracyscore)
print('\n')
#NAIVE BAYES
print('NAIVE BAYES:')
print('_____________')
print('\n')
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)
print('classification report:')
print(classification_report(y_test,y_pred))
print('\n')
print('confusion metrix:')
print(confusion_matrix(y_test,y_pred))
accuracy=metrics.accuracy_score(y_test,y_pred)
nbaccuracyscore=accuracy*100
print("accuracy score:",nbaccuracyscore)
print('\n')
#LOGISTIC REGRESSION
print('LOGISTIC REGRESSION')
print('____________________')
print('\n')
logreg=LogisticRegression()
logreg.fit(X_train,y_train)
regpred=logreg.predict(X_test)
print('classification report:')
print(classification_report(y_test,regpred))
print('\n')
print('confusion metrix:')
print(confusion_matrix(y_test,regpred))
accuracy=metrics.accuracy_score(y_test,regpred)
regaccuracyscore=accuracy*100
print("accuracy score:",regaccuracyscore)
print('\n')
finres=pd.DataFrame({'MODEL':['SVM','NAIVE BAYES','LOGISTIC REGRESSION'],'ACCURACY SCORE':[svmaccuracyscore,nbaccuracyscore,regaccuracyscore]})
finres=finres.sort_values(by='ACCURACY SCORE',ascending=False)
print(finres)
