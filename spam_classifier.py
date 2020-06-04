import pandas as pd
import numpy as np

df = pd.read_table('../Datasets/sms.tsv', header=None, names=['label','message'])

# print(df.head())

x = df.message
y = df.label

from sklearn.model_selection import train_test_split

X_train,X_test, y_train, y_test = train_test_split(x, y, random_state=16)

##print(X_train.shape)
##print(X_test.shape)
##print(y_train.shape)
##print(y_test.shape)


from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()
cv.fit_transform(X_train)

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
X_train_dtm = cv.transform(X_train)
X_test_dtm = cv.transform(X_test)
# Logistic Regression Model
lr = LogisticRegression()
lr.fit(X_train_dtm, y_train)
y_pred_lr = lr.predict(X_test_dtm)
print(metrics.accuracy_score(y_test, y_pred_lr))

# SVM classifier
from sklearn.svm import SVC

svc = SVC()
svc.fit(X_train_dtm, y_train)
y_pred_svc = svc.predict(X_test_dtm)
print(metrics.accuracy_score(y_test, y_pred_svc))




























