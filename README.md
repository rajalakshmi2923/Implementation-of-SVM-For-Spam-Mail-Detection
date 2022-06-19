# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1. Import the necessary packages using import statement.
2. Read the given csv file and print the number of contents to be displayed.
3. Split the dataset using train_test_split.
4. Calculate Y_Pred and accuracy.
5. Display the result. 

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Rajalakshmi R
RegisterNumber: 212219040116   
*/
import pandas as pd
data=pd.read_csv("spam.csv",encoding='latin-1')
data.head()
data.info()
data.isnull().sum()
x=data["EmailText"].values
y=data["Label"].values
from sklearn.model_selection import train_test_split 
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output:
### Dataset:
![Screenshot (58)](https://user-images.githubusercontent.com/87656716/174479545-7c7bf717-94b4-4da7-9cdb-3422c3a2e2ec.png)

### Dataset Information:
![Screenshot (60)](https://user-images.githubusercontent.com/87656716/174479674-651d04d1-9434-4017-80d0-16a989b33dc0.png)

### Null Dataset:
![Screenshot (62)](https://user-images.githubusercontent.com/87656716/174479962-9cb39255-4235-43dd-be12-f0613148e0dd.png)

### Detected Spam:
![Screenshot (64)](https://user-images.githubusercontent.com/87656716/174480214-45adda76-e3ba-4d53-bc55-78584ed2a602.png)

### Accuracy score:
![Screenshot (66)](https://user-images.githubusercontent.com/87656716/174480286-0ef4401d-91d7-4614-b288-c4444346b036.png)



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
