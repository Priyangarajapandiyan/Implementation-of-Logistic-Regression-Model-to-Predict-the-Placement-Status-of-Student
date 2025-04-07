# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required packages and print the present data.

2.Print the placement data and salary data.

3.Find the null and duplicate values.

4.Using logistic regression find the predicted values of accuracy , confusion matrices.

5.Display the results. 

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: R.PRIYANGA
RegisterNumber: 212223230161 
*/

import pandas as pd
data=pd.read_csv("Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])       
data1 

x=data1.iloc[:,:-1]
x
y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:

TOP 5 ELEMENTS

![image](https://github.com/user-attachments/assets/6396c75f-679e-4ed4-81b1-cbae5ac14706)
![image](https://github.com/user-attachments/assets/602327b8-6c6f-42fc-8a4d-32c767c35de0)
![image](https://github.com/user-attachments/assets/c6779062-9136-41aa-a837-e760890039a5)


## DATA DUPILICATE

![image](https://github.com/user-attachments/assets/1924ccb2-404c-45b6-94f3-cba914bb34de)


## PRINT DATA:

![image](https://github.com/user-attachments/assets/723c9fd6-548e-46c6-8c39-58a1b829a80a)

## DATA - STATUS:

![image](https://github.com/user-attachments/assets/aca547d2-34b5-4bb2-b3dc-651022af34d0)

## Y_PREDICTION ARRAY:

![image](https://github.com/user-attachments/assets/1730cc71-65d7-4482-bd93-be81cc4e9d0a)

## CONFUSION ARRAY:

![image](https://github.com/user-attachments/assets/be72ed6f-9278-4234-9497-fb290a4335f1)

## ACCURACY VALUE:

![image](https://github.com/user-attachments/assets/c8a6fbda-614e-4ff3-b006-c715947fc6cf)


## CLASSIFICATION REPORT:

![image](https://github.com/user-attachments/assets/da2df697-e104-4678-b529-d90fc9e3d56e)


## PREDICTION OF LR:

![image](https://github.com/user-attachments/assets/87425a48-e42c-4990-936c-e732eef9cf03)




## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
