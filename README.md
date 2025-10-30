# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries.

2.Upload and read the dataset.

3.Check for any null values using the isnull() function.

4.From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.

5.Find the accuracy of the model and predict the required values by importing the required module from sklearn.
## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: GEETHU R
RegisterNumber:  212224040089
*/
```
~~~
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

data = pd.read_csv(r"C:\Users\admin\Downloads\Employee.csv")

print("OUTPUT:\nDATA HEAD:")
print(data.head(), "\n")

print("DATASET INFO:")
print(data.info(), "\n")

print("NULL DATASET:")
print(data.isnull().sum(), "\n")

print("VALUES COUNT IN THE LEFT COLUMN:")
print(data['left'].value_counts(), "\n")

le = LabelEncoder()
data['salary'] = le.fit_transform(data['salary'])

print("DATASET TRANSFORMED HEAD:")
print(data.head(), "\n")

X = data[['satisfaction_level', 'last_evaluation', 'number_project',
          'average_montly_hours', 'time_spend_company',
          'Work_accident', 'promotion_last_5years', 'salary']]
y = data['left']

print("X.HEAD:")
print(X.head(), "\n")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)
dt = DecisionTreeClassifier(criterion='entropy')
dt.fit(X_train, y_train)

y_pred = dt.predict(X_test)
accuracy = metrics.accuracy_score(y_test, y_pred)
print("ACCURACY:", accuracy, "\n")

sample = [[0.5, 0.8, 9, 260, 6, 0, 1, 2]]
prediction = dt.predict(sample)
print("DATA PREDICTION:", prediction, "\n")
~~~

## Output:
### DATA HEAD:
<img width="1266" height="282" alt="datahead" src="https://github.com/user-attachments/assets/03c13c93-73fb-46b7-a57f-0d929fda7aba" />


### DATASET INFO:
<img width="551" height="485" alt="data set" src="https://github.com/user-attachments/assets/ef6e60de-0e4a-45d6-adab-5541b67bce11" />


### NULL DATASET:
<img width="312" height="251" alt="null data set" src="https://github.com/user-attachments/assets/59a13cc7-5e1e-4b55-84c1-7e077fa7abb7" />



### VALUES COUNT IN THE LEFT COLUMN
<img width="331" height="88" alt="value count in left column" src="https://github.com/user-attachments/assets/00a09b76-394d-4cda-9d89-d6990b749db6" />


### X.HEAD
<img width="1260" height="281" alt="x head" src="https://github.com/user-attachments/assets/95a2dae8-55d8-4db4-badf-881e89ca51c1" />



### ACCURACY
<img width="1275" height="173" alt="accuracy" src="https://github.com/user-attachments/assets/2307e8ae-0945-4bf4-b466-c2032020d96c" />

### DATA PREDICTION
<img width="300" height="77" alt="data prediction" src="https://github.com/user-attachments/assets/57bccca1-8090-4dc1-b780-454e65900000" />


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
