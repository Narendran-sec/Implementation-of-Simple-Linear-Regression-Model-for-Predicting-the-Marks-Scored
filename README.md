# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Use the standard libraries in python.
2. Set variables for assigning dataset values.
3. Import LinearRegression from the sklearn.
4. Assign the points for representing the graph.
5. Predict the regression for marks by using the representation of graph.
6. Compare the graphs and hence we obtain the LinearRegression for the given datas. 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Narendran K
RegisterNumber:  212223230135
*/
```
```
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
df=pd.read_csv("student_scores.csv")
df.head()
```
```
df.head()
```
```
df.tail()
```
```
X=df.iloc[:,:-1].values
X
```

```
Y=df.iloc[:,1].values
Y
```
```
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
Y_pred
Y_test
```
```
plt.scatter(X_train,Y_train,color="skyblue")
plt.plot(X_train,regressor.predict(X_train),color="blue")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```
```
mse=mean_squared_error(Y_test,Y_pred)
print("MSE = ",mse)

mae=mean_absolute_error(Y_test,Y_pred)
print("MAE = ",mae)

rmse=np.sqrt(mse)
print("RMSE = ",rmse)
```

## Output:

![output 1](https://github.com/user-attachments/assets/f1d4adb5-d683-497b-a8df-79244d9472e1)
![output 2](https://github.com/user-attachments/assets/2872bd2f-bfd3-410e-b9f6-788614f44e74)

![output 3](https://github.com/user-attachments/assets/b4a20189-a650-4891-8664-7126f8761b45)

![output 4](https://github.com/user-attachments/assets/047e7da2-767d-4c55-ba08-e6ad4e58aee0)

![output 5](https://github.com/user-attachments/assets/b943f5ca-754d-4deb-b72c-e6cb542f4941)

![output 6](https://github.com/user-attachments/assets/bc855c4f-4128-48ad-b8ce-da35f2dc3488)

![output 7](https://github.com/user-attachments/assets/f54181f5-1c7a-4a61-a521-5ce8641db41d)

## Result:

Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
