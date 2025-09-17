# BLENDED_LEARNING
# Implementation-of-Linear-and-Polynomial-Regression-Models-for-Predicting-Car-Prices

## AIM:
To write a program to predict car prices using Linear Regression and Polynomial Regression models.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Data Collection: Import essential libraries like pandas, numpy, sklearn, matplotlib, and seaborn. Load the dataset using pandas.read_csv().
2. Data Preprocessing: Address any missing values in the dataset. Select key features for training the models. Split the dataset into training and testing sets with train_test_split().
3. Linear Regression: Initialize the Linear Regression model from sklearn. Train the model on the training data using .fit(). Make predictions on the test data using .predict(). Evaluate model performance with metrics such as Mean Squared Error (MSE) and the R² score.

4. Polynomial Regression: Use PolynomialFeatures from sklearn to create polynomial features. Fit a Linear Regression model to the transformed polynomial features. Make predictions and evaluate performance similar to the linear regression model.
5. Visualization: Plot the regression lines for both Linear and Polynomial models. Visualize residuals to assess model performance.

## Program:
```
/*
Program to implement Linear and Polynomial Regression models for predicting car prices.
Developed by: SUMITH M
RegisterNumber: 212224230279
import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
df=pd.read_csv('CarPrice_Assignment.csv')
x=df[['enginesize','horsepower','citympg','highwaympg']]
y=df['price']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.transform(x_test)
model=LinearRegression()
model.fit(x_train_scaled,y_train)
y_pred=model.predict(x_test_scaled)
print("="*50)
print("MODEL COEFFICIENTS:")
for feature, coef in zip(x.columns, model.coef_):
    print(f"{feature:>12}: {coef:>10.2f}")
print(f"{'Intercept':>12}:{model.intercept_:>10.2f}")
print("="*50)

print("\nMODEL PERFORMANCE:")
mse=mean_squared_error(y_test, y_pred)
print('Mean squared error=',mse)
rmse=np.sqrt(mse)
print('Root mean squared error=',rmse)
r2score=r2_score(y_test, y_pred)
print('R-squared=',r2score)

# 1. Linearity check
plt.figure(figsize=(10,5))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y.min(),y.max()],[y.min(),y.max()],'r--')
plt.title("Linearity check: Actual vs Predicted prices")
plt.xlabel("Actual Price($)")
plt.ylabel("Predicted Price($)")
plt.grid(True)
plt.show()

# 2. Independence (Durbin-Watson)
residuals=y_test-y_pred
dw_test=sm.stats.durbin_watson(residuals)
print(f"\nDurbin-Watson Statistic:{dw_test:.2f}","\n(Values close to 2 indicate no autocorrelation)")

# 3. Homoscedasticity
plt.figure(figsize=(10,5))
sns.residplot(x=y_pred, y=residuals, lowess=True, line_kws={'color':'red'})
plt.title("Homoscedasticity Check: Residuals vs Predicted")
plt.xlabel("Predicted Price ($)")
plt.ylabel("Residuals ($)")
plt.grid(True)
plt.show()

# 4. Normality of residuals
fig, (ax1,ax2)=plt.subplots(1,2,figsize=(12,5))
sns.histplot(residuals, kde=True, ax=ax1)
ax1.set_title("Residuals Distribution")
sm.qqplot(residuals, line='45', fit=True, ax=ax2)
ax2.set_title("Q-Q Plot")
plt.tight_layout()
plt.show()



*/
```

## Output:


<img width="497" height="298" alt="image" src="https://github.com/user-attachments/assets/74ada7c0-f22d-479f-85c4-4d75c336afcc" />
<img width="1258" height="645" alt="image" src="https://github.com/user-attachments/assets/2a20ba8d-ac50-4903-99bc-4c989fa68e02" />
<img width="1217" height="627" alt="image" src="https://github.com/user-attachments/assets/53583d1e-5304-4ab2-b875-bcd0426998c2" />
<img width="1488" height="625" alt="image" src="https://github.com/user-attachments/assets/a6202e80-f872-4c3a-bc46-fcefd48bc5e5" />







## Result:
Thus, the program to implement Linear and Polynomial Regression models for predicting car prices was written and verified using Python programming.
