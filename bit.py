import numpy as np
import pandas as pd
import pylab as pl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


df = pd.read_csv('bitcoin.csv') #Load our data
print(df.head)
cdf = df.drop('Date', axis=1) #Drop the date column (axis=1 means column)
print(cdf.columns)

X = df[['Open', 'High', 'Low', 'Close']] #Select x features
y= df['Volume'] #Select our target variable

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)  #Train/test
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
x_scaled = scaler.transform(X_test)

#Train linear model
model = LinearRegression()
model.fit(X_scaled, y_train)

#Predict
y_pred = model.predict(x_scaled)

#Evaluate
print('MSE: ', mean_squared_error(y_test, y_pred))
print('R2: %.2f', r2_score(y_test, y_pred))

#Model coefficients
print("Coefficients: ", model.coef_)
print("Intercept: ", model.intercept_)

#scatter plot: actual vs predicted
plt.figure(figsize=(8, 8))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Volume")
plt.ylabel("Predicted Volume")
plt.title("Actual vs Predicted Volume (Scatter Plot)")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # Perfect prediction line
plt.grid(True)
plt.show()

#Plots
plt.figure(figsize=(14, 6))  # Wider for better visibility
plt.plot(y_test.values, label='Actual Volume', marker='o', linestyle='-', alpha=0.7)
plt.plot(y_pred, label='Predicted Volume', marker='x', linestyle='--', alpha=0.7)
plt.legend()
plt.title('Actual vs Predicted Volume (Entire Test Set)')
plt.xlabel('Sample Index')
plt.ylabel('Volume')
plt.grid(True)
plt.tight_layout()
plt.show()