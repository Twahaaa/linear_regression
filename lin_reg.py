import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from Reg_class import LinearRegression
import pandas as pd
df=pd.read_csv('Salary_dataset.csv')
print(df.head())
X=df[['YearsExperience']].values
y=df['Salary'].values
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=4)

fig=plt.figure(figsize=(8,6))
plt.scatter(X,y,color="b",marker="o",s=30,label="Actual Data")

model = LinearRegression(lr=0.01, n_iteration=1000)

model.fit(X_train,y_train)
predictions =model.predict(X_test)

def mean_sq_err(y_test,predictions):
    return np.mean((y_test-predictions)**2)

mse=mean_sq_err(y_test,predictions)
print(mse)

x_range = np.linspace(np.min(X), np.max(X), 100).reshape(-1, 1)
y_range = model.predict(x_range)  

plt.plot(x_range, y_range, color='red', label="Regression Line", linewidth=2)

plt.title('Linear Regression - Years of Experience vs Salary')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend()
plt.grid(True)
plt.show()