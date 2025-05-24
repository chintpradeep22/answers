import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

df = pd.read_csv("housing.csv")

df.fillna(df.mean(numeric_only=True), inplace=True)
non_numeric_cols = df.select_dtypes(exclude=['int64', 'float64']).columns


label_encoder = LabelEncoder()
for col in non_numeric_cols:
    df[col] = label_encoder.fit_transform(df[col].astype(str))

X = df.drop("MEDV",axis=1)
y = df["MEDV"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=50)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


#Linear Regression

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n🔹 Linear Regression Results:")
print("MSE:", mse)
print("RMSE:", rmse)
print("R² Score:", r2)

#Lasso 
from sklearn.linear_model import Lasso

lasso_model = Lasso()
lasso_model.fit(X_train, y_train)
lass_y_pred = lasso_model.predict(X_test)

lass_mse = mean_squared_error(y_test, lass_y_pred)
lass_rmse = np.sqrt(lass_mse)
lass_r2 = r2_score(y_test, lass_y_pred)

print("\n🔹 Lasso Regression Results:")
print("MSE:", lass_mse)
print("RMSE:", lass_rmse)
print("R² Score:", lass_r2)

#Ridge

from sklearn.linear_model import Ridge

rid_model = Ridge()
rid_model.fit(X_train, y_train)
rid_y_pred = rid_model.predict(X_test)

rid_mse = mean_squared_error(y_test, rid_y_pred)
rid_rmse = np.sqrt(rid_mse)
rid_r2 = r2_score(y_test, rid_y_pred)

print("\n🔹 Ridge Regression Results:")
print("MSE:", rid_mse)
print("RMSE:", rid_rmse)
print("R² Score:", np.mean(rid_y_pred))

#Polynomial Regression

from sklearn.preprocessing import PolynomialFeatures

pf_model = PolynomialFeatures(degree=2)
pf_train = pf_model.fit_transform(X_train)
pf_test = pf_model.transform(X_test)

pl_model = LinearRegression()
pl_model.fit(pf_train, y_train)
pl_y_pred = pl_model.predict(pf_test)

pf_mse = mean_squared_error(y_test, pl_y_pred)
pf_rmse = np.sqrt(pf_mse)
pf_r2 = r2_score(y_test, pl_y_pred)

print("\n🔹 Ridge Regression Results:")
print("MSE:", pf_mse)
print("RMSE:", pf_rmse)
print("R² Score:", pf_r2)

#Gradient Desent new

from sklearn.linear_model import SGDRegressor

sgd = SGDRegressor()

sgd.fit(X_train,y_train)
sgd_y_pred = sgd.predict(X_test)

mse3 = mean_squared_error(y_test, sgd_y_pred)
print(mse3)

#Gradient Desent

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load dataset
df = pd.read_csv("housing.csv")

# Step 2: Handle missing values
df.fillna(df.mean(numeric_only=True), inplace=True)

# Step 3: Drop non-numeric columns or encode if necessary
df = df.select_dtypes(include=['int64', 'float64'])

# Step 4: Split features and target
X = df.drop("PTRATIO", axis=1)
y = df["PTRATIO"]

# Step 5: Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 6: Add bias term (column of 1s)
X_scaled = np.c_[np.ones(X_scaled.shape[0]), X_scaled]  # shape: (n_samples, n_features+1)

# Step 7: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Step 8: Gradient Descent function
def gradient_descent_multivariate(X, y, lr=0.01, iterations=1000):
    n_samples, n_features = X.shape
    theta = np.zeros(n_features)
    y = y.to_numpy()
    cost_history = []

    for i in range(iterations):
        y_pred = np.dot(X, theta)
        error = y_pred - y
        cost = (1 / (2 * n_samples)) * np.dot(error, error)
        cost_history.append(cost)

        gradients = (1 / n_samples) * np.dot(X.T, error)
        theta -= lr * gradients

        if i % 100 == 0:
            print(f"Iteration {i:4d} ➤ Cost: {cost:.4f}")

    return theta, cost_history

# Step 9: Run gradient descent
theta, cost_history = gradient_descent_multivariate(X_train, y_train, lr=0.01, iterations=1000)

# Step 10: Predictions on test set
y_pred = np.dot(X_test, theta)

# Step 11: Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("\n🔹 Evaluation on Test Set:")
print("MSE:", mse)
print("RMSE:", np.sqrt(mse))
print("R² Score:", r2)
# Step 12: Plot cost over iterations
plt.title("Gradient Descent Convergence")
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.grid(True)
plt.show()
