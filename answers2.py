import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Lasso, LinearRegression, Ridge, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score

df = pd.read_csv("BostonHousing.csv")
df = df.dropna()

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Make y suitable for classification
median_price = y.median()
y_class = (y > median_price).astype(int)
y = y_class

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --------------------------------------------
# Regression Models
# --------------------------------------------

# Linear Regression
from sklearn.linear_model import LinearRegression
lr_model = LinearRegression()
lr_model.fit(X_train_reg, y_train_reg)
y_pred_lr = lr_model.predict(X_test_reg)
print("Linear Regression MSE:", mean_squared_error(y_test_reg, y_pred_lr))

# Lasso Regression
from sklearn.linear_model import Lasso
lasso_model = Lasso(alpha=0.1)
lasso_model.fit(X_train_reg, y_train_reg)
y_pred_lasso = lasso_model.predict(X_test_reg)
print("Lasso Regression MSE:", mean_squared_error(y_test_reg, y_pred_lasso))

# Ridge Regression
from sklearn.linear_model import Ridge
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train_reg, y_train_reg)
y_pred_ridge = ridge_model.predict(X_test_reg)
print("Ridge Regression MSE:", mean_squared_error(y_test_reg, y_pred_ridge))

# Polynomial Regression
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train_reg)
X_test_poly = poly.transform(X_test_reg)

poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train_reg)
y_pred_poly = poly_model.predict(X_test_poly)
print("Polynomial Regression MSE:", mean_squared_error(y_test_reg, y_pred_poly))


model2 = LogisticRegression(max_iter=10000)
model2.fit(X_train, y_train)
y_pred_log = model2.predict(X_test)
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_log))

from sklearn.svm import SVC
model3 = SVC(kernel = 'linear')
model3.fit(X_train, y_train)
y_pred_svm = model3.predict(X_test)

print("SVM accuracy: ", accuracy_score(y_test, y_pred_svm))

from sklearn.naive_bayes import GaussianNB
model4 = GaussianNB()
model4.fit(X_train, y_train)
y_pred_gass = model4.predict(X_test)

print("Gaussian accuracy:", accuracy_score(y_test, y_pred_gass))

from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
model5 = DecisionTreeClassifier()
model5.fit(X_train, y_train)

plt.figure(figsize=(30,200))  # Bigger size
plot_tree(model5, 
          filled=True, 
          feature_names=X.columns.tolist(), 
          class_names=["Low", "High"])
plt.title("Decision Tree for Boston Housing")
plt.show()

from sklearn.ensemble import BaggingClassifier
model6 = BaggingClassifier()
model6.fit(X_train, y_train)
y_pred_bc = model6.predict(X_test)

print("Bagging accuracy:", accuracy_score(y_test,y_pred_bc))

# Boosting (AdaBoost)
from sklearn.ensemble import AdaBoostClassifier
boost_model = AdaBoostClassifier(estimator=DecisionTreeClassifier(), n_estimators=50, random_state=42)
boost_model.fit(X_train_class, y_train_class)
y_pred_boost = boost_model.predict(X_test_class)
print("Boosting (AdaBoost) Accuracy:", accuracy_score(y_test_class, y_pred_boost))

# Stacking
from sklearn.ensemble import StackingClassifier
estimators = [
    ('dt', DecisionTreeClassifier()),
    ('svc', SVC(probability=True))
]
stack_model = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(max_iter=10000))
stack_model.fit(X_train_class, y_train_class)
y_pred_stack = stack_model.predict(X_test_class)
print("Stacking Accuracy:", accuracy_score(y_test_class, y_pred_stack))

# --------------------------------------------
# Clustering Model
# --------------------------------------------
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_cluster)

plt.figure(figsize=(6,6))
plt.scatter(X_cluster[:, 0], X_cluster[:, 1], c=clusters, cmap='viridis')
plt.title("KMeans Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
