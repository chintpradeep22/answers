import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay

# Load dataset
df = pd.read_csv("train.csv")

# Drop columns that are not useful for this classification task
df = df.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1)
df = df.dropna()

# Handle missing values
df['Age'].fillna(df['Age'].mean())
df['Embarked'].fillna(df['Embarked'].mode()[0])

# Encode categorical features
df['Sex'] = LabelEncoder().fit_transform(df['Sex'])       # male=1, female=0
df['Embarked'] = LabelEncoder().fit_transform(df['Embarked'])

# Define X and y
X = df.drop("Survived", axis=1)
y = df["Survived"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=50)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier, BaggingClassifier, StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

# logistic regression model
#log_model = LogisticRegression(max_iter=1000)
#log_model = SVC(kernel='linear') 
#log_model = RandomForestClassifier(n_estimators=100, random_state=42)
#log_model = DecisionTreeClassifier(random_state=42)
#log_model = GaussianNB()
#log_model = AdaBoostClassifier(estimator=DecisionTreeClassifier(),n_estimators=50, random_state=42)
#log_model = BaggingClassifier(estimator=DecisionTreeClassifier(),n_estimators=10, random_state=42)
#Stacking-----
estimators = [
    ('dt', DecisionTreeClassifier()),
    ('svc', SVC(probability=True))
]
log_model = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(max_iter=1000), passthrough=True)

# Train model

log_model.fit(X_train, y_train)
log_y_pred = log_model.predict(X_test)

# Evaluation
log_accuracy = accuracy_score(y_test, log_y_pred)
log_precision = precision_score(y_test, log_y_pred)
log_recall = recall_score(y_test, log_y_pred)

print(f"🔹 Logistic Regression Results:")
print(f"Accuracy:  {log_accuracy:.4f}")
print(f"Precision: {log_precision:.4f}")
print(f"Recall:    {log_recall:.4f}")

# Confusion matrix
log_cm = confusion_matrix(y_test, log_y_pred)
log_disp = ConfusionMatrixDisplay(confusion_matrix=log_cm)
log_disp.plot(cmap=plt.cm.Blues)
plt.title("Logistic Regression - Confusion Matrix")
plt.show()


#K-means
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Step 1: Reduce dimensionality using PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train)

# Step 2: Fit KMeans on PCA-reduced data
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_pca)
labels = kmeans.labels_
print("Confusion matrix (clusters vs labels):")
print(confusion_matrix(y_train, labels))

# Step 3: Get cluster centers in PCA space
centers_pca = kmeans.cluster_centers_

# Step 4: Plot clusters and cluster centers
plt.figure(figsize=(6, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', label='Data Points')
plt.scatter(centers_pca[:, 0], centers_pca[:, 1], c='red', s=200, marker='X', label='Cluster Centers')
plt.title("KMeans Clustering (PCA Reduced)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend()
plt.grid(True)
plt.show()
