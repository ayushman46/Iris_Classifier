# 🌸 Iris Flower Classification - VS Code Version 🌸
# Author: [Your Name]
# Goal: Predict Iris species based on flower measurements

# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Optional: Make plots look nice
sns.set(style="whitegrid")

# Step 2: Load the dataset
# Make sure 'Iris.csv' is in the same folder as this script
df = pd.read_csv("Iris.csv", encoding='ISO-8859-1')


# Step 3: Explore the dataset
print("\nFirst 5 rows:")
print(df.head())

print("\n Dataset Info:")
print(df.info())

print("\n Statistical Summary:")
print(df.describe())

print("\n Missing Values:")
print(df.isnull().sum())

# Step 4: Clean the data (drop 'Id' column)
df.drop('Id', axis=1, inplace=True)

# Step 5: Visualize the data
# Species count
plt.figure(figsize=(6,4))
sns.countplot(data=df, x='Species', palette='viridis')
plt.title("Species Count")
plt.show()

# Pairplot
sns.pairplot(df, hue='Species', palette='husl')
plt.suptitle("Pairplot of Iris Features", y=1.02)
plt.show()

# Heatmap of correlations
plt.figure(figsize=(6,5))
sns.heatmap(df.drop('Species', axis=1).corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()

# Step 6: Encode target labels (Species)
le = LabelEncoder()
df['Species'] = le.fit_transform(df['Species'])  # Setosa=0, Versicolor=1, Virginica=2

print("\nEncoded Species:")
print(df['Species'].unique())

# Step 7: Prepare features and labels
X = df.drop('Species', axis=1)
y = df['Species']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\n Train Samples: {X_train.shape[0]}, Test Samples: {X_test.shape[0]}")

# Step 8: Train Logistic Regression Model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Step 9: Make Predictions
y_pred = model.predict(X_test)

# Step 10: Evaluate the Model
print("\n Accuracy Score:", accuracy_score(y_test, y_pred))

print("\n Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Confusion Matrix
conf_mat = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='YlGnBu',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Step 11: Predict a New Sample
sample = np.array([[5.1, 3.5, 1.4, 0.2]])  # Example measurements
prediction = model.predict(sample)
species = le.inverse_transform(prediction)[0]

print(f"\n Predicted Species for sample {sample.tolist()[0]}: {species}")
