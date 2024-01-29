#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load the CSV file
data = pd.read_csv('C:/Users/DIVYA/Desktop/model-1/Training.csv')
data = data.dropna(axis=1)

# Separate features (X) and labels (y)
X = data.iloc[:, 1:-1]  # Exclude the first column (disease) and the last column (prognosis)
y = data['prognosis']

# Encode the target labels
le = LabelEncoder()
y = le.fit_transform(y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the KNN model
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)

# Predictions on the test set
y_pred = knn_model.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")


# In[ ]:




