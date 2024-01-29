import streamlit as st
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

# Load the CSV file
data = pd.read_csv('cleaned_dataset.csv')

# Separate features (X) and labels (y)
X = data.iloc[:, 1:-1]  # Exclude the first column (disease) and the last column (prognosis)
y = data['prognosis']

# Encode the target labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Create and train the KNN model
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X, y_encoded)

# Function to get user input
def get_user_input():
    user_input = []
    for symptom in X.columns:
        response = st.checkbox(symptom)
        user_input.append(1 if response else 0)
    return user_input

# Get user input
st.title("Disease Prediction App")
st.write("Select the symptoms:")
user_input = get_user_input()

# Add a submit button
if st.button("Submit"):
    # Make prediction
    prediction = knn_model.predict([user_input])

    # Reverse transform the prediction to get the disease name
    predicted_disease = le.inverse_transform(prediction)[0]

    # Display the result
    st.subheader("Prediction:")
    st.write(f"The predicted disease is: {predicted_disease}")
