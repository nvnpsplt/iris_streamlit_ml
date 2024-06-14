import streamlit as st
import pandas as pd
import joblib

# Load the model
model = joblib.load('classification_model.joblib')

# Streamlit header
st.header("Iris Flower Classification:")

# Load and display the image
st.image("image.png", use_column_width=True)

# Instruction to the user
st.write("Please insert values to get Iris class prediction")

# Create sliders for input
SepalLengthCm = st.slider('SepalLengthCm:', 0.0, 6.0)
SepalWidthCm = st.slider('SepalWidthCm:', 0.0, 6.0)
PetalLengthCm = st.slider('PetalLengthCm:', 0.0, 6.0)
PetalWidthCm = st.slider('PetalWidthCm:', 0.0, 6.0)

# Prepare the input data
data = {
    'SepalLengthCm': SepalLengthCm,
    'SepalWidthCm': SepalWidthCm,
    'PetalLengthCm': PetalLengthCm,
    'PetalWidthCm': PetalWidthCm
}
features = pd.DataFrame(data, index=[0])

# Make predictions
pred_proba = model.predict_proba(features)
prediction = model.predict(features)

# Display prediction percentages
st.subheader('Prediction Percentages:')
st.write('**Probability of Iris Class being Iris-setosa is (in %):**', pred_proba[0][0] * 100)
st.write('**Probability of Iris Class being Iris-versicolor is (in %):**', pred_proba[0][1] * 100)
st.write('**Probability of Iris Class being Iris-virginica (in %):**', pred_proba[0][2] * 100)
