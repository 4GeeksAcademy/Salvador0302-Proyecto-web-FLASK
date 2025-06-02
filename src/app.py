from pickle import load
import streamlit as st

model = load(open("../models/decision_tree_classifier_gini_5_3_10_42.sav", "rb"))
class_dict = {
    "0" : "No tiene diabetes",
    "1" : "Si tiene diabetes",
}

st.title("Prediciendo la diabetes!!!")

val1 = st.slider("Pregnancies", min_value = 0, max_value = 20, step = 1)
val2 = st.slider("Glucose", min_value = 0, max_value = 250, step = 1)
val3 = st.slider("SkinThickness", min_value = 0, max_value = 150, step = 1)
val4 = st.slider("BMI", min_value = 0, max_value = 90, step = 1)
val5 = st.slider("DiabetesPedigreeFunction", min_value = 0.0, max_value = 4.0, step = 0.1)
val6 = st.slider("Age", min_value = 18, max_value = 90, step = 1)

if st.button("Predict"):
    prediction = str(model.predict([[val1, val2, val3, val4, val5, val6]])[0])
    pred_class = class_dict[prediction]
    st.write("Prediction:", pred_class)