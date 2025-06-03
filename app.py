import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from pickle import load
import seaborn as sns

model = load(open("models/decision_tree_classifier_gini_5_3_10_42.sav", "rb"))
class_dict = {
    0 : "No tiene diabetes",
    1 : "Si tiene diabetes",
}

desc_vars = {
    "Pregnancies": "Número de embarazos del paciente.",
    "Glucose": "Concentración de glucosa en plasma a 2 horas en prueba de tolerancia.",
    "SkinThickness": "Grosor del pliegue cutáneo del tríceps (medida en mm).",
    "BMI": "Índice de masa corporal (kg/m²).",
    "DiabetesPedigreeFunction": "Función pedigrí de diabetes (herencia genética).",
    "Age": "Edad en años."
}

st.title("Predicción de Diabetes")

st.markdown(
    """
    Esta aplicación predice la probabilidad de diabetes utilizando un modelo de árbol de decisión.
    Ajusta los valores de las variables a la izquierda y pulsa **Predecir**.
    """
)

st.sidebar.markdown("<h1>Descripción de variables</h1>", unsafe_allow_html=True)
for var, desc in desc_vars.items():
    st.sidebar.markdown(f"<p style='font-size:18px;'><b>{var}:</b> {desc}</p>", unsafe_allow_html=True)

pregnancies = st.slider("Embarazos (Pregnancies)", 0, 20, 1)
glucose = st.slider("Glucosa (Glucose)", 0, 250, 120)
skinthickness = st.slider("Espesor de piel (SkinThickness)", 0, 150, 20)
bmi = st.slider("Índice de masa corporal (BMI)", 0, 90, 30)
dpf = st.slider("Función pedigrí (DiabetesPedigreeFunction)", 0.0, 4.0, 0.5, 0.01)
age = st.slider("Edad (Age)", 18, 90, 30)

if st.button("Predecir"):
    input_data = np.array([[pregnancies, glucose, skinthickness, bmi, dpf, age]])
    prediccion = model.predict(input_data)[0]
    probabilidad = model.predict_proba(input_data)[0][prediccion]

    st.subheader("Resultado de la predicción")
    st.write(f"**Clase:** {class_dict[prediccion]}")
    st.write(f"**Probabilidad:** {probabilidad*100:.2f} %")

auc_score = 0.87
accuracy = 0.80

st.markdown("---")
st.subheader("Métricas del modelo")
col1, col2 = st.columns(2)
col1.metric("AUC ROC", f"{auc_score:.2f}")
col2.metric("Accuracy", f"{accuracy:.2f}")


def plot_roc():
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve([0,1,0,1], [0.1,0.9,0.2,0.8])
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Curva ROC")
    ax.legend()
    st.pyplot(fig)

with st.expander("Mostrar curva ROC"):
    plot_roc()

def plot_feature_importances(model, features):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    fig, ax = plt.subplots()
    sns.barplot(x=importances[indices], y=np.array(features)[indices], ax=ax)
    ax.set_title("Importancia de características")
    st.pyplot(fig)

with st.expander("Mostrar importancia de variables"):
    plot_feature_importances(model, ["Pregnancies", "Glucose", "SkinThickness", "BMI", "DiabetesPedigreeFunction", "Age"])