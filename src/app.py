from flask import Flask, request, render_template
from pickle import load

app = Flask(__name__)
model = load(open("../models/decision_tree_classifier_gini_5_4_10_42.sav", "rb"))
class_dict = {
    "0": "No tiene diabetes",
    "1": "Si tiene diabetes",
}

@app.route("/", methods = ["GET", "POST"])

def index():
    if request.method == "POST":
        
        val1 = float(request.form["embarazos"])
        val2 = float(request.form["Glucosa"])
        val3 = float(request.form["espesor de la piel"])
        val4 = float(request.form["BMI"])
        val5 = float(request.form["funcion pedigri de diabetes"])
        val6 = float(request.form["Edad"])
        
        data = [[val1, val2, val3, val4, val5, val6]]
        prediction = str(model.predict(data)[0])
        pred_class = class_dict[prediction]
    else:
        pred_class = None
    
    return render_template("index.html", prediction = pred_class)