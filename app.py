from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

dt_model = pickle.load(open("model_dt.pkl", "rb"))
svc_model = pickle.load(open("model_svc.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

model_names = ["Decision Tree", "SVC"]

@app.route("/")
def home():
    return render_template("index.html", model_names=model_names)


@app.route("/predict", methods=["POST"])
def predict():
    selected_model = request.form.get("model", "Decision Tree")

    data = [
        float(request.form["Pregnancies"]),
        float(request.form["Glucose"]),
        float(request.form["BloodPressure"]),
        float(request.form["SkinThickness"]),
        float(request.form["Insulin"]),
        float(request.form["BMI"]),
        float(request.form["DiabetesPedigreeFunction"]),
        float(request.form["Age"])
    ]

    df = pd.DataFrame([data], columns=[
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age"
    ])

    scaled = scaler.transform(df)

    # Pilih model berdasarkan input user
    if selected_model == "Decision Tree":
        result = dt_model.predict(scaled)[0]
    else:
        result = svc_model.predict(scaled)[0]

    prediction = (
        "Diabetic"
        if result == 1
        else "Non-Diabetic"
    )

    return render_template(
        "index.html",
        prediction=prediction,
        model_names=model_names,
        selected_model=selected_model
    )


if __name__ == "__main__":
    app.run(debug=True)