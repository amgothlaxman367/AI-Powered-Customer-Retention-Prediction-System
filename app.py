from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load Model
with open("WA_Fn-UseC_-Telco-Customer-Churn.pkl", "rb") as f:
    model = pickle.load(f)

# Load Scaler
with open("standard_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)


@app.route("/", methods=["GET", "POST"])
def index():

    prediction = None

    if request.method == "POST":

        try:

            # ===============================
            # Get User Inputs
            # ===============================

            gender = request.form.get("gender")
            Partner = request.form.get("Partner")
            Dependents = request.form.get("Dependents")
            PhoneService = request.form.get("PhoneService")
            MultipleLines = request.form.get("MultipleLines")
            InternetService = request.form.get("InternetService")
            OnlineSecurity = request.form.get("OnlineSecurity")
            OnlineBackup = request.form.get("OnlineBackup")
            DeviceProtection = request.form.get("DeviceProtection")
            TechSupport = request.form.get("TechSupport")
            StreamingTV = request.form.get("StreamingTV")
            StreamingMovies = request.form.get("StreamingMovies")
            Contract = request.form.get("Contract")
            PaperlessBilling = request.form.get("PaperlessBilling")
            PaymentMethod = request.form.get("PaymentMethod")
            Sim = request.form.get("Sim")

            # ===============================
            # Convert to 29 Features
            # ===============================

            features = [

                # Gender
                1 if gender == "Male" else 0,

                # Partner
                1 if Partner == "Yes" else 0,

                # Dependents
                1 if Dependents == "Yes" else 0,

                # Phone Service
                1 if PhoneService == "Yes" else 0,

                # Multiple Lines
                1 if MultipleLines == "No phone service" else 0,
                1 if MultipleLines == "Yes" else 0,

                # Internet Service
                1 if InternetService == "Fiber optic" else 0,
                1 if InternetService == "No" else 0,

                # Online Security
                1 if OnlineSecurity == "No internet service" else 0,
                1 if OnlineSecurity == "Yes" else 0,

                # Online Backup
                1 if OnlineBackup == "No internet service" else 0,
                1 if OnlineBackup == "Yes" else 0,

                # Device Protection
                1 if DeviceProtection == "No internet service" else 0,
                1 if DeviceProtection == "Yes" else 0,

                # Tech Support
                1 if TechSupport == "No internet service" else 0,
                1 if TechSupport == "Yes" else 0,

                # Streaming TV
                1 if StreamingTV == "No internet service" else 0,
                1 if StreamingTV == "Yes" else 0,

                # Streaming Movies
                1 if StreamingMovies == "No internet service" else 0,
                1 if StreamingMovies == "Yes" else 0,

                # Contract
                1 if Contract == "One year" else 0,
                1 if Contract == "Two year" else 0,

                # Paperless Billing
                1 if PaperlessBilling == "Yes" else 0,

                # Payment Method
                1 if PaymentMethod == "Credit card (automatic)" else 0,
                1 if PaymentMethod == "Electronic check" else 0,
                1 if PaymentMethod == "Mailed check" else 0,

                # SIM
                1 if Sim == "BSNL" else 0,
                1 if Sim == "Jio" else 0,
                1 if Sim == "Vi" else 0

            ]

            # ===============================
            # Prediction
            # ===============================

            features = np.array(features).reshape(1, -1)

            features_scaled = scaler.transform(features)

            result = model.predict(features_scaled)[0]

            if result == 1:
                prediction = "Customer Will Churn ❌"
            else:
                prediction = "Customer Will Stay ✅"

        except Exception as e:

            print("Error :", e)

            prediction = f"Error : {e}"

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)