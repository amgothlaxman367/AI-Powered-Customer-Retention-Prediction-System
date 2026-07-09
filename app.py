from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model and scaler
with open("WA_Fn-UseC_-Telco-Customer-Churn.pkl", "rb") as f:
    model = pickle.load(f)

with open("standard_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)


@app.route("/", methods=["GET", "POST"])
def index():

    prediction = None

    if request.method == "POST":

        try:

            # Get user inputs
            gender = request.form["gender"]
            Partner = request.form["Partner"]
            Dependents = request.form["Dependents"]
            PhoneService = request.form["PhoneService"]
            MultipleLines = request.form["MultipleLines"]
            InternetService = request.form["InternetService"]
            OnlineSecurity = request.form["OnlineSecurity"]
            OnlineBackup = request.form["OnlineBackup"]
            DeviceProtection = request.form["DeviceProtection"]
            TechSupport = request.form["TechSupport"]
            StreamingTV = request.form["StreamingTV"]
            StreamingMovies = request.form["StreamingMovies"]
            Contract = request.form["Contract"]
            PaperlessBilling = request.form["PaperlessBilling"]
            PaymentMethod = request.form["PaymentMethod"]
            Sim = request.form["Sim"]


            # Convert into 29 features

            features = [

                # gender
                1 if gender=="Male" else 0,

                # Partner
                1 if Partner=="Yes" else 0,

                # Dependents
                1 if Dependents=="Yes" else 0,

                # PhoneService
                1 if PhoneService=="Yes" else 0,

                # MultipleLines
                1 if MultipleLines=="No phone service" else 0,
                1 if MultipleLines=="Yes" else 0,


                # InternetService
                1 if InternetService=="Fiber optic" else 0,
                1 if InternetService=="No" else 0,


                # OnlineSecurity
                1 if OnlineSecurity=="No internet service" else 0,
                1 if OnlineSecurity=="Yes" else 0,


                # OnlineBackup
                1 if OnlineBackup=="No internet service" else 0,
                1 if OnlineBackup=="Yes" else 0,


                # DeviceProtection
                1 if DeviceProtection=="No internet service" else 0,
                1 if DeviceProtection=="Yes" else 0,


                # TechSupport
                1 if TechSupport=="No internet service" else 0,
                1 if TechSupport=="Yes" else 0,


                # StreamingTV
                1 if StreamingTV=="No internet service" else 0,
                1 if StreamingTV=="Yes" else 0,


                # StreamingMovies
                1 if StreamingMovies=="No internet service" else 0,
                1 if StreamingMovies=="Yes" else 0,


                # Contract
                1 if Contract=="One year" else 0,
                1 if Contract=="Two year" else 0,


                # PaperlessBilling
                1 if PaperlessBilling=="Yes" else 0,


                # PaymentMethod
                1 if PaymentMethod=="Credit card (automatic)" else 0,
                1 if PaymentMethod=="Electronic check" else 0,
                1 if PaymentMethod=="Mailed check" else 0,


                # Sim
                1 if Sim=="BSNL" else 0,
                1 if Sim=="Jio" else 0,
                1 if Sim=="Vi" else 0
            ]


            features_array = np.array([features])


            features_scaled = scaler.transform(features_array)


            result = model.predict(features_scaled)[0]


            if result == 1:
                prediction = "Customer Will Churn"
            else:
                prediction = "Customer Will Stay"


        except Exception as e:
            prediction = f"Error: {str(e)}"


    return render_template("index.html", prediction=prediction)



if __name__ == "__main__":
    app.run(debug=True)