from flask import Flask, render_template, request, redirect, url_for
import pickle
import numpy as np
import MySQLdb

app = Flask(__name__)

# Load the model and scaler
with open('/Users/yesaswimadabattula/Documents/cvd_website/ml_model/model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('/Users/yesaswimadabattula/Documents/cvd_website/ml_model/scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# MySQL database configuration
db = MySQLdb.connect(
    host="localhost",
    user="root",
    passwd="22875777",
    db="heart_disease_db"
)
cursor = db.cursor()

# Route for the main page
@app.route('/')
def index():
    return render_template('index.html')

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get form data
        age = request.form['age']
        sex = request.form['sex']
        cigsPerDay = request.form['cigsPerDay']
        totChol = request.form['totChol']
        sysBP = request.form['sysBP']
        glucose = request.form['glucose']

        # Print the received values to check if any are missing
        print(f"Age: {age}, Sex: {sex}, Cigarettes per Day: {cigsPerDay}, Total Cholesterol: {totChol}, Systolic BP: {sysBP}, Glucose: {glucose}")

        # Save user data to MySQL
        query = "INSERT INTO patient_data (age, sex, cigsPerDay, totChol, sysBP, glucose) VALUES (%s, %s, %s, %s, %s, %s)"
        values = (age, sex, cigsPerDay, totChol, sysBP, glucose)
        cursor.execute(query, values)
        db.commit()

        # Prepare data for prediction
        input_data = np.array([[age, sex, cigsPerDay, totChol, sysBP, glucose]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)

        # Prepare result
        result = "High Risk" if prediction[0] == 1 else "Low Risk"

        # Pass entered data and prediction result to the template
        return render_template('result.html', age=age, sex=sex, cigsPerDay=cigsPerDay, totChol=totChol,
                               sysBP=sysBP, glucose=glucose, prediction=result)


if __name__ == '__main__':
    app.run(debug=True, port=5001)
