import os
import pickle
import numpy as np
import sqlite3
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

# Load trained model and preprocessing tools
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model.pkl')
DB_PATH = os.path.join(BASE_DIR, 'credit_approval.db')

model_payload = None
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, 'rb') as f:
        model_payload = pickle.load(f)
    print("Model payload loaded successfully.")
else:
    print(f"WARNING: Model file not found at {MODEL_PATH}. Make sure to run modeling first.")

# Connect to database helper
def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.row_factory = sqlite3.Row
    return conn

# Home Route
@app.route('/')
def home():
    return render_template('home.html')

# Prediction Form Route
@app.route('/predict_form')
def predict_form():
    # If model is loaded, we can pass category options to form if needed
    categories = {
        'income_types': ['Working', 'Commercial associate', 'Pensioner', 'State servant', 'Student'],
        'education_types': ['Secondary / secondary special', 'Higher education', 'Incomplete higher', 'Lower secondary', 'Academic degree'],
        'family_statuses': ['Married', 'Single / not married', 'Civil marriage', 'Separated', 'Widow'],
        'housing_types': ['House / apartment', 'With parents', 'Rented apartment', 'Municipal apartment', 'Office apartment', 'Co-op apartment']
    }
    return render_template('index.html', categories=categories)

# Prediction Result Route
@app.route('/predict', methods=['POST'])
def predict():
    if not model_payload:
        return "Model not loaded. Please train the model first.", 500
        
    try:
        # Extract inputs from form
        gender = request.form.get('gender') # 'M' or 'F'
        own_car = request.form.get('own_car') # 'Y' or 'N'
        own_realty = request.form.get('own_realty') # 'Y' or 'N'
        income = float(request.form.get('income'))
        income_type = request.form.get('income_type')
        education = request.form.get('education')
        family_status = request.form.get('family_status')
        housing_type = request.form.get('housing_type')
        
        # Age in years converted to Days Birth (absolute value)
        age_years = float(request.form.get('age'))
        days_birth = int(age_years * 365.25)
        
        # Years employed converted to Days Employed (absolute value)
        years_employed = float(request.form.get('years_employed'))
        if years_employed == 0:
            days_employed = 365243
        else:
            days_employed = int(years_employed * 365.25)
        
        family_members = float(request.form.get('family_members'))
        
        # Credit history inputs
        emi_paid_off = float(request.form.get('emi_paid_off', 0))
        emi_pastdues = float(request.form.get('emi_pastdues', 0))
        num_loans = float(request.form.get('num_loans', 0))
        
        # Retrieve label encoders and scale tools
        encoders = model_payload['label_encoders']
        scaler = model_payload['scaler']
        model = model_payload['model']
        feature_cols = model_payload['feature_cols']
        
        # Preprocess / Encode inputs
        try:
            enc_gender = encoders['CODE_GENDER'].transform([gender])[0]
            enc_car = encoders['FLAG_OWN_CAR'].transform([own_car])[0]
            enc_realty = encoders['FLAG_OWN_REALTY'].transform([own_realty])[0]
            enc_income_type = encoders['NAME_INCOME_TYPE'].transform([income_type])[0]
            enc_education = encoders['NAME_EDUCATION_TYPE'].transform([education])[0]
            enc_family_status = encoders['NAME_FAMILY_STATUS'].transform([family_status])[0]
            enc_housing_type = encoders['NAME_HOUSING_TYPE'].transform([housing_type])[0]
        except ValueError as e:
            # Fallback for unseen classes during manual input
            print(f"Encoding warning: {e}. Using default values.")
            enc_gender = 0
            enc_car = 0
            enc_realty = 0
            enc_income_type = 4 # 'Working' default index
            enc_education = 4 # 'Secondary / secondary special' default index
            enc_family_status = 1 # 'Married' default index
            enc_housing_type = 1 # 'House / apartment' default index
            
        # Assemble feature vector in the exact order model was trained
        # feature_cols order:
        # ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'AMT_INCOME_TOTAL', 
        #  'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 
        #  'DAYS_BIRTH', 'DAYS_EMPLOYED', 'CNT_FAM_MEMBERS', 'EMI_Paid_Off', 'EMI_of_Pastdues', 'Number_of_Loans']
        
        input_data = [
            enc_gender, enc_car, enc_realty, income,
            enc_income_type, enc_education, enc_family_status, enc_housing_type,
            days_birth, days_employed, family_members, emi_paid_off, emi_pastdues, num_loans
        ]
        
        # Scale inputs using a DataFrame to maintain feature names and suppress warning
        import pandas as pd
        input_df = pd.DataFrame([input_data], columns=feature_cols)
        input_scaled = scaler.transform(input_df)
        
        # Make Prediction
        prediction = model.predict(input_scaled)[0]
        
        # Determine approval result text and risk category
        if prediction == 1:
            result = "Approved"
            risk = "Minimal Risk" if emi_pastdues == 0 else "Low Risk"
        else:
            result = "Rejected"
            risk = "High Risk" if emi_pastdues > 5 else "Moderate Risk"
            
        # Audit Log: Save to SQLite database implementing the ER schema
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Insert or reference a User (use guest User 2)
        user_id = 2 
        
        # Insert Applicant Details
        cursor.execute('''
            INSERT INTO Applicant_Details 
            (UserID, IncomeType, EducationType, FamilyStatus, HousingType, EmploymentDays)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (user_id, income_type, education, family_status, housing_type, int(days_employed)))
        applicant_id = cursor.lastrowid
        
        # Insert Credit History summary record (simulate current month balance as 0)
        cursor.execute('''
            INSERT INTO Credit_History 
            (ApplicantID, MonthlyBalance, PaymentStatus, OverdueStatus)
            VALUES (?, 0, ?, ?)
        ''', (applicant_id, 'C' if result == 'Approved' else '2', f'{int(emi_pastdues)} Overdue Months'))
        
        # Retrieve ML Model ID (insert or retrieve best model)
        cursor.execute('SELECT ModelID FROM ML_Model WHERE ModelName = ?', (model_payload['model_name'],))
        model_row = cursor.fetchone()
        if model_row:
            model_id = model_row['ModelID']
        else:
            cursor.execute('''
                INSERT INTO ML_Model (ModelName, AlgorithmType, Accuracy, ModelFile)
                VALUES (?, ?, ?, 'model.pkl')
            ''', (model_payload['model_name'], model_payload['model_name'], float(model_payload['accuracy'])))
            model_id = cursor.lastrowid
            
        # Insert Approval Prediction
        prediction_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cursor.execute('''
            INSERT INTO Approval_Prediction 
            (ApplicantID, ModelID, ApprovalResult, RiskCategory, PredictionDate)
            VALUES (?, ?, ?, ?, ?)
        ''', (applicant_id, model_id, result, risk, prediction_date))
        
        conn.commit()
        conn.close()
        
        # Package data to display on result page
        display_info = {
            'gender': 'Male' if gender == 'M' else 'Female',
            'income': f"${income:,.2f}",
            'income_type': income_type,
            'education': education,
            'risk_category': risk,
            'model_used': model_payload['model_name'],
            'prediction_text': f"Credit Card {result}",
            'status': result.lower() # 'approved' or 'rejected'
        }
        
        return render_template('result.html', info=display_info)
        
    except Exception as e:
        print(f"Error making prediction: {e}")
        return f"An error occurred while making prediction: {e}", 400

# View Recent predictions
@app.route('/admin/database')
def view_database():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Query join table of predictions and applicants
    cursor.execute('''
        SELECT 
            p.PredictionID, p.ApprovalResult, p.RiskCategory, p.PredictionDate,
            a.IncomeType, a.EducationType, a.HousingType, a.EmploymentDays,
            m.ModelName
        FROM Approval_Prediction p
        JOIN Applicant_Details a ON p.ApplicantID = a.ApplicantID
        JOIN ML_Model m ON p.ModelID = m.ModelID
        ORDER BY p.PredictionID DESC
        LIMIT 20
    ''')
    rows = cursor.fetchall()
    conn.close()
    
    return render_template('database_view.html', rows=rows)

# Model Performance Dashboard Route
@app.route('/admin/performance')
def model_performance():
    metrics = {
        'Logistic Regression': {'Accuracy': '79.14%', 'F1': '88.21%', 'AUC': '0.7389'},
        'Decision Tree': {'Accuracy': '92.34%', 'F1': '95.99%', 'AUC': '0.6137'},
        'Random Forest': {'Accuracy': '95.48%', 'F1': '97.68%', 'AUC': '0.7868'},
        'XGBoost': {'Accuracy': '96.15%', 'F1': '98.03%', 'AUC': '0.7696'}
    }
    return render_template('model_performance.html', metrics=metrics)

# IBM Watson ML optional serving integration skeleton configuration
"""
# Helper template for Cloud-based deployment of the model serving via IBM Watson Machine Learning SDK
# To use, install: pip install ibm-watson-machine-learning
# Add credentials to environment variables: WATSON_API_KEY, WATSON_URL, WATSON_SPACE_ID

from ibm_watson_machine_learning import APIClient

wml_credentials = {
    "url": os.getenv("WATSON_URL", "https://us-south.ml.cloud.ibm.com"),
    "apikey": os.getenv("WATSON_API_KEY", "")
}
client = APIClient(wml_credentials)
client.set.default_space(os.getenv("WATSON_SPACE_ID", ""))

def predict_via_watson(features):
    # payload format:
    # payload_scoring = {"input_data": [{"fields": feature_cols, "values": [features]}]}
    # response = client.deployments.score(deployment_id, payload_scoring)
    # return response['predictions'][0]['values'][0][0]
    pass
"""

if __name__ == '__main__':
    app.run(debug=True)
