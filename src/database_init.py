import os
import sqlite3
from datetime import datetime
from werkzeug.security import generate_password_hash

def get_db_connection(db_path):
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_path = os.path.join(base_dir, 'credit_approval.db')
    
    print(f"Initializing database at: {db_path}")
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    
    # 1. Create Users Table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS Users (
        UserID INTEGER PRIMARY KEY AUTOINCREMENT,
        Name TEXT NOT NULL,
        Email TEXT UNIQUE NOT NULL,
        Password TEXT NOT NULL,
        Role TEXT NOT NULL
    );
    ''')
    
    # 2. Create Applicant_Details Table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS Applicant_Details (
        ApplicantID INTEGER PRIMARY KEY AUTOINCREMENT,
        UserID INTEGER,
        IncomeType TEXT,
        EducationType TEXT,
        FamilyStatus TEXT,
        HousingType TEXT,
        EmploymentDays INTEGER,
        FOREIGN KEY (UserID) REFERENCES Users(UserID) ON DELETE CASCADE
    );
    ''')
    
    # 3. Create Credit_History Table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS Credit_History (
        HistoryID INTEGER PRIMARY KEY AUTOINCREMENT,
        ApplicantID INTEGER,
        MonthlyBalance INTEGER,
        PaymentStatus TEXT,
        OverdueStatus TEXT,
        FOREIGN KEY (ApplicantID) REFERENCES Applicant_Details(ApplicantID) ON DELETE CASCADE
    );
    ''')
    
    # 4. Create ML_Model Table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS ML_Model (
        ModelID INTEGER PRIMARY KEY AUTOINCREMENT,
        ModelName TEXT NOT NULL,
        AlgorithmType TEXT NOT NULL,
        Accuracy REAL NOT NULL,
        ModelFile TEXT NOT NULL
    );
    ''')
    
    # 5. Create Approval_Prediction Table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS Approval_Prediction (
        PredictionID INTEGER PRIMARY KEY AUTOINCREMENT,
        ApplicantID INTEGER UNIQUE,
        ModelID INTEGER,
        ApprovalResult TEXT NOT NULL,
        RiskCategory TEXT NOT NULL,
        PredictionDate TEXT NOT NULL,
        FOREIGN KEY (ApplicantID) REFERENCES Applicant_Details(ApplicantID) ON DELETE CASCADE,
        FOREIGN KEY (ModelID) REFERENCES ML_Model(ModelID)
    );
    ''')
    
    # Seed Data
    print("Seeding database with initial records...")
    
    # Seed Users
    admin_pw = generate_password_hash('admin123')
    user_pw = generate_password_hash('pass123')
    cursor.execute("INSERT OR IGNORE INTO Users (UserID, Name, Email, Password, Role) VALUES (1, 'Admin SmartBridge', 'admin@smartbridge.com', ?, 'Manager');", (admin_pw,))
    cursor.execute("INSERT OR IGNORE INTO Users (UserID, Name, Email, Password, Role) VALUES (2, 'John Doe', 'john@gmail.com', ?, 'Applicant');", (user_pw,))
    
    # Seed Applicant Details
    cursor.execute("INSERT OR IGNORE INTO Applicant_Details (ApplicantID, UserID, IncomeType, EducationType, FamilyStatus, HousingType, EmploymentDays) VALUES (1, 2, 'Working', 'Secondary / secondary special', 'Married', 'House / apartment', 1200);")
    
    # Seed Credit History
    cursor.execute("INSERT OR IGNORE INTO Credit_History (HistoryID, ApplicantID, MonthlyBalance, PaymentStatus, OverdueStatus) VALUES (1, 1, -12, 'C', 'None');")
    cursor.execute("INSERT OR IGNORE INTO Credit_History (HistoryID, ApplicantID, MonthlyBalance, PaymentStatus, OverdueStatus) VALUES (2, 1, -11, '0', '1-29 Days Overdue');")
    
    # ML_Model will be populated when training is completed or in this script
    # Let's check if we can insert placeholder/models later or now
    cursor.execute("INSERT OR IGNORE INTO ML_Model (ModelID, ModelName, AlgorithmType, Accuracy, ModelFile) VALUES (1, 'Random Forest Classifier', 'Random Forest', 0.92, 'model.pkl');")
    
    # Seed Approval Prediction
    cursor.execute("INSERT OR IGNORE INTO Approval_Prediction (PredictionID, ApplicantID, ModelID, ApprovalResult, RiskCategory, PredictionDate) VALUES (1, 1, 1, 'Approved', 'Low Risk', ?);", (datetime.now().strftime('%Y-%m-%d %H:%M:%S'),))
    
    conn.commit()
    conn.close()
    print("Database initialization and seeding completed successfully.")

if __name__ == '__main__':
    main()
