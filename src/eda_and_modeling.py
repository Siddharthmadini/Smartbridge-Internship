import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from imblearn.combine import SMOTETomek

def run_eda(app_df, credit_df, plot_dir):
    print("=== Running Exploratory Data Analysis (EDA) ===")
    
    # 1. Descriptive Stats
    print("\nApplicant Record Describe:")
    print(app_df.describe(include='all'))
    
    # 2. Univariate Analysis (Value Counts and Count Plots)
    categorical_cols = ['OCCUPATION_TYPE', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS']
    for col in categorical_cols:
        if col in app_df.columns:
            plt.figure(figsize=(10, 5))
            sns.countplot(data=app_df, y=col, order=app_df[col].value_counts().index, palette='viridis')
            plt.title(f'Distribution of {col}')
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f'countplot_{col.lower()}.png'))
            plt.close()
            print(f"Saved countplot for {col}")
            
    # 3. Multivariate Analysis (Correlation Heatmap on numerical features)
    numeric_cols = app_df.select_dtypes(include=[np.number]).columns.tolist()
    # Filter out ID
    if 'ID' in numeric_cols:
        numeric_cols.remove('ID')
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(app_df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap of Applicant Numerical Features')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'correlation_heatmap.png'))
    plt.close()
    print("Saved correlation heatmap")

def d_tree(xtrain, xtest, ytrain, ytest):
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(xtrain, ytrain)
    ypred = dt.predict(xtest)
    return dt, ypred

def main():
    # Setup paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    plot_dir = os.path.join(base_dir, 'static', 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    
    # Load data
    app_df = pd.read_csv(os.path.join(data_dir, 'application_record.csv'))
    credit_df = pd.read_csv(os.path.join(data_dir, 'credit_record.csv'))
    
    print(f"Loaded application record shape: {app_df.shape}")
    print(f"Loaded credit record shape: {credit_df.shape}")
    
    # Run EDA
    run_eda(app_df, credit_df, plot_dir)
    
    # Preprocessing
    print("\n=== Data Preprocessing ===")
    
    # 1. Missing values
    print("Null values count in applicant data:")
    print(app_df.isnull().sum())
    
    # Drop OCCUPATION_TYPE as it has ~30% missing values
    if 'OCCUPATION_TYPE' in app_df.columns:
        app_df.drop('OCCUPATION_TYPE', axis=1, inplace=True)
        print("Dropped OCCUPATION_TYPE column")
        
    # 2. Clean applicant data
    # Create family dependency
    app_df['FAMILY_DEPENDENCY'] = app_df['CNT_CHILDREN'] + app_df['CNT_FAM_MEMBERS']
    
    # Absolute values for negative features
    app_df['DAYS_BIRTH'] = app_df['DAYS_BIRTH'].abs()
    app_df['DAYS_EMPLOYED'] = app_df['DAYS_EMPLOYED'].abs()
    
    # 3. Group credit records to build aggregated credit features and target
    # TARGET LABEL CREATION: convert multi-class STATUS to binary:
    # 1 for Approved (C, X, 0, 1) and 0 for Rejected (2, 3, 4, 5)
    def to_binary(status):
        if status in ['0', '1', 'C', 'X']:
            return 1   # Approved
        return 0       # Not Approved / Rejected
        
    credit_df['STATUS_BIN'] = credit_df['STATUS'].apply(to_binary)
    
    # Derive aggregated features per ID
    # - open_month (min MONTHS_BALANCE)
    # - end_months (max MONTHS_BALANCE)
    # - window (max - min + 1)
    # - Number_of_Loans (months of active loan record where status is not 'X')
    # - EMI_Paid_Off (count of 'C')
    # - EMI_of_Pastdues (count of '0', '1', '2', '3', '4', '5')
    # - target label: 0 if ANY month was '2','3','4','5' (delinquency of 60+ days), else 1
    
    print("Processing credit history records...")
    credit_grouped = credit_df.groupby('ID').agg(
        open_month=('MONTHS_BALANCE', 'min'),
        end_months=('MONTHS_BALANCE', 'max'),
        Number_of_Loans=('STATUS', lambda x: (x != 'X').sum()),
        EMI_Paid_Off=('STATUS', lambda x: (x == 'C').sum()),
        EMI_of_Pastdues=('STATUS', lambda x: x.isin(['0', '1', '2', '3', '4', '5']).sum()),
        STATUS_BIN=('STATUS_BIN', 'min')  # 0 is the minimum, so if there is any 0, client becomes 0
    ).reset_index()
    
    credit_grouped['window'] = credit_grouped['end_months'] - credit_grouped['open_month'] + 1
    print(f"Processed credit record shape: {credit_grouped.shape}")
    
    # 4. Merge applicant data and credit data FIRST (to ensure we keep IDs with history)
    final_df = app_df.merge(credit_grouped, on='ID', how='inner')
    print(f"Merged raw dataset shape: {final_df.shape}")
    
    # 5. Drop duplicates on demographic features AFTER merge
    features_to_check = [c for c in app_df.columns if c != 'ID']
    final_df = final_df.drop_duplicates(subset=features_to_check, keep='first')
    print(f"Shape after dropping duplicate applicant features: {final_df.shape}")
    
    # 6. Encode Categorical Columns
    categorical_cols = [
        'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 
        'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE'
    ]
    
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        final_df[col] = le.fit_transform(final_df[col])
        label_encoders[col] = le
        print(f"Encoded {col} (classes: {le.classes_.tolist()})")
        
    # Features & Target definition
    feature_cols = [
        'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'AMT_INCOME_TOTAL', 
        'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 
        'DAYS_BIRTH', 'DAYS_EMPLOYED', 'CNT_FAM_MEMBERS', 'EMI_Paid_Off', 'EMI_of_Pastdues', 'Number_of_Loans'
    ]
    
    X = final_df[feature_cols]
    y = final_df['STATUS_BIN']
    
    print("\nClass distribution before train-test split:")
    print(y.value_counts())
    
    # Train-test split (80/20) with stratification to maintain class balance
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Handle Class Imbalance with SMOTETomek ONLY on training set (avoids data leakage)
    smote_tomek = SMOTETomek(random_state=42)
    X_train_res, y_train_res = smote_tomek.fit_resample(X_train, y_train)
    print("Class distribution after SMOTETomek (training only):")
    print(pd.Series(y_train_res).value_counts())
    
    # Scaling numerical features (Fit only on training resampled features)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_res)
    X_test_scaled = scaler.transform(X_test)
    
    # === Models Training and Evaluation ===
    print("\n=== Model Training & Comparison ===")
    
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree': None, # special function call
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss')
    }
    
    results = {}
    trained_models = {}
    
    for model_name, clf in models.items():
        print(f"\n--- Training {model_name} ---")
        if model_name == 'Decision Tree':
            clf, y_pred = d_tree(X_train_scaled, X_test_scaled, y_train_res, y_test)
        else:
            clf.fit(X_train_scaled, y_train_res)
            y_pred = clf.predict(X_test_scaled)
            
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(f"Accuracy: {acc:.4f} | F1-Score: {f1:.4f}")
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        
        results[model_name] = {'F1': f1, 'Accuracy': acc}
        trained_models[model_name] = clf

    # Choose the best model based on F1-score
    best_model_name = max(results, key=lambda k: results[k]['F1'])
    best_model = trained_models[best_model_name]
    print(f"\nBest Model selected for deployment: {best_model_name} with F1-Score: {results[best_model_name]['F1']:.4f}")
    
    # Generate and Save ROC Curve Plot
    from sklearn.metrics import roc_curve, auc
    plt.figure(figsize=(10, 8))
    for model_name, clf in trained_models.items():
        if hasattr(clf, "predict_proba"):
            y_prob = clf.predict_proba(X_test_scaled)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.4f})')
            print(f"{model_name} ROC-AUC: {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    roc_plot_path = os.path.join(plot_dir, 'roc_curve.png')
    plt.savefig(roc_plot_path)
    plt.close()
    print(f"Saved ROC Curve comparison plot to {roc_plot_path}")
    
    # Generate and Save Feature Importance Plot
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        plt.figure(figsize=(12, 6))
        sns.barplot(x=importances[indices], y=np.array(feature_cols)[indices], palette='viridis')
        plt.title(f'Feature Importances of Best Model ({best_model_name})')
        plt.xlabel('Relative Importance')
        plt.ylabel('Features')
        plt.tight_layout()
        fi_plot_path = os.path.join(plot_dir, 'feature_importance.png')
        plt.savefig(fi_plot_path)
        plt.close()
        print(f"Saved Feature Importance plot to {fi_plot_path}")
    elif hasattr(best_model, 'coef_'):
        importances = np.abs(best_model.coef_[0])
        indices = np.argsort(importances)[::-1]
        plt.figure(figsize=(12, 6))
        sns.barplot(x=importances[indices], y=np.array(feature_cols)[indices], palette='viridis')
        plt.title(f'Feature Importances (Absolute Coefficients) of Best Model ({best_model_name})')
        plt.xlabel('Relative Importance (Absolute Coefficient)')
        plt.ylabel('Features')
        plt.tight_layout()
        fi_plot_path = os.path.join(plot_dir, 'feature_importance.png')
        plt.savefig(fi_plot_path)
        plt.close()
        print(f"Saved Feature Importance plot to {fi_plot_path}")

    # Save model and preprocessing assets
    model_filepath = os.path.join(base_dir, 'model.pkl')
    model_payload = {
        'model_name': best_model_name,
        'model': best_model,
        'scaler': scaler,
        'label_encoders': label_encoders,
        'feature_cols': feature_cols,
        'accuracy': results[best_model_name]['Accuracy'],
        'f1_score': results[best_model_name]['F1']
    }
    
    with open(model_filepath, 'wb') as f:
        pickle.dump(model_payload, f)
        
    print(f"Saved serialized model payload to {model_filepath}")

if __name__ == '__main__':
    main()
