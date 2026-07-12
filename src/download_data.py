import os
import urllib.request

def download_pair(app_url, credit_url, app_path, credit_path):
    print(f"Attempting to download pair:\nApp: {app_url}\nCredit: {credit_url}")
    try:
        # Check and download application record
        urllib.request.urlretrieve(app_url, app_path)
        # Check and download credit record
        urllib.request.urlretrieve(credit_url, credit_path)
        print("Success! Both files downloaded.")
        return True
    except Exception as e:
        print(f"Failed with this pair: {e}")
        # Clean up partial downloads
        if os.path.exists(app_path):
            os.remove(app_path)
        if os.path.exists(credit_path):
            os.remove(credit_path)
        return False

def main():
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    app_path = os.path.join(data_dir, 'application_record.csv')
    credit_path = os.path.join(data_dir, 'credit_record.csv')
    
    # Candidate source pairs
    candidates = [
        (
            'https://raw.githubusercontent.com/semasuka/Credit-card-approval-prediction-classification/master/datasets/application_record.csv',
            'https://raw.githubusercontent.com/semasuka/Credit-card-approval-prediction-classification/master/datasets/credit_record.csv'
        ),
        (
            'https://raw.githubusercontent.com/damaniayesh/Credit-Card-Approval-Prediction/master/application_record.csv',
            'https://raw.githubusercontent.com/damaniayesh/Credit-Card-Approval-Prediction/master/credit_record.csv'
        ),
        (
            'https://raw.githubusercontent.com/shiraz-30/Credit-Card-Approval-Prediction-Model/main/application_record.csv',
            'https://raw.githubusercontent.com/shiraz-30/Credit-Card-Approval-Prediction-Model/main/credit_record.csv'
        ),
        (
            'https://raw.githubusercontent.com/gromart/Credit-Card-Approval-Prediction/main/application_record.csv',
            'https://raw.githubusercontent.com/gromart/Credit-Card-Approval-Prediction/main/credit_record.csv'
        )
    ]
    
    success = False
    for app_url, credit_url in candidates:
        if download_pair(app_url, credit_url, app_path, credit_path):
            success = True
            break
            
    if not success:
        print("Could not download the datasets from any candidate URL. Please verify your internet connection.")
        exit(1)

if __name__ == '__main__':
    main()
