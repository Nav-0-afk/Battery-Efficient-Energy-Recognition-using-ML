import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

def load_and_preprocess_data():
    os.makedirs('artifacts', exist_ok=True)

    # 1. Data importing
    feature_info = pd.read_csv('dataset/UCI HAR Dataset/features.txt', sep=r'\s+', header=None, names=['index', 'name'])
    activity_labels = pd.read_csv('dataset/UCI HAR Dataset/activity_labels.txt', sep=r'\s+', header=None, names=['id', 'activity'])

    X_train = pd.read_csv('dataset/UCI HAR Dataset/train/X_train.txt', sep=r'\s+', header=None)
    X_train.columns = feature_info['name']

    y_train = pd.read_csv('dataset/UCI HAR Dataset/train/y_train.txt', sep=r'\s+', header=None, names=['activity_id'])
    y_train['activity_name'] = y_train['activity_id'].map(activity_labels.set_index('id')['activity'])

    X_test = pd.read_csv('dataset/UCI HAR Dataset/test/X_test.txt', sep=r'\s+', header=None)
    y_test = pd.read_csv('dataset/UCI HAR Dataset/test/y_test.txt', sep=r'\s+', header=None, names=['activity_id'])# Encoding: 0 to 5 instead of 1 to 6
    y_train_encoded = (y_train['activity_id'] - 1).values.ravel()
    y_test_encoded = (y_test['activity_id'] - 1).values.ravel()

    X_train_np = X_train.values
    X_test_np = X_test.values

    # 2. Scaling
    print("Scaling the values")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_np)
    X_test_scaled = scaler.transform(X_test_np)

    # 3. SAVE ARTIFACTS
    joblib.dump(scaler, 'artifacts/scaler.joblib')
    np.save('artifacts/X_train_scaled.npy', X_train_scaled)
    np.save('artifacts/X_test_scaled.npy', X_test_scaled)
    np.save('artifacts/y_train_encoded.npy', y_train_encoded)
    np.save('artifacts/y_test_encoded.npy', y_test_encoded)
    
    print("Preprocessing completed.")

if __name__ == "__main__":
    load_and_preprocess_data()