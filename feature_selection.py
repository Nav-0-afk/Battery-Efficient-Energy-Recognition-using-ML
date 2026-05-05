import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
import joblib

def select_features():
    X_train = np.load('artifacts/X_train_scaled.npy')
    y_train = np.load('artifacts/y_train_encoded.npy')
    
    print("Starting RF-RFE.")
    
    rf = RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=42)
    
    selector = RFE(estimator=rf, n_features_to_select=100, step=5, verbose=1)
    selector.fit(X_train, y_train)
    
    selected_indices = np.where(selector.support_)[0]
    
    joblib.dump(selected_indices, 'artifacts/selected_indices.joblib')
    print(f"Feature selection complete. Kept {len(selected_indices)} features out of 561.")

if __name__ == "__main__":
    select_features()