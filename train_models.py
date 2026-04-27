import numpy as np
from sklearn.svm import SVC
from xgboost import XGBClassifier
import joblib
from sklearn.ensemble import IsolationForest
import os

def train_hierarchical_models():
    os.makedirs('models', exist_ok=True)
    
    X_train_full = np.load('artifacts/X_train_scaled.npy')
    y_train = np.load('artifacts/y_train_encoded.npy')
    selected_indices = joblib.load('artifacts/selected_indices.joblib')
    
    X_train_reduced = X_train_full[:, selected_indices]

    # --- STAGE 0 PREP: Anomaly Detection ---
    print("Training Stage 0 (Isolation Forest)...")
    # contamination=0.01 assumes ~1% of your training data might be noisy, 
    # and strictly bounds the "normal" envelope.
    stage0_if = IsolationForest(contamination=0.01, random_state=42, n_jobs=-1)
    stage0_if.fit(X_train_reduced) 
    joblib.dump(stage0_if, 'models/stage0_if.joblib')
    
    # --- STAGE 1 PREP: Gatekeeper ---
    # UCI HAR Encoded: 0, 1, 2 = Dynamic. 3, 4, 5 = Static.
    # We map Dynamic -> 1, Static -> 0
    y_train_binary = np.isin(y_train, [0, 1, 2]).astype(int) 
    
    print("Training Stage 1 (Gatekeeper SVM)...")
    stage1_svm = SVC(kernel='linear', probability=False, random_state=42)
    stage1_svm.fit(X_train_reduced, y_train_binary)
    joblib.dump(stage1_svm, 'models/stage1_svm.joblib')
    
    # --- STAGE 2 PREP: Specialist ---
    # Train XGBoost ONLY on the data where the original label was dynamic
    dynamic_mask = (y_train_binary == 1)
    X_train_dynamic = X_train_reduced[dynamic_mask]
    y_train_dynamic = y_train[dynamic_mask]
    
    print("Training Stage 2 (Heavy XGBoost)...")
    # eval_metric avoids warnings in newer XGBoost versions
    stage2_xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', n_jobs=-1, random_state=42)
    stage2_xgb.fit(X_train_dynamic, y_train_dynamic)
    joblib.dump(stage2_xgb, 'models/stage2_xgb.joblib')
    
    print("Pipeline trained and models saved.")

if __name__ == "__main__":
    train_hierarchical_models()