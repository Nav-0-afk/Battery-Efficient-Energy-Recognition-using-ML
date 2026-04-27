import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from lightgbm import LGBMClassifier
from sklearn.ensemble import IsolationForest
import joblib
import os

def train_hierarchical_models():
    os.makedirs('models', exist_ok=True)
    
    X_train_full = np.load('artifacts/X_train_scaled.npy')
    y_train = np.load('artifacts/y_train_encoded.npy')
    selected_indices = joblib.load('artifacts/selected_indices.joblib')
    
    X_train_reduced = X_train_full[:, selected_indices]

    # --- STAGE 0: Anomaly Gate ---
    print("Training Stage 0 (Isolation Forest)...")
    stage0_if = IsolationForest(contamination=0.01, random_state=42, n_jobs=-1)
    stage0_if.fit(X_train_reduced) 
    joblib.dump(stage0_if, 'models/stage0_if.joblib')
    
    # --- STAGE 1: Gatekeeper (Static vs Dynamic) ---
    print("Training Stage 1 (Logistic Regression Gatekeeper)...")
    y_train_binary = np.isin(y_train, [0, 1, 2]).astype(int) # 1=Dynamic, 0=Static
    stage1_lr = LogisticRegression(random_state=42, max_iter=1000)
    stage1_lr.fit(X_train_reduced, y_train_binary)
    joblib.dump(stage1_lr, 'models/stage1_lr.joblib')

    # --- STAGE 1B: Static Specialist ---
    # --- STAGE 1B: Static Specialist ---
    print("Training Stage 1B (LinearSVC for Static)...")
    static_mask = (y_train_binary == 0)
    X_train_static = X_train_reduced[static_mask]
    y_train_static = y_train[static_mask]
    
    # Use LinearSVC instead of SVC(kernel='linear')
    # dual=False is explicitly required when samples > features (which is true here)
    stage1b_svm = LinearSVC(random_state=42, dual=False, max_iter=2000)
    stage1b_svm.fit(X_train_static, y_train_static)
    joblib.dump(stage1b_svm, 'models/stage1b_svm.joblib')
    
    # --- STAGE 2: Dynamic Specialist ---
    print("Training Stage 2 (LightGBM for Dynamic)...")
    dynamic_mask = (y_train_binary == 1)
    X_train_dynamic = X_train_reduced[dynamic_mask]
    y_train_dynamic = y_train[dynamic_mask]
    
    stage2_lgbm = LGBMClassifier(random_state=42, n_jobs=-1)
    stage2_lgbm.fit(X_train_dynamic, y_train_dynamic)
    
    # Save for Python execution
    joblib.dump(stage2_lgbm, 'models/stage2_lgbm.joblib')
    # Save native booster format for Treelite C compilation
    stage2_lgbm.booster_.save_model('models/stage2_lgbm_native.txt')
    
    print("Pipeline trained and models saved.")

if __name__ == "__main__":
    train_hierarchical_models()