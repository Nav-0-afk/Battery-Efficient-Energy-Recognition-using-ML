import numpy as np
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDOneClassSVM
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
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
    # --- STAGE 0: Anomaly Gate ---
    print("Training Stage 0 (Original: Isolation Forest | Alt: SGD One-Class SVM)...")
    
    # Original: Isolation Forest
    stage0_if = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1, n_estimators=40, max_samples=16)
    stage0_if.fit(X_train_reduced) 
    joblib.dump(stage0_if, 'models/stage0_if.joblib')
    
    # Alternative: Linear SGD One-Class SVM
    # 'nu' acts similarly to 'contamination' (upper bound on fraction of training errors)
    stage0_ocsvm = SGDOneClassSVM(nu=0.01, random_state=42)
    stage0_ocsvm.fit(X_train_reduced)
    joblib.dump(stage0_ocsvm, 'models/stage0_ocsvm.joblib')
    
    # --- STAGE 1: Gatekeepers ---
    print("Training Stage 1 (Original: Logistic Regression | Alt: Decision Tree)...")
    y_train_binary = np.isin(y_train, [0, 1, 2]).astype(int) # 1=Dynamic, 0=Static
    
    stage1_lr = LogisticRegression(random_state=42, max_iter=1000)
    stage1_lr.fit(X_train_reduced, y_train_binary)
    joblib.dump(stage1_lr, 'models/stage1_lr.joblib')
    
    stage1_dt = DecisionTreeClassifier(max_depth=3, random_state=42)
    stage1_dt.fit(X_train_reduced, y_train_binary)
    joblib.dump(stage1_dt, 'models/stage1_dt.joblib')

    # --- STAGE 1B: Static Specialists ---
    print("Training Stage 1B (Original: LinearSVC | Alt: SGDClassifier)...")
    static_mask = (y_train_binary == 0)
    X_train_static = X_train_reduced[static_mask]
    y_train_static = y_train[static_mask]
    
    stage1b_svm = LinearSVC(random_state=42, dual=False, max_iter=2000)
    stage1b_svm.fit(X_train_static, y_train_static)
    joblib.dump(stage1b_svm, 'models/stage1b_svm.joblib')

    stage1b_sgd = SGDClassifier(random_state=42, max_iter=2000)
    stage1b_sgd.fit(X_train_static, y_train_static)
    joblib.dump(stage1b_sgd, 'models/stage1b_sgd.joblib')
    
    # --- STAGE 2: Dynamic Specialists ---
    print("Training Stage 2 (Original: LightGBM | Alt: XGBoost)...")
    dynamic_mask = (y_train_binary == 1)
    X_train_dynamic = X_train_reduced[dynamic_mask]
    y_train_dynamic = y_train[dynamic_mask]
    
    stage2_lgbm = LGBMClassifier(random_state=42, n_jobs=-1)
    stage2_lgbm.fit(X_train_dynamic, y_train_dynamic)
    joblib.dump(stage2_lgbm, 'models/stage2_lgbm.joblib')
    stage2_lgbm.booster_.save_model('models/stage2_lgbm_native.txt')

    stage2_xgb = XGBClassifier(eval_metric='mlogloss', random_state=42, n_jobs=-1,n_estimators=40, max_depth=3)
    stage2_xgb.fit(X_train_dynamic, y_train_dynamic)
    joblib.dump(stage2_xgb, 'models/stage2_xgb.joblib')
    
    print("All Original and Alternative pipelines trained and saved.")

if __name__ == "__main__":
    train_hierarchical_models()