import numpy as np
import os
import joblib
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDOneClassSVM
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import RandomizedSearchCV

def train_hierarchical():
    os.makedirs('models', exist_ok=True)
    
    X_train_full = np.load('artifacts/X_train_scaled.npy')
    y_train = np.load('artifacts/y_train_encoded.npy')
    selected_indices = joblib.load('artifacts/selected_indices.joblib')
    
    X_train_reduced = X_train_full[:, selected_indices]
    
    # Dictionary to store the best parameters for the final printout
    best_params_summary = {}

    # --- STAGE 0: Anomaly Gate (UNSUPERVISED - NO CV) ---
    print("\n--- Training Stage 0 (Static Fit) ---")
    stage0_if = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1, n_estimators=40, max_samples=16)
    stage0_if.fit(X_train_reduced) 
    joblib.dump(stage0_if, 'models/stage0_if.joblib')
    
    stage0_ocsvm = SGDOneClassSVM(nu=0.01, random_state=42)
    stage0_ocsvm.fit(X_train_reduced)
    joblib.dump(stage0_ocsvm, 'models/stage0_ocsvm.joblib')

    # --- STAGE 1: Gatekeepers (SUPERVISED) ---
    print("\n--- Tuning Stage 1 ---")
    y_train_binary = np.isin(y_train, [0, 1, 2]).astype(int) 
    
    lr_params = {'C': [0.01, 0.1, 1, 10], 'max_iter': [1000, 2000]}
    lr_search = RandomizedSearchCV(LogisticRegression(random_state=42), lr_params, n_iter=5, cv=3, n_jobs=-1, verbose=1, random_state=42)
    lr_search.fit(X_train_reduced, y_train_binary)
    joblib.dump(lr_search.best_estimator_, 'models/stage1_lr.joblib')
    best_params_summary['Stage 1 - Logistic Regression'] = lr_search.best_params_

    dt_params = {'max_depth': [3, 5, 7, None], 'min_samples_split': [2, 5, 10]}
    dt_search = RandomizedSearchCV(DecisionTreeClassifier(random_state=42), dt_params, n_iter=5, cv=3, n_jobs=-1, verbose=1, random_state=42)
    dt_search.fit(X_train_reduced, y_train_binary)
    joblib.dump(dt_search.best_estimator_, 'models/stage1_dt.joblib')
    best_params_summary['Stage 1 - Decision Tree'] = dt_search.best_params_

    # --- STAGE 1B: Static Specialists ---
    print("\n--- Tuning Stage 1B ---")
    static_mask = (y_train_binary == 0)
    X_train_static = X_train_reduced[static_mask]
    y_train_static = y_train[static_mask]
    
    svc_params = {'C': [0.1, 1, 10], 'tol': [1e-3, 1e-4]}
    svc_search = RandomizedSearchCV(LinearSVC(random_state=42, dual=False, max_iter=2000), svc_params, n_iter=5, cv=3, n_jobs=-1, verbose=1, random_state=42)
    svc_search.fit(X_train_static, y_train_static)
    joblib.dump(svc_search.best_estimator_, 'models/stage1b_svm.joblib')
    best_params_summary['Stage 1B - Linear SVC'] = svc_search.best_params_

    sgd_params = {'alpha': [1e-4, 1e-3, 1e-2], 'penalty': ['l2', 'l1', 'elasticnet']}
    sgd_search = RandomizedSearchCV(SGDClassifier(random_state=42, max_iter=2000), sgd_params, n_iter=5, cv=3, n_jobs=-1, verbose=1, random_state=42)
    sgd_search.fit(X_train_static, y_train_static)
    joblib.dump(sgd_search.best_estimator_, 'models/stage1b_sgd.joblib')
    best_params_summary['Stage 1B - SGD Classifier'] = sgd_search.best_params_

    # --- STAGE 2: Dynamic Specialists ---
    print("\n--- Tuning Stage 2 (Heavy Computation) ---")
    dynamic_mask = (y_train_binary == 1)
    X_train_dynamic = X_train_reduced[dynamic_mask]
    y_train_dynamic = y_train[dynamic_mask]
    
    lgbm_params = {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 5, 7]}
    lgbm_search = RandomizedSearchCV(LGBMClassifier(random_state=42, n_jobs=-1), lgbm_params, n_iter=5, cv=3, n_jobs=1, verbose=1, random_state=42)
    lgbm_search.fit(X_train_dynamic, y_train_dynamic)
    joblib.dump(lgbm_search.best_estimator_, 'models/stage2_lgbm.joblib')
    lgbm_search.best_estimator_.booster_.save_model('models/stage2_lgbm_native.txt')
    best_params_summary['Stage 2 - LightGBM'] = lgbm_search.best_params_

    xgb_params = {'n_estimators': [40, 80], 'max_depth': [3, 4, 5], 'learning_rate': [0.05, 0.1]}
    xgb_search = RandomizedSearchCV(XGBClassifier(eval_metric='mlogloss', random_state=42, n_jobs=-1), xgb_params, n_iter=5, cv=3, n_jobs=1, verbose=1, random_state=42)
    xgb_search.fit(X_train_dynamic, y_train_dynamic)
    joblib.dump(xgb_search.best_estimator_, 'models/stage2_xgb.joblib')
    best_params_summary['Stage 2 - XGBoost'] = xgb_search.best_params_

    # --- FINAL HYPERPARAMETER PRINTOUT ---
    print("\n" + "="*60)
    print("FINAL TUNED HYPERPARAMETERS SUMMARY")
    print("="*60)
    for model_name, params in best_params_summary.items():
        print(f"\n[{model_name}]")
        for param_name, param_value in params.items():
            print(f"   --> {param_name}: {param_value}")
    print("\n" + "="*60)
    print("All pipelines tuned, trained, and saved.")

if __name__ == "__main__":
    train_hierarchical()