import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import RandomizedSearchCV
import joblib

def select_features():
    X_train = np.load('artifacts/X_train_scaled.npy')
    y_train = np.load('artifacts/y_train_encoded.npy')
    
    print("Starting RandomizedSearchCV for RFE + RFC...")
    
    # 1. Initialize the base estimator
    # Use n_jobs=-1 here to parallelize tree building
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    
    # 2. Initialize RFE
    rfe = RFE(estimator=rf)
    
    # 3. Define the hyperparameter search space
    # Target RFE parameters directly, target RFC parameters using 'estimator__'
    param_distributions = {
        'n_features_to_select': [100, 150, 200, 250],
        'step': [5, 10, 20],
        'estimator__n_estimators': [30, 50, 100],
        'estimator__max_depth': [None, 5, 10, 20],
        'estimator__min_samples_split': [2, 5, 10]
    }
    
    # 4. Configure Randomized Search
    # n_jobs=1 is set here to prevent nested parallelization conflicts with RF's n_jobs=-1
    search = RandomizedSearchCV(
        estimator=rfe,
        param_distributions=param_distributions,
        n_iter=10,          # Number of random combinations to test
        cv=3,               # 3-fold cross-validation
        scoring='accuracy',
        verbose=2,
        random_state=42,
        n_jobs=1            
    )
    
    print("Fitting RandomizedSearchCV")
    search.fit(X_train, y_train)
    
    print(f"\nBest Parameters Found: {search.best_params_}")
    print(f"Best CV Accuracy: {search.best_score_:.4f}")
    
    # 5. Extract the best fitted RFE model
    best_rfe = search.best_estimator_
    selected_indices = np.where(best_rfe.support_)[0]
    
    joblib.dump(selected_indices, 'artifacts/selected_indices.joblib')
    print(f"Feature selection complete. Kept {len(selected_indices)} features out of 561.")

if __name__ == "__main__":
    select_features()