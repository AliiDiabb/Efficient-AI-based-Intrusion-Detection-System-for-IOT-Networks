import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, validation_curve, learning_curve, GridSearchCV,RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import warnings
import xgboost as xgb, lightgbm as lgb


warnings.filterwarnings('ignore')

# Load the data Set
def load_and_prepare_data(file_path, classification_type='binary'):
    """
    Load EdgeIIoTset dataset and prepare for training
    
    Args:
        file_path (str): Path to the EdgeIIoTset CSV file
        classification_type (str): 'binary', '6-class', or '15-class'
    
    Returns:
        tuple: X_train, X_test, y_train, y_test, feature_names, label_encoder, scaler, class_info
    """
    # Load the dataset
    print(f"Loading EdgeIIoTset dataset for {classification_type} classification...")
    df = pd.read_csv(file_path)
    print(f"Dataset loaded successfully. Shape: {df.shape}")
    
    # Display basic information about the dataset
    print(f"\nDataset Info:")
    print(f"Total samples: {len(df)}")
    print(f"Total features: {df.shape[1]}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    
    # Determine target column based on classification type
    if classification_type == 'binary':
        if 'Attack_label' in df.columns:
            target_col = 'Attack_label'
            print("Using 'Attack_label' for binary classification (Normal vs Attack)")
        else:
            print("Attack_label not found. Creating binary target from Attack_type...")
            if 'Attack_type' in df.columns:
                df['Attack_label'] = (df['Attack_type'] != 'Normal').astype(int)
                target_col = 'Attack_label'
            else:
                raise ValueError("Neither Attack_label nor Attack_type found in dataset")
    
    elif classification_type == '6-class':
        if 'Attack_type' in df.columns:
            target_col = '6_Attack'
            print("Using '6_Attack' for 6-class classification")
          
        else:
            raise ValueError("Attack_type column not found for 6-class classification")
        
    elif classification_type == '15-class':
        if 'Attack_type' in df.columns:
            target_col = 'Attack_type'
            print("Using 'Attack_type' for 15-class classification (all attack types)")
        else:
            raise ValueError("Attack_type column not found for 15-class classification")
    
    else:
        raise ValueError("classification_type must be 'binary', '6-class', or '15-class'")
    
    print(f"Using '{target_col}' as target variable for {classification_type} classification")
    
    # Separate features and target
    feature_cols = [col for col in df.columns if col not in ['Attack_label', 'Attack_type','6_Attack', '6_Attack_text']]
    X = df[feature_cols]
    y = df[target_col]
    
    # Handle non-numeric features if any
    for col in X.columns:
        if X[col].dtype == 'object':
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
    
    # Encode target variable if it's categorical
    label_encoder = None
    if y.dtype == 'object':
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
        class_names = label_encoder.classes_
    else:
        class_names = np.unique(y)
    
    print(f"Features shape: {X.shape}")
    print(f"Target distribution ({classification_type}):")
    target_counts = pd.Series(y).value_counts().sort_index()
    
    if label_encoder:
        for i, count in enumerate(target_counts):
            print(f"  {class_names[i]}: {count} samples")
    else:
        print(target_counts)
    
    # Check for class imbalance
    imbalance_ratio = target_counts.min() / target_counts.max()
    print(f"Class imbalance ratio: {imbalance_ratio:.3f}")
    if imbalance_ratio < 0.1:
        print("⚠️  Severe class imbalance detected - this may lead to overfitting!")
    elif imbalance_ratio < 0.3:
        print("⚠️  Moderate class imbalance detected - monitoring for overfitting recommended")
    
    # Scale features for models that need it (SVM, KNN, gradient boosting)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split the data (80/20 split)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nData split completed:")
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")
    
    class_info = {
        'num_classes': len(class_names),
        'class_names': class_names,
        'target_col': target_col,
        'classification_type': classification_type
    }
    
    return X_train, X_test, y_train, y_test, X.columns.tolist(), label_encoder, scaler, class_info

# enhanced cross validation 
def enhanced_cross_validation(model, X_train, y_train, model_name, cv_folds=5, is_multiclass=False):
    """
    Enhanced cross-validation with overfitting detection for both binary and multiclass
    """
    print(f"\n🔍 Enhanced Cross-Validation for {model_name}")
    print("-" * 50)
    
    # Stratified K-Fold for better handling of imbalanced data
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    # Cross-validation scores
    cv_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='accuracy')
    
    # Additional metrics - use appropriate averaging for multiclass
    scoring_average = 'macro' if is_multiclass else 'binary'
    if is_multiclass:
        precision_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='precision_macro')
        recall_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='recall_macro')
        f1_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='f1_macro')
    else:
        precision_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='precision')
        recall_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='recall')
        f1_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='f1')
    
    # Training vs Validation performance (overfitting check)
    train_scores = []
    val_scores = []
    
    for train_idx, val_idx in skf.split(X_train, y_train):
        X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
        y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]
        
        # Train on fold
        model.fit(X_train_fold, y_train_fold)
        
        # Score on training and validation
        train_score = model.score(X_train_fold, y_train_fold)
        val_score = model.score(X_val_fold, y_val_fold)
        
        train_scores.append(train_score)
        val_scores.append(val_score)
    
    train_scores = np.array(train_scores)
    val_scores = np.array(val_scores)
    
    # Overfitting indicators
    mean_train_score = train_scores.mean()
    mean_val_score = val_scores.mean()
    overfitting_gap = mean_train_score - mean_val_score
    
    # Variance analysis
    cv_variance = cv_scores.var()
    
    print(f"Cross-validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    print(f"Cross-validation Precision: {precision_scores.mean():.4f} (+/- {precision_scores.std() * 2:.4f})")
    print(f"Cross-validation Recall: {recall_scores.mean():.4f} (+/- {recall_scores.std() * 2:.4f})")
    print(f"Cross-validation F1-Score: {f1_scores.mean():.4f} (+/- {f1_scores.std() * 2:.4f})")
    
    print(f"\n📊 Overfitting Analysis:")
    print(f"Training Score: {mean_train_score:.4f}")
    print(f"Validation Score: {mean_val_score:.4f}")
    print(f"Overfitting Gap: {overfitting_gap:.4f}")
    print(f"CV Variance: {cv_variance:.6f}")
    
    # Overfitting warnings
    if overfitting_gap > 0.05:
        print("🚨 WARNING: Significant overfitting detected!")
        if overfitting_gap > 0.1:
            print("🚨 CRITICAL: Severe overfitting - model may not generalize well!")
    elif overfitting_gap > 0.02:
        print("⚠️  Mild overfitting detected - monitor performance")
    else:
        print("✅ No significant overfitting detected")
    
    if cv_variance > 0.01:
        print("⚠️  High variance in CV scores - model may be unstable")
    
    return {
        'cv_accuracy_mean': cv_scores.mean(),
        'cv_accuracy_std': cv_scores.std(),
        'cv_precision_mean': precision_scores.mean(),
        'cv_recall_mean': recall_scores.mean(),
        'cv_f1_mean': f1_scores.mean(),
        'train_score_mean': mean_train_score,
        'val_score_mean': mean_val_score,
        'overfitting_gap': overfitting_gap,
        'cv_variance': cv_variance,
        'individual_cv_scores': cv_scores
    }




def train_lightgbm_Optimized(X_train, X_test, y_train, y_test, class_info):
    """
    Train and evaluate LightGBM model with enhanced overfitting detection
    """
    print("\n" + "="*50)
    print("Training LightGBM Classifier")
    print("="*50)
    
    start_time = time.time()
    is_multiclass = class_info['num_classes'] > 2
    
    # Initialize the model with regularization
    lgbm_model = lgb.LGBMClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=15,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        learning_rate=0.1,
        n_jobs=-1,
        verbose=-1
    )
    
    # Strategy 1: Quick screening with RandomizedSearchCV
    print("🔍 Phase 1: Quick parameter screening...")
    
    # Broad parameter space for initial screening
    param_distributions = {
        'n_estimators': [50, 100, 150, 200],
        'max_depth': [10, 15, 20, 25, -1],
        'min_child_samples': [10, 20, 30, 40],
        'subsample': [0.6, 0.7, 0.8, 0.9],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
        'learning_rate': [0.05, 0.1, 0.15, 0.2],
        'reg_alpha': [0, 0.1, 0.5, 1.0],
        'reg_lambda': [0, 0.1, 0.5, 1.0],
        'class_weight': [None, 'balanced']
    }
    
    lgbm_base = lgb.LGBMClassifier(
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    
    # Quick randomized search - test 20 combinations
    random_search = RandomizedSearchCV(
        lgbm_base,
        param_distributions,
        n_iter=20,  # Test 20 random combinations
        cv=3,
        scoring='f1_macro' if is_multiclass else 'accuracy',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    random_search.fit(X_train, y_train)
    
    print(f"✅ Phase 1 completed in {time.time() - start_time:.1f}s")
    print(f"Best random search score: {random_search.best_score_:.4f}")
    
    # Print Phase 1 best parameters
    print("\n📊 Phase 1 - Best Parameters from Random Search:")
    print("-" * 45)
    for param, value in random_search.best_params_.items():
        print(f"  {param:<20}: {value}")
    print("-" * 45)
    
    # Strategy 2: Fine-tune around best parameters
    print("\n🎯 Phase 2: Fine-tuning best parameters...")
    
    phase2_start = time.time()
    best_params = random_search.best_params_
    
    # Create focused grid around best parameters
    fine_tune_grid = {}
    
    # Fine-tune n_estimators around best
    best_n_est = best_params['n_estimators']
    fine_tune_grid['n_estimators'] = [
        max(50, best_n_est - 50),
        best_n_est,
        best_n_est + 50
    ]
    
    # Fine-tune max_depth around best
    if best_params['max_depth'] != -1:
        best_depth = best_params['max_depth']
        fine_tune_grid['max_depth'] = [
            max(5, best_depth - 5),
            best_depth,
            best_depth + 5,
            -1
        ]
    else:
        fine_tune_grid['max_depth'] = [20, 25, -1]
    
    # Fine-tune learning_rate around best
    best_lr = best_params['learning_rate']
    fine_tune_grid['learning_rate'] = [
        max(0.01, best_lr - 0.05),
        best_lr,
        min(0.3, best_lr + 0.05)
    ]
    
    # Keep other good parameters fixed or with minimal variation
    fine_tune_grid['min_child_samples'] = [best_params['min_child_samples']]
    fine_tune_grid['subsample'] = [best_params['subsample']]
    fine_tune_grid['colsample_bytree'] = [best_params['colsample_bytree']]
    fine_tune_grid['reg_alpha'] = [best_params['reg_alpha']]
    fine_tune_grid['reg_lambda'] = [best_params['reg_lambda']]
    fine_tune_grid['class_weight'] = [best_params['class_weight']]
    
    # Fine-tuning grid search
    fine_search = GridSearchCV(
        lgbm_base,
        fine_tune_grid,
        cv=3,
        scoring='f1_macro' if is_multiclass else 'accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    fine_search.fit(X_train, y_train)
    
    print(f"✅ Phase 2 completed in {time.time() - phase2_start:.1f}s")
    print(f"Final best score: {fine_search.best_score_:.4f}")
    
    # Get final optimized model
    lgbm_model = fine_search.best_estimator_
    final_params = fine_search.best_params_
    
    # Print Final Best Parameters in a formatted way
    print("\n🏆 FINAL OPTIMAL HYPERPARAMETERS:")
    print("=" * 50)
    print(f"{'Parameter':<20} | {'Value':<15} | {'Type'}")
    print("-" * 50)
    for param, value in sorted(final_params.items()):
        param_type = type(value).__name__
        print(f"{param:<20} | {str(value):<15} | {param_type}")
    print("=" * 50)
    
    # Also print improvement from Phase 1 to Phase 2
    score_improvement = fine_search.best_score_ - random_search.best_score_
    print(f"\n📈 Score Improvement: {score_improvement:+.4f}")
    print(f"   Phase 1 Score: {random_search.best_score_:.4f}")
    print(f"   Phase 2 Score: {fine_search.best_score_:.4f}")
    
    # Enhanced cross-validation on the optimized model
    print("\n🔍 Performing enhanced cross-validation analysis...")
    cv_results = enhanced_cross_validation(lgbm_model, X_train, y_train, "LightGBM", is_multiclass=is_multiclass)
    
    training_time = time.time() - start_time
    
    # Make predictions
    start_pred = time.time()
    y_pred = lgbm_model.predict(X_test)
    prediction_time = time.time() - start_pred
    
    # Calculate test accuracy
    test_accuracy = accuracy_score(y_test, y_pred)
    train_accuracy = lgbm_model.score(X_train, y_train)
    final_overfitting_gap = train_accuracy - test_accuracy
    
    print(f"\n📊 FINAL RESULTS:")
    print(f"Training completed in {training_time:.2f} seconds")
    print(f"Prediction completed in {prediction_time:.2f} seconds")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Final Overfitting Gap: {final_overfitting_gap:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return {
        'model': lgbm_model,
        'model_name': 'LightGBM',
        'test_accuracy': test_accuracy,
        'train_accuracy': train_accuracy,
        'final_overfitting_gap': final_overfitting_gap,
        'training_time': training_time,
        'prediction_time': prediction_time,
        'y_pred': y_pred,
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
        'feature_importance': lgbm_model.feature_importances_,
        'cv_results': cv_results,
        'best_hyperparameters': final_params,  # Added this for easy access
        'hyperparameter_search_history': {
            'phase1_best_params': random_search.best_params_,
            'phase1_best_score': random_search.best_score_,
            'phase2_best_params': final_params,
            'phase2_best_score': fine_search.best_score_,
            'score_improvement': score_improvement
        }
    }
    
def train_lightgbm(X_train, X_test, y_train, y_test, class_info):
    """
    Train and evaluate LightGBM model
    """    
    print("\n" + "="*50)
    print("Training LightGBM Classifier")
    print("="*50)
    
    start_time = time.time()
    is_multiclass = class_info['num_classes'] > 2
    
    # Configure LightGBM parameters
    if is_multiclass:
        lgb_params = {
            'max_depth': -1,
            'min_child_samples': 40,
            'subsample': 0.7,
            'colsample_bytree': 0.6,
            'learning_rate': 0.05,
            'reg_alpha': 0,
            'reg_lambda': 0,
            'class_weight': None
        }
    else:
        lgb_params = {
            'max_depth': -1,
            'min_child_samples': 40,
            'subsample': 0.7,
            'colsample_bytree': 0.6,
            'learning_rate': 0.05,
            'reg_alpha': 0,
            'reg_lambda': 0,
            'class_weight': None
        }
        
        
        # this for binary
        # ====================
        
        # lgb_params = {
        #     'objective': 'binary',
        #     'metric': 'binary_logloss',
        #     'boosting_type': 'gbdt',
        #     'num_leaves': 31,
        #     'learning_rate': 0.05,
        #     'feature_fraction': 0.8,
        #     'bagging_fraction': 0.8,
        #     'bagging_freq': 5,
        #     'min_child_samples': 20,
        #     'lambda_l1': 0.1,
        #     'lambda_l2': 0.1,
        #     'random_state': 42,
        #     'n_jobs': -1,
        # }
    
    # Initialize model
    lgb_model = lgb.LGBMClassifier(**lgb_params, n_estimators=100)
    
    # Enhanced cross-validation
    cv_results = enhanced_cross_validation(lgb_model, X_train, y_train, "LightGBM", is_multiclass=is_multiclass)
    
    # Train the model with early stopping
    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric='multi_logloss' if is_multiclass else 'binary_logloss',
        callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)]  # FIX: Use callbacks instead of early_stopping_rounds
    )
    
    training_time = time.time() - start_time
    
    # Make predictions
    start_pred = time.time()
    y_pred = lgb_model.predict(X_test)
    prediction_time = time.time() - start_pred
    
    # Calculate accuracies
    test_accuracy = accuracy_score(y_test, y_pred)
    train_accuracy = lgb_model.score(X_train, y_train)
    final_overfitting_gap = train_accuracy - test_accuracy
    
    print(f"Training completed in {training_time:.2f} seconds")
    print(f"Prediction completed in {prediction_time:.2f} seconds")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Final Overfitting Gap: {final_overfitting_gap:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return {
        'model': lgb_model,
        'model_name': 'LightGBM',
        'test_accuracy': test_accuracy,
        'train_accuracy': train_accuracy,
        'final_overfitting_gap': final_overfitting_gap,
        'training_time': training_time,
        'prediction_time': prediction_time,
        'y_pred': y_pred,
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
        'feature_importance': lgb_model.feature_importances_,
        'cv_results': cv_results
    }


def train_xgboost_optimized(X_train, X_test, y_train, y_test, class_info):
    """
    Train and evaluate XGBoost model with enhanced overfitting detection
    """    
    print("\n" + "="*50)
    print("Training XGBoost Classifier")
    print("="*50)
    
    start_time = time.time()
    is_multiclass = class_info['num_classes'] > 2
    
    # Initialize the model with regularization
    if is_multiclass:
        xgb_model = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=class_info['num_classes'],
            eval_metric='mlogloss',
            n_estimators=100,
            random_state=42,
            max_depth=6,
            min_child_weight=1,
            subsample=0.8,
            colsample_bytree=0.8,
            learning_rate=0.1,
            n_jobs=-1,
            verbosity=0
        )
    else:
        xgb_model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            n_estimators=100,
            random_state=42,
            max_depth=6,
            min_child_weight=1,
            subsample=0.8,
            colsample_bytree=0.8,
            learning_rate=0.1,
            n_jobs=-1,
            verbosity=0
        )
    
    # Strategy 1: Quick screening with RandomizedSearchCV
    print("🔍 Phase 1: Quick parameter screening...")
    
    # Broad parameter space for initial screening
    param_distributions = {            
        'n_estimators': [50, 100, 150, 200],
        'max_depth': [10, 15, 20, 25, -1],
        'min_child_samples': [10, 20, 30, 40],
        'subsample': [0.6, 0.7, 0.8, 0.9],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
        'learning_rate': [0.05, 0.1, 0.15, 0.2],
        'reg_alpha': [0, 0.1, 0.5, 1.0],
        'reg_lambda': [0, 0.1, 0.5, 1.0],
        'class_weight': [None, 'balanced']
            
    }
    
    # Base model for search
    if is_multiclass:
        xgb_base = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=class_info['num_classes'],
            eval_metric='mlogloss',
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
    else:
        xgb_base = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
    
    # Quick randomized search - test 20 combinations
    random_search = RandomizedSearchCV(
        xgb_base,
        param_distributions,
        n_iter=20,  # Test 20 random combinations
        cv=3,
        scoring='f1_macro' if is_multiclass else 'accuracy',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    random_search.fit(X_train, y_train)
    
    print(f"✅ Phase 1 completed in {time.time() - start_time:.1f}s")
    print(f"Best random search score: {random_search.best_score_:.4f}")
    
    # Print Phase 1 best parameters
    print("\n📊 Phase 1 - Best Parameters from Random Search:")
    print("-" * 45)
    for param, value in random_search.best_params_.items():
        print(f"  {param:<20}: {value}")
    print("-" * 45)
    
    # Strategy 2: Fine-tune around best parameters
    print("\n🎯 Phase 2: Fine-tuning best parameters...")
    
    phase2_start = time.time()
    best_params = random_search.best_params_
    
    # Create focused grid around best parameters
    fine_tune_grid = {}
    
    # Fine-tune n_estimators around best
    best_n_est = best_params.get('n_estimators', 100)
    fine_tune_grid['n_estimators'] = [
        max(50, best_n_est - 50),
        best_n_est,
        best_n_est + 50
    ]
    
    # Fine-tune max_depth around best
    best_depth = best_params.get('max_depth', 6)
    fine_tune_grid['max_depth'] = [
        max(3, best_depth - 1),
        best_depth,
        best_depth + 1
    ]
    
    # Fine-tune learning_rate around best
    best_lr = best_params.get('learning_rate', 0.1)
    fine_tune_grid['learning_rate'] = [
        max(0.01, best_lr - 0.05),
        best_lr,
        min(0.3, best_lr + 0.05)
    ]
    
    # Fine-tune min_child_weight around best
    best_mcw = best_params.get('min_child_weight', 1)
    fine_tune_grid['min_child_weight'] = [
        max(1, best_mcw - 1),
        best_mcw,
        best_mcw + 1
    ]
    
    # Keep other good parameters fixed or with minimal variation
    fine_tune_grid['subsample'] = [best_params.get('subsample', 0.8)]
    fine_tune_grid['colsample_bytree'] = [best_params.get('colsample_bytree', 0.8)]
    fine_tune_grid['reg_alpha'] = [best_params.get('reg_alpha', 0)]
    fine_tune_grid['reg_lambda'] = [best_params.get('reg_lambda', 0)]
    fine_tune_grid['gamma'] = [best_params.get('gamma', 0)]
    
    # Fine-tuning grid search
    try:
        fine_search = GridSearchCV(
            xgb_base,
            fine_tune_grid,
            cv=3,
            scoring='f1_macro' if is_multiclass else 'accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        fine_search.fit(X_train, y_train)
        
        print(f"✅ Phase 2 completed in {time.time() - phase2_start:.1f}s")
        print(f"Final best score: {fine_search.best_score_:.4f}")
        
        # Get final optimized model
        xgb_model = fine_search.best_estimator_
        final_params = fine_search.best_params_
        
        # Calculate score improvement
        score_improvement = fine_search.best_score_ - random_search.best_score_
        
    except Exception as e:
        print(f"⚠️ Error in Phase 2: {e}")
        print("Using best model from Phase 1 instead...")
        xgb_model = random_search.best_estimator_
        final_params = random_search.best_params_
        score_improvement = 0.0
    
    # Print Final Best Parameters in a formatted way
    print("\n🏆 FINAL OPTIMAL HYPERPARAMETERS:")
    print("=" * 50)
    print(f"{'Parameter':<20} | {'Value':<15} | {'Type'}")
    print("-" * 50)
    for param, value in sorted(final_params.items()):
        param_type = type(value).__name__
        print(f"{param:<20} | {str(value):<15} | {param_type}")
    print("=" * 50)
    
    # Also print improvement from Phase 1 to Phase 2
    print(f"\n📈 Score Improvement: {score_improvement:+.4f}")
    print(f"   Phase 1 Score: {random_search.best_score_:.4f}")
    if 'fine_search' in locals():
        print(f"   Phase 2 Score: {fine_search.best_score_:.4f}")
    else:
        print(f"   Phase 2 Score: {random_search.best_score_:.4f} (fallback)")
    
    # Enhanced cross-validation on the optimized model
    print("\n🔍 Performing enhanced cross-validation analysis...")
    cv_results = enhanced_cross_validation(xgb_model, X_train, y_train, "XGBoost", is_multiclass=is_multiclass)
    
    training_time = time.time() - start_time
    
    # Make predictions
    start_pred = time.time()
    y_pred = xgb_model.predict(X_test)
    prediction_time = time.time() - start_pred
    
    # Calculate test accuracy
    test_accuracy = accuracy_score(y_test, y_pred)
    train_accuracy = xgb_model.score(X_train, y_train)
    final_overfitting_gap = train_accuracy - test_accuracy
    
    print(f"\n📊 FINAL RESULTS:")
    print(f"Training completed in {training_time:.2f} seconds")
    print(f"Prediction completed in {prediction_time:.2f} seconds")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Final Overfitting Gap: {final_overfitting_gap:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return {
        'model': xgb_model,
        'model_name': 'XGBoost',
        'test_accuracy': test_accuracy,
        'train_accuracy': train_accuracy,
        'final_overfitting_gap': final_overfitting_gap,
        'training_time': training_time,
        'prediction_time': prediction_time,
        'y_pred': y_pred,
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
        'feature_importance': xgb_model.feature_importances_,
        'cv_results': cv_results,
        'best_hyperparameters': final_params,  # Added this for easy access
        'hyperparameter_search_history': {
            'phase1_best_params': random_search.best_params_,
            'phase1_best_score': random_search.best_score_,
            'phase2_best_params': final_params,
            'phase2_best_score': fine_search.best_score_ if 'fine_search' in locals() else random_search.best_score_,
            'score_improvement': score_improvement
        }
    }
    
def train_xgboost(X_train, X_test, y_train, y_test, class_info):
    """
    Train and evaluate XGBoost model
    """
    
    print("\n" + "="*50)
    print("Training XGBoost Classifier")
    print("="*50)
    
    start_time = time.time()
    is_multiclass = class_info['num_classes'] > 2
    
    # Configure XGBoost parameters
    if is_multiclass:
        xgb_params = {
            'max_depth': 3,
            'min_child_samples': 3,
            'subsample': 0.8,
            'colsample_bytree': 0.7,
            'learning_rate': 0.7,
            'reg_alpha': 0,
            'reg_lambda': 0.01,
            'class_weight': None
        }
    else:
        xgb_params = {
            'max_depth': 3,
            'min_child_samples': 3,
            'subsample': 0.8,
            'colsample_bytree': 0.7,
            'learning_rate': 0.7,
            'reg_alpha': 0,
            'reg_lambda': 0.01,
            'class_weight': None
        }
    
    # Initialize model
    xgb_model = xgb.XGBClassifier(**xgb_params, n_estimators=23)
    
    # Enhanced cross-validation
    cv_results = enhanced_cross_validation(xgb_model, X_train, y_train, "XGBoost", is_multiclass=is_multiclass)
    
    # Train the model with early stopping
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False  # FIX: Use verbose=False instead of early_stopping_rounds
    )
    
    training_time = time.time() - start_time
    
    # Make predictions
    start_pred = time.time()
    y_pred = xgb_model.predict(X_test)
    prediction_time = time.time() - start_pred
    
    # Calculate accuracies
    test_accuracy = accuracy_score(y_test, y_pred)
    train_accuracy = xgb_model.score(X_train, y_train)
    final_overfitting_gap = train_accuracy - test_accuracy
    
    print(f"Training completed in {training_time:.2f} seconds")
    print(f"Prediction completed in {prediction_time:.2f} seconds")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Final Overfitting Gap: {final_overfitting_gap:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return {
        'model': xgb_model,
        'model_name': 'XGBoost',
        'test_accuracy': test_accuracy,
        'train_accuracy': train_accuracy,
        'final_overfitting_gap': final_overfitting_gap,
        'training_time': training_time,
        'prediction_time': prediction_time,
        'y_pred': y_pred,
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
        'feature_importance': xgb_model.feature_importances_,
        'cv_results': cv_results
    }


def train_random_forest(X_train, X_test, y_train, y_test, class_info):
    """
    Train and evaluate Random Forest model with enhanced overfitting detection
    """
    print("\n" + "="*50)
    print("Training Random Forest Classifier")
    print("="*50)
    
    start_time = time.time()
    is_multiclass = class_info['num_classes'] > 2
    
   # Initialize the model with regularization
    rf_model = RandomForestClassifier(
        n_estimators=2,
        random_state=42,
        max_depth=25,
        min_samples_split=10,
        min_samples_leaf=10,
        max_features=None,
        bootstrap=True,
        oob_score=True,
        n_jobs=-1,
    )
    
    
    # rf_model = RandomForestClassifier(
    #     n_estimators=10,  
    #     max_depth=10,        
    #     min_samples_leaf=6,    
    #     min_samples_split=5,    
    #     random_state=42,
    #     bootstrap=True,
    #     oob_score=True,
    #     n_jobs=-1,
    # )
    
    
    
    """_summary_ 

        n_estimators=200,
        random_state=42,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=1,
        max_features=None,
        bootstrap=True,
        oob_score=True,
        n_jobs=-1,
    """
    
    # Enhanced cross-validation
    cv_results = enhanced_cross_validation(rf_model, X_train, y_train, "Random Forest", is_multiclass=is_multiclass)
    
    # Train the model
    rf_model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # Make predictions
    start_pred = time.time()
    y_pred = rf_model.predict(X_test)
    prediction_time = time.time() - start_pred
    
    # Calculate test accuracy
    test_accuracy = accuracy_score(y_test, y_pred)
    train_accuracy = rf_model.score(X_train, y_train)
    final_overfitting_gap = train_accuracy - test_accuracy
    oob_score = rf_model.oob_score_
    
    print(f"Training completed in {training_time:.2f} seconds")
    print(f"Prediction completed in {prediction_time:.2f} seconds")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Out-of-Bag Score: {oob_score:.4f}")
    print(f"Final Overfitting Gap: {final_overfitting_gap:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return {
        'model': rf_model,
        'model_name': 'Random Forest',
        'test_accuracy': test_accuracy,
        'train_accuracy': train_accuracy,
        'oob_score': oob_score,
        'final_overfitting_gap': final_overfitting_gap,
        'training_time': training_time,
        'prediction_time': prediction_time,
        'y_pred': y_pred,
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
        'feature_importance': rf_model.feature_importances_,
        'cv_results': cv_results
    }

def train_random_forest_Optimized(X_train, X_test, y_train, y_test, class_info):
    """
    Train and evaluate Random Forest model with enhanced overfitting detection
    """
    print("\n" + "="*50)
    print("Training Random Forest Classifier")
    print("="*50)
    
    start_time = time.time()
    is_multiclass = class_info['num_classes'] > 2
    
    # Initialize the model with regularization
    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        bootstrap=True,
        oob_score=True,
        n_jobs=-1
    )
    
    # Strategy 1: Quick screening with RandomizedSearchCV
    print("🔍 Phase 1: Quick parameter screening...")
    
    # Broad parameter space for initial screening
    param_distributions = {
        'n_estimators': [50, 100, 150, 200],
        'max_depth': [10, 15, 20, 25, None],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 4, 8],
        'max_features': ['sqrt', 'log2', None],
        'class_weight': [None, 'balanced'],
        'bootstrap': [True, False]              # Node splitting condition
    }
    
    
    
    """
    this is the better parameter to obtain the better solutino:
    final_params: {
        'bootstrap': True, 
        'class_weight': None, 
        'max_depth': 20,
        'max_features': None, 
        'min_samples_leaf': 1,
        'min_samples_split': 10, 
        'n_estimators': 200}
    """
    
    rf_base = RandomForestClassifier(
        random_state=42,
        n_jobs=-1,  # Use all 12 cores
        oob_score=True
    )
    
    # Quick randomized search - test 20 combinations
    random_search = RandomizedSearchCV(
        rf_base,
        param_distributions,
        n_iter=20,  # Test 20 random combinations
        cv=3,
        scoring='f1_macro' if is_multiclass else 'accuracy',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    random_search.fit(X_train, y_train)
    
    print(f"✅ Phase 1 completed in {time.time() - start_time:.1f}s")
    print(f"Best random search score: {random_search.best_score_:.4f}")
    
    # Strategy 2: Fine-tune around best parameters
    print("🎯 Phase 2: Fine-tuning best parameters...")
    
    phase2_start = time.time()
    best_params = random_search.best_params_
    
    # Create focused grid around best parameters
    fine_tune_grid = {}
    
    # Fine-tune n_estimators around best
    best_n_est = best_params['n_estimators']
    fine_tune_grid['n_estimators'] = [
        max(50, best_n_est - 50),
        best_n_est,
        best_n_est + 50
    ]
    
    # Fine-tune max_depth around best
    if best_params['max_depth'] is not None:
        best_depth = best_params['max_depth']
        fine_tune_grid['max_depth'] = [
            max(5, best_depth - 5),
            best_depth,
            best_depth + 5,
            None
        ]
    else:
        fine_tune_grid['max_depth'] = [20, 25, None]
    
    # Keep other good parameters fixed or with minimal variation
    fine_tune_grid['min_samples_split'] = [best_params['min_samples_split']]
    fine_tune_grid['min_samples_leaf'] = [best_params['min_samples_leaf']]
    fine_tune_grid['max_features'] = [best_params['max_features']]
    fine_tune_grid['class_weight'] = [best_params['class_weight']]
    fine_tune_grid['bootstrap'] = [best_params['bootstrap']]
    
    # Fine-tuning grid search
    fine_search = GridSearchCV(
        rf_base,
        fine_tune_grid,
        cv=3,
        scoring='f1_macro' if is_multiclass else 'accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    fine_search.fit(X_train, y_train)
    
    print(f"✅ Phase 2 completed in {time.time() - phase2_start:.1f}s")
    print(f"Final best score: {fine_search.best_score_:.4f}")
    
    # Get final optimized model
    rf_model = fine_search.best_estimator_
    final_params = fine_search.best_params_
    
    print("final_params:",final_params)
    
    # Enhanced cross-validation on the optimized model
    print("🔍 Performing enhanced cross-validation analysis...")
    cv_results = enhanced_cross_validation(rf_model, X_train, y_train, "Random Forest", is_multiclass=is_multiclass)
    
    training_time = time.time() - start_time
    
    # Make predictions
    start_pred = time.time()
    y_pred = rf_model.predict(X_test)
    prediction_time = time.time() - start_pred
    
    # Calculate test accuracy
    test_accuracy = accuracy_score(y_test, y_pred)
    train_accuracy = rf_model.score(X_train, y_train)
    final_overfitting_gap = train_accuracy - test_accuracy
    oob_score = rf_model.oob_score_
    
    print(f"Training completed in {training_time:.2f} seconds")
    print(f"Prediction completed in {prediction_time:.2f} seconds")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Out-of-Bag Score: {oob_score:.4f}")
    print(f"Final Overfitting Gap: {final_overfitting_gap:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return {
        'model': rf_model,
        'model_name': 'Random Forest',
        'test_accuracy': test_accuracy,
        'train_accuracy': train_accuracy,
        'oob_score': oob_score,
        'final_overfitting_gap': final_overfitting_gap,
        'training_time': training_time,
        'prediction_time': prediction_time,
        'y_pred': y_pred,
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
        'feature_importance': rf_model.feature_importances_,
        'cv_results': cv_results
    }



def calculate_hardware_metrics(model):
    """
    Calculate hardware requirements for different model types
    """
    import sys
    import pickle
    
    model_type = type(model).__name__
    
    if model_type == 'RandomForestClassifier':
        return calculate_rf_metrics(model)
    elif model_type == 'LGBMClassifier':
        return calculate_lightgbm_metrics(model)
    elif model_type == 'XGBClassifier':
        return calculate_xgboost_metrics(model)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def calculate_rf_metrics(rf_model):
    """Calculate hardware metrics for Random Forest"""
    import sys
    import pickle
    
    # Random Forest specific calculations
    total_nodes = sum([tree.tree_.node_count for tree in rf_model.estimators_])
    max_depth = max([tree.tree_.max_depth for tree in rf_model.estimators_])
    n_estimators = rf_model.n_estimators
    
    # Model size estimation
    model_size_bytes = sys.getsizeof(pickle.dumps(rf_model))
    flash_MB = model_size_bytes / (1024 * 1024)
    
    # RAM usage during inference
    ram_MB = flash_MB * 1.5
    
    # Computational operations
    decision_ops = total_nodes * 2  
    
    return {
        'total_nodes': total_nodes,
        'max_depth': max_depth,
        'n_estimators': n_estimators,
        'flash_MB': flash_MB,
        'ram_MB': ram_MB,
        'decision_ops': decision_ops
    }

def calculate_lightgbm_metrics(lgbm_model):
    """Calculate hardware metrics for LightGBM"""
    import sys
    import pickle
    
    # LightGBM specific calculations
    n_estimators = lgbm_model.n_estimators
    max_depth = lgbm_model.max_depth if lgbm_model.max_depth > 0 else -1
    
    # Estimate total nodes (LightGBM is more efficient than RF)
    avg_nodes_per_tree = 63 if max_depth == -1 else (2 ** min(max_depth, 10)) - 1
    total_nodes = n_estimators * avg_nodes_per_tree
    
    # Model size estimation
    model_size_bytes = sys.getsizeof(pickle.dumps(lgbm_model))
    flash_MB = model_size_bytes / (1024 * 1024)
    
    # RAM usage during inference (LightGBM is memory efficient)
    ram_MB = flash_MB * 1.2
    
    # Computational operations (LightGBM is optimized)
    decision_ops = total_nodes * 1.5
    
    return {
        'total_nodes': total_nodes,
        'max_depth': max_depth,
        'n_estimators': n_estimators,
        'flash_MB': flash_MB,
        'ram_MB': ram_MB,
        'decision_ops': decision_ops
    }

def calculate_xgboost_metrics(xgb_model):
    """Calculate hardware metrics for XGBoost"""
    import sys
    import pickle
    
    # XGBoost specific calculations
    n_estimators = xgb_model.n_estimators
    max_depth = xgb_model.max_depth
    
    # Estimate total nodes (XGBoost builds balanced trees)
    avg_nodes_per_tree = (2 ** max_depth) - 1
    total_nodes = n_estimators * avg_nodes_per_tree
    
    # Model size estimation
    model_size_bytes = sys.getsizeof(pickle.dumps(xgb_model))
    flash_MB = model_size_bytes / (1024 * 1024)
    
    # RAM usage during inference
    ram_MB = flash_MB * 1.3
    
    # Computational operations
    decision_ops = total_nodes * 1.8
    
    return {
        'total_nodes': total_nodes,
        'max_depth': max_depth,
        'n_estimators': n_estimators,
        'flash_MB': flash_MB,
        'ram_MB': ram_MB,
        'decision_ops': decision_ops
    }

# Plot the results

def plot_confusion_matrices(models_results, y_test):
    """
    Plot confusion matrices for all models
    """
    # Filter out None results
    valid_results = [result for result in models_results if result is not None]
    
    if not valid_results:
        print("No valid model results to plot")
        return
    
    num_models = len(valid_results)
    cols = 2
    rows = (num_models + cols - 1) // cols  # Ceiling division
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, 6*rows))
    if rows == 1:
        axes = [axes] if cols == 1 else axes
    else:
        axes = axes.flatten()
    
    for i, result in enumerate(valid_results):
        cm = confusion_matrix(y_test, result['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[i], cmap='Blues')
        axes[i].set_title(f'{result["model_name"]} - Confusion Matrix')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')
    
    # Hide unused subplots
    for i in range(num_models, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_overfitting_analysis(models_results):
    """
    Plot overfitting analysis for all models
    """
    # Filter out None results
    valid_results = [result for result in models_results if result is not None]
    
    if not valid_results:
        print("No valid model results to plot")
        return
    
    model_names = [result['model_name'] for result in valid_results]
    train_scores = [result['train_accuracy'] for result in valid_results]
    test_scores = [result['test_accuracy'] for result in valid_results]
    overfitting_gaps = [result['final_overfitting_gap'] for result in valid_results]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Training vs Test Accuracy
    x = np.arange(len(model_names))
    width = 0.35
    
    ax1.bar(x - width/2, train_scores, width, label='Training Accuracy', alpha=0.8)
    ax1.bar(x + width/2, test_scores, width, label='Test Accuracy', alpha=0.8)
    ax1.set_xlabel('Models')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Training vs Test Accuracy')
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Overfitting Gap
    colors = ['red' if gap > 0.05 else 'orange' if gap > 0.02 else 'green' for gap in overfitting_gaps]
    ax2.bar(x, overfitting_gaps, color=colors, alpha=0.8)
    ax2.set_xlabel('Models')
    ax2.set_ylabel('Overfitting Gap')
    ax2.set_title('Overfitting Analysis')
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_names, rotation=45, ha='right')
    ax2.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='Significant Overfitting')
    ax2.axhline(y=0.02, color='orange', linestyle='--', alpha=0.7, label='Mild Overfitting')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('overfitting_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_feature_importance(rf_result, feature_names, top_n=20):
    """
    Plot feature importance for Random Forest
    """
    if rf_result is None or 'feature_importance' not in rf_result:
        print("No feature importance data available")
        return
        
    # Get top N features
    importance = rf_result['feature_importance']
    indices = np.argsort(importance)[::-1][:top_n]
    
    plt.figure(figsize=(12, 8))
    plt.title(f'Top {top_n} Feature Importance - Random Forest')
    plt.bar(range(top_n), importance[indices])
    plt.xticks(range(top_n), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()

def compare_models(models_results):
    """
    Compare performance of all models with enhanced overfitting metrics
    
    Args:
        models_results (list): List of model results
    """
    # Filter out None results
    valid_results = [result for result in models_results if result is not None]
    
    if not valid_results:
        print("No valid model results to compare")
        return pd.DataFrame()
    
    print("\n" + "="*80)
    print("MODEL PERFORMANCE COMPARISON WITH OVERFITTING ANALYSIS")
    print("="*80)
    
    # Create comparison dataframe
    comparison_data = []
    for result in valid_results:
        cv_results = result['cv_results']
        comparison_data.append({
            'Model': result['model_name'],
            'Test Accuracy': result['test_accuracy'],
            'Train Accuracy': result['train_accuracy'],
            'Overfitting Gap': result['final_overfitting_gap'],
            'CV Mean': cv_results['cv_accuracy_mean'],
            'CV Std': cv_results['cv_accuracy_std'],
            'CV Variance': cv_results['cv_variance'],
            'Training Time (s)': result['training_time'],
            'Prediction Time (s)': result['prediction_time']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False, float_format='%.4f'))
    
    # Find best model based on test accuracy and overfitting
    print("\n" + "="*80)
    print("MODEL RANKING ANALYSIS")
    print("="*80)
    
    # Best test accuracy
    best_accuracy_idx = comparison_df['Test Accuracy'].idxmax()
    best_accuracy_model = comparison_df.iloc[best_accuracy_idx]
    
    # Least overfitting
    least_overfitting_idx = comparison_df['Overfitting Gap'].idxmin()
    least_overfitting_model = comparison_df.iloc[least_overfitting_idx]
    
    # Most stable (lowest CV variance)
    most_stable_idx = comparison_df['CV Variance'].idxmin()
    most_stable_model = comparison_df.iloc[most_stable_idx]
    
    print(f"🎯 HIGHEST TEST ACCURACY: {best_accuracy_model['Model']}")
    print(f"   Test Accuracy: {best_accuracy_model['Test Accuracy']:.4f}")
    print(f"   Overfitting Gap: {best_accuracy_model['Overfitting Gap']:.4f}")
    
    print(f"\n✅ LEAST OVERFITTING: {least_overfitting_model['Model']}")
    print(f"   Overfitting Gap: {least_overfitting_model['Overfitting Gap']:.4f}")
    print(f"   Test Accuracy: {least_overfitting_model['Test Accuracy']:.4f}")
    
    print(f"\n📊 MOST STABLE: {most_stable_model['Model']}")
    print(f"   CV Variance: {most_stable_model['CV Variance']:.6f}")
    print(f"   Test Accuracy: {most_stable_model['Test Accuracy']:.4f}")
    
    # Overall recommendation
    print(f"\n🏆 RECOMMENDED MODEL:")
    # Score models based on test accuracy and inverse overfitting gap
    comparison_df['Combined Score'] = (
        comparison_df['Test Accuracy'] - 
        comparison_df['Overfitting Gap'] * 0.5 -  # Penalty for overfitting
        comparison_df['CV Variance'] * 10  # Penalty for instability
    )
    
    best_overall_idx = comparison_df['Combined Score'].idxmax()
    best_overall_model = comparison_df.iloc[best_overall_idx]
    
    print(f"   {best_overall_model['Model']}")
    print(f"   Test Accuracy: {best_overall_model['Test Accuracy']:.4f}")
    print(f"   Overfitting Gap: {best_overall_model['Overfitting Gap']:.4f}")
    print(f"   CV Variance: {best_overall_model['CV Variance']:.6f}")
    
    # Overfitting warnings
    print("\n" + "="*80)
    print("OVERFITTING WARNINGS")
    print("="*80)
    
    for _, row in comparison_df.iterrows():
        gap = row['Overfitting Gap']
        variance = row['CV Variance']
        
        if gap > 0.1:
            print(f"🚨 CRITICAL: {row['Model']} - Severe overfitting (gap: {gap:.4f})")
        elif gap > 0.05:
            print(f"⚠️  WARNING: {row['Model']} - Significant overfitting (gap: {gap:.4f})")
        elif gap > 0.02:
            print(f"⚠️  CAUTION: {row['Model']} - Mild overfitting (gap: {gap:.4f})")
        else:
            print(f"✅ GOOD: {row['Model']} - No significant overfitting (gap: {gap:.4f})")
        
        if variance > 0.01:
            print(f"   ⚠️  High variance detected - model may be unstable")
    
    return comparison_df

def save_best_model(models_results, comparison_df):
    """
    Save the best performing model (considering overfitting)
    """
    # Filter out None results
    valid_results = [result for result in models_results if result is not None]
    
    if not valid_results or comparison_df.empty:
        print("No valid models to save")
        return None
    
    # Use the model with best combined score (accuracy - overfitting penalty)
    best_model_idx = comparison_df['Combined Score'].idxmax()
    best_model_result = valid_results[best_model_idx]
    best_model = best_model_result['model']
    model_name = best_model_result['model_name'].replace(' ', '_').lower()
    
    filename = f'best_model_{model_name}.joblib'
    joblib.dump(best_model, filename)
    print(f"\n💾 Best model saved as: {filename}")
    print(f"   Model: {best_model_result['model_name']}")
    print(f"   Test Accuracy: {best_model_result['test_accuracy']:.4f}")
    print(f"   Overfitting Gap: {best_model_result['final_overfitting_gap']:.4f}")
    
    return filename

def generate_overfitting_recommendations(models_results, comparison_df):
    """
    Generate specific recommendations to address overfitting
    """
    # Filter out None results
    valid_results = [result for result in models_results if result is not None]
    
    if not valid_results:
        print("No valid model results for recommendations")
        return
    
    print("\n" + "="*80)
    print("OVERFITTING PREVENTION RECOMMENDATIONS")
    print("="*80)
    
    # General recommendations
    print("🔧 GENERAL RECOMMENDATIONS:")
    print("   • Use cross-validation for all model selections")
    print("   • Monitor training vs validation performance")
    print("   • Implement early stopping when possible")
    print("   • Use regularization techniques")
    print("   • Collect more training data if possible")
    
    # Model-specific recommendations
    print("\n🎯 MODEL-SPECIFIC RECOMMENDATIONS:")
    
    for result in valid_results:
        model_name = result['model_name']
        gap = result['final_overfitting_gap']
        
        print(f"\n{model_name}:")
        if gap > 0.05:
            if 'Decision Tree' in model_name:
                print("   • Increase min_samples_split and min_samples_leaf")
                print("   • Reduce max_depth")
                print("   • Use cost complexity pruning (ccp_alpha)")
            elif 'Random Forest' in model_name:
                print("   • Reduce max_depth")
                print("   • Increase min_samples_split")
                print("   • Use fewer features (max_features)")
            elif 'KNN' in model_name:
                print("   • Increase k value")
                print("   • Use distance weighting")
                print("   • Apply feature selection")
            elif 'SVM' in model_name:
                print("   • Reduce C parameter (increase regularization)")
                print("   • Use simpler kernel (linear instead of RBF)")
                print("   • Apply feature scaling")
            elif 'LightGBM' in model_name:
                print("   • Increase min_child_samples")
                print("   • Reduce num_leaves")
                print("   • Increase regularization (lambda_l1, lambda_l2)")
            elif 'XGBoost' in model_name:
                print("   • Increase min_child_weight")
                print("   • Reduce max_depth")
                print("   • Increase regularization (reg_alpha, reg_lambda)")
        else:
            print("   ✅ Good generalization - no specific changes needed")
    
    # Data recommendations
    print("\n📊 DATA RECOMMENDATIONS:")
    high_overfitting_models = [r for r in valid_results if r['final_overfitting_gap'] > 0.05]
    
    if high_overfitting_models:
        print("   • Consider data augmentation techniques")
        print("   • Implement feature selection to reduce dimensionality")
        print("   • Check for data leakage")
        print("   • Ensure proper train/validation/test split")
        print("   • Balance classes if severe imbalance exists")
    else:
        print("   ✅ Current data preprocessing appears adequate")
    
    # Deployment recommendations
    print("\n🚀 DEPLOYMENT RECOMMENDATIONS:")
    if not comparison_df.empty:
        best_model_name = comparison_df.loc[comparison_df['Combined Score'].idxmax(), 'Model']
        print(f"   • Deploy {best_model_name} for production")
        print("   • Implement monitoring for performance drift")
        print("   • Set up alerts for accuracy degradation")
        print("   • Plan for model retraining schedule")
        print("   • Use A/B testing for model updates")




def main():

    file_path = "Result of ML Data 47 Features/Dataset Deploable/ML_Depoable_Features_With 6-Class.csv"
    
    classification_types = ['binary']
    
    # Store results for table
    results_data = []
    test_accuracies = []
    
    for class_type in classification_types:
        print(f"\n🎯 Running {class_type} classification experiment...")
        
        X_train, X_test, y_train, y_test, feature_names, label_encoder, scaler, class_info = load_and_prepare_data(file_path, class_type)
        
        if hasattr(X_train, 'values'):
            X_train = X_train.values
            X_test = X_test.values
        if hasattr(y_train, 'values'):
            y_train = y_train.values
            y_test = y_test.values
        train_xgboost_optimized
        #model_result = train_random_forest(X_train, X_test, y_train, y_test, class_info)
        #model_result = train_xgboost(X_train, X_test, y_train, y_test, class_info)
        model_result = train_lightgbm(X_train, X_test, y_train, y_test, class_info)
        
        #Generate visualizations
        try:
            print("\n📊 Generating visualizations...")
            plot_confusion_matrices([model_result], y_test)
            plot_overfitting_analysis([model_result])
            
        except Exception as e:
            print(f"Visualization error: {e}")
        
        # Store test accuracy
        test_accuracies.append(model_result['test_accuracy'])
        
        # Extract CV metrics
        cv_accuracy_std = model_result['cv_results']['cv_accuracy_std']
        cv_f1_mean = model_result['cv_results']['cv_f1_mean']
        cv_f1_std = model_result['cv_results'].get('cv_f1_std', 0)  # If not available
        
        hardware_metrics = calculate_hardware_metrics(model_result['model'])
        
        # Store data for final table
        results_data.append({
            'Model': class_type,
            'Test_Acc': model_result['test_accuracy'],
            'F1_Score': cv_f1_mean,
            'Std_Acc': cv_accuracy_std,
            'Std_F1': cv_f1_std,
            'Flash_MB': hardware_metrics['flash_MB'],
            'RAM_MB': hardware_metrics['ram_MB'],
            'FLOPs': hardware_metrics['decision_ops']
        })
        
        print(f"\n📊 Results for {class_type}:")
        print(f"Test Accuracy: {model_result['test_accuracy']:.4f}")
        print(f"CV Accuracy Std: {cv_accuracy_std:.4f}")
        print(f"Total Nodes: {hardware_metrics['total_nodes']}")
        print(f"Max Depth: {hardware_metrics['max_depth']}")
        print(f"Decision Ops: {hardware_metrics['decision_ops']}")
        print(f"Flash MB: {hardware_metrics['flash_MB']:.4f}")
        print(f"RAM MB: {hardware_metrics['ram_MB']:.4f}")
    
    # Calculate standard deviation for test accuracy
    test_accuracy_mean = np.mean(test_accuracies)
    test_accuracy_std = np.std(test_accuracies)
    
    print(f"\n📈 TEST ACCURACY ANALYSIS:")
    print("="*50)
    print(f"Test Accuracy Mean: {test_accuracy_mean:.4f}")
    print(f"Test Accuracy Std:  {test_accuracy_std:.4f}")
    print(f"Test Accuracy Range: {test_accuracy_mean:.4f} ± {test_accuracy_std:.4f}")
    
    # Show individual accuracies
    print(f"\nIndividual Test Accuracies:")
    for i, (class_type, accuracy) in enumerate(zip(classification_types, test_accuracies)):
        print(f"  {class_type}: {accuracy:.4f}")
    
    # Summary Table
    print(f"\n📋 SUMMARY TABLE:")
    print("="*90)
    print(f"{'Model':<10} {'Test_Acc':<10} {'F1_Score':<10} {'Std_Acc':<10} {'Std_F1':<10} {'Flash_MB':<10} {'RAM_MB':<10} {'FLOPs':<10}")
    print("-" * 90)
    for data in results_data:
        print(f"{data['Model']:<10} {data['Test_Acc']:<10.4f} {data['F1_Score']:<10.4f} {data['Std_Acc']:<10.4f} {data['Std_F1']:<10.4f} {data['Flash_MB']:<10.4f} {data['RAM_MB']:<10.4f} {data['FLOPs']:<10}")
    
    return test_accuracies

if __name__ == "__main__":
    main()
    
    
    
    
    


