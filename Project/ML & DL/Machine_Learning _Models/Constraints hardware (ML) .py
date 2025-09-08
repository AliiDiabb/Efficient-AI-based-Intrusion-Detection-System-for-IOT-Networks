from sklearn.datasets import make_classification
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, ParameterGrid,cross_val_score, StratifiedKFold, validation_curve, learning_curve, GridSearchCV,RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import warnings

warnings.filterwarnings('ignore')

# Import gradient boosting models
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    print("⚠️  LightGBM not available. Install with: pip install lightgbm")
    LIGHTGBM_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    print("⚠️  XGBoost not available. Install with: pip install xgboost")
    XGBOOST_AVAILABLE = False

# 🧪 Step 1: Simulated Dataset Setup
# ================================

#Load and prepare data 
def load_and_prepare_data(file_path, classification_type='binary', sample_size=156986):
    """
    Load EdgeIIoTset dataset, prepare for training, and sample 2000 real data points.
    
    Args:
        file_path (str): Path to the EdgeIIoTset CSV file
        classification_type (str): 'binary', '6-class', or '15-class'
        sample_size (int): Number of samples to select from the dataset
    
    Returns:
        tuple: X_train, X_test, y_train, y_test, feature_names, label_encoder, scaler, class_info
    """
    print(f"Loading EdgeIIoTset dataset for {classification_type} classification...")
    df = pd.read_csv(file_path)
    print(f"Dataset loaded successfully. Shape: {df.shape}")

    print(f"\nDataset Info:")
    print(f"Total samples: {len(df)}")
    print(f"Total features: {df.shape[1]}")
    print(f"Missing values: {df.isnull().sum().sum()}")

    # Determine target column
    if classification_type == 'binary':
        if 'Attack_label' in df.columns:
            target_col = 'Attack_label'
            print("Using 'Attack_label' for binary classification (Normal vs Attack)")
        elif 'Attack_type' in df.columns:
            df['Attack_label'] = (df['Attack_type'] != 'Normal').astype(int)
            target_col = 'Attack_label'
        else:
            raise ValueError("Neither Attack_label nor Attack_type found in dataset")

    elif classification_type == '6-class':
        if 'Attack_type' not in df.columns:
            raise ValueError("Attack_type column not found for 6-class classification")
        target_col = '6_Attack'
        print("Using 'Attack_type' for 6-class classification")
       

    elif classification_type == '15-class':
        if 'Attack_type' not in df.columns:
            raise ValueError("Attack_type column not found for 15-class classification")
        target_col = 'Attack_type'
        print("Using 'Attack_type' for 15-class classification")

    else:
        raise ValueError("classification_type must be 'binary', '6-class', or '15-class'")

    print(f"Using '{target_col}' as target variable for {classification_type} classification")

    # Separate features and target
    feature_cols = [col for col in df.columns if col not in ['Attack_label', 'Attack_type','6_Attack', '6_Attack_text']]
    X = df[feature_cols]
    y = df[target_col]

    # Handle non-numeric features
    for col in X.columns:
        if X[col].dtype == 'object':
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))

    # Encode target
    label_encoder = None
    if y.dtype == 'object':
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
        class_names = label_encoder.classes_
    else:
        class_names = np.unique(y)

    print(f"Features shape before sampling: {X.shape}")
    print(f"Target distribution before sampling:")

    target_counts = pd.Series(y).value_counts().sort_index()
    if label_encoder:
        for i, count in enumerate(target_counts):
            print(f"  {class_names[i]}: {count} samples")
    else:
        print(target_counts)

    # Stratified sampling of 2000 samples
    if sample_size < len(X):
        X, _, y, _ = train_test_split(
            X, y, train_size=sample_size, stratify=y, random_state=42
        )
        print(f"Sampled {sample_size} rows with stratified class distribution.")
    else:
        print("⚠️ Requested sample_size exceeds dataset size. Using full dataset.")

    # Check for imbalance
    sampled_counts = pd.Series(y).value_counts().sort_index()
    imbalance_ratio = sampled_counts.min() / sampled_counts.max()
    print(f"Class imbalance ratio: {imbalance_ratio:.3f}")
    if imbalance_ratio < 0.1:
        print("⚠️  Severe class imbalance detected.")
    elif imbalance_ratio < 0.3:
        print("⚠️  Moderate class imbalance detected.")

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=42
    )

    print(f"\nFinal Data Split:")
    print(f"Training: {X_train.shape[0]} samples")
    print(f"Testing : {X_test.shape[0]} samples")

    class_info = {
        'num_classes': len(class_names),
        'class_names': class_names,
        'target_col': target_col,
        'classification_type': classification_type
    }

    return X_train, X_test, y_train, y_test, X.columns.tolist(), label_encoder, scaler, class_info

# ================================
# 🧠 Step 2: Hardware Estimator for Tree Models
# ================================
def hw_measures_tree_model(model, model_type='random_forest', input_dim=None):
    """
    Estimate hardware requirements for different tree-based models
    """
    if model_type == 'random_forest':
        estimators = model.estimators_
        total_nodes = sum(est.tree_.node_count for est in estimators)
        max_depth = max(est.tree_.max_depth for est in estimators)
        
    elif model_type == 'lightgbm':
        # LightGBM model analysis
        try:
            booster = model.booster_
            model_str = booster.model_to_string()
            
            # Parse model string to estimate nodes
            lines = model_str.split('\n')
            total_nodes = len([line for line in lines if 'split_feature' in line or 'leaf_value' in line])
            
            # Get max depth from model params
            max_depth = model.params.get('max_depth', 6) if hasattr(model, 'params') else 6
            
        except:
            # Fallback estimation
            n_estimators = model.n_estimators if hasattr(model, 'n_estimators') else 100
            total_nodes = n_estimators * 50  # Rough estimate
            max_depth = 6
            
    elif model_type == 'xgboost':
        # XGBoost model analysis
        try:
            total_nodes = 0
            max_depth = 0
            
            # Get model dump
            trees = model.get_booster().get_dump()
            
            for tree in trees:
                # Count nodes in tree string representation
                nodes_in_tree = tree.count('[') + tree.count('leaf=')
                total_nodes += nodes_in_tree
                
                # Estimate depth from indentation
                depth = max([line.count('\t') for line in tree.split('\n')] + [0])
                max_depth = max(max_depth, depth)
                
        except:
            # Fallback estimation
            n_estimators = model.n_estimators if hasattr(model, 'n_estimators') else 100
            total_nodes = n_estimators * 50  # Rough estimate
            max_depth = 6
            
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    estimated_decision_ops = total_nodes
    flash_bytes = total_nodes * 64  # Approx. 64 bytes per node
    ram_bytes = max_depth * (4 * input_dim if input_dim else 32)

    return {
        'total_nodes': total_nodes,
        'max_depth': max_depth,
        'decision_ops': estimated_decision_ops,
        'flash_MB': flash_bytes / (1024 ** 2),
        'ram_MB': ram_bytes / (1024 ** 2)
    }

def evaluate_model(model_type, params, X_train, X_test, y_train, y_test, hw_constraints):
    """
    Train and evaluate a specific model type with given parameters
    """
    try:
        if model_type == 'random_forest':
            model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
            
        elif model_type == 'lightgbm':
            if not LIGHTGBM_AVAILABLE:
                return None
            # Configure LightGBM to suppress warnings
            lgb_params = {
                **params,
                'random_state': 42,
                'n_jobs': -1,
                'verbose': -1,  # Suppress output
                'force_row_wise': True,  # Remove threading overhead warning
            }
            model = lgb.LGBMClassifier(**lgb_params)
            
        elif model_type == 'xgboost':
            if not XGBOOST_AVAILABLE:
                return None
            xgb_params = {
                **params,
                'random_state': 42,
                'n_jobs': -1,
                'verbosity': 0,  # Suppress output
            }
            model = xgb.XGBClassifier(**xgb_params)
            
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Train model
        model.fit(X_train, y_train)

        # Evaluate performance and resource usage
        hw = hw_measures_tree_model(model, model_type, input_dim=X_train.shape[1])
        acc = accuracy_score(y_test, model.predict(X_test))

        # Check if it fits within hardware limits
        fits_constraints = (
            hw['flash_MB'] <= hw_constraints['flash_MB'] and
            hw['ram_MB'] <= hw_constraints['ram_MB'] and
            hw['decision_ops'] <= hw_constraints['decision_ops']
        )

        return {
            'model_type': model_type,
            'params': params,
            'accuracy': round(acc, 4),
            'fits_constraints': fits_constraints,
            **{k: round(v, 4) for k, v in hw.items()}
        }

    except Exception as e:
        print(f"[!] Error evaluating {model_type} with params {params}: {e}")
        return None

def main():
    """
    Main execution function comparing Random Forest, LightGBM, and XGBoost
    """
    print("🔒 EdgeIIoTset Network Security Analysis")
    print("🤖 Comparing Random Forest, LightGBM, and XGBoost Models")
    print("="*80)
    
    # Check available libraries
    available_models = ['random_forest']
    if LIGHTGBM_AVAILABLE:
        available_models.append('lightgbm')
    if XGBOOST_AVAILABLE:
        available_models.append('xgboost')
    
    print(f"Available models: {', '.join(available_models)}")
    
    # FIX: Update file path to a more common location
    file_path = "../Result of ML Data 47 Features/Dataset Deploable/ML_Depoable_Features_With 6-Class.csv"  

    
    # Load and prepare data
    X_train, X_test, y_train, y_test, feature_names, label_encoder, scaler, class_info = load_and_prepare_data(
        file_path,
        classification_type='binary',
        sample_size=156986  #this is the real size of the dataset
    )        
    
    # ================================
    # ⚙️ Step 3: Hardware Constraints
    # ================================
    HW_CONSTRAINTS = {
        'flash_MB': 0.3,
        'ram_MB': 0.05,
        'decision_ops': 1_500_000
    }
    
    """
    Flash memory: 200,000,000 nodes × 64 bytes ≈ 12,800 MB
    RAM: 20 depth × (4 × input_dim) ≈ 2-10 MB depending on feature count
    Decision operations: ~200,000,000
    """

    print(f"\n🔧 Hardware Constraints:")
    print(f"Flash Memory: {HW_CONSTRAINTS['flash_MB']} MB")
    print(f"RAM Memory: {HW_CONSTRAINTS['ram_MB']} MB")
    print(f"Decision Ops: {HW_CONSTRAINTS['decision_ops']:,}")

    # ================================
    # 🔍 Step 4: Model-Specific Parameter Grids
    # ================================
    param_grids = {
        'random_forest': {
            'n_estimators': 2,          # Single value instead of list
            'max_depth': 25,              # Single value instead of list
            'min_samples_leaf': 10,      # Single value instead of list
            'max_features': None,      # Single value instead of list
            'min_samples_split': 10,    # this constraint the Hardware
            'bootstrap': True,
            'oob_score': True,
            'class_weight': None,

            
            },
        'lightgbm': {
            'n_estimators': [100],
            'max_depth': [-1],
            'min_child_samples': [40],
            'subsample': [0.7],
            'colsample_bytree': [0.6],
            'learning_rate': [0.05],
            'reg_alpha': [0],
            'reg_lambda': [0],
            'class_weight': [None]
        },
        #you can update this to ensure 94.96% instead 94.93%
        'xgboost': {
            'n_estimators': [23],
            'max_depth': [3],
            'min_child_samples': [3],
            'subsample': [0.8],
            'colsample_bytree': [0.7],
            'learning_rate': [0.7],
            'reg_alpha': [0],
            'reg_lambda': [0.01],
            'class_weight': [None],
            }
    }

    # ================================
    # 🚀 Step 5: Evaluate All Models
    # ================================
    all_results = []
    constrained_results = []

    for model_type in available_models:
        print(f"\n🔍 Evaluating {model_type.upper()} models...")
        
        # Handle Random Forest differently (single configuration)
        if model_type == 'random_forest':
            params = param_grids[model_type]
            print(f"Testing single Random Forest configuration...")
            
            result = evaluate_model(model_type, params, X_train, X_test, y_train, y_test, HW_CONSTRAINTS)
            
            if result:
                all_results.append(result)
                if result['fits_constraints']:
                    constrained_results.append(result)
                
                print(f"  Random Forest: Accuracy={result['accuracy']:.4f}, "
                      f"Constraints={'✓' if result['fits_constraints'] else '✗'}")
        
        # Handle LightGBM and XGBoost with parameter grids
        else:
            if model_type in param_grids:
                grid = list(ParameterGrid(param_grids[model_type]))
            else:
                continue
                
            print(f"Testing {len(grid)} parameter combinations...")
            
            model_results = []
            for i, params in enumerate(grid):
                result = evaluate_model(model_type, params, X_train, X_test, y_train, y_test, HW_CONSTRAINTS)
                
                if result:
                    all_results.append(result)
                    model_results.append(result)
                    
                    if result['fits_constraints']:
                        constrained_results.append(result)
                
                # Progress indicator
                if (i + 1) % 10 == 0:
                    print(f"  Progress: {i + 1}/{len(grid)} completed")
            
            # Show best model for this type
            if model_results:
                best_model = max(model_results, key=lambda x: x['accuracy'])
                print(f"  Best {model_type}: Accuracy={best_model['accuracy']:.4f}, "
                      f"Constraints={'✓' if best_model['fits_constraints'] else '✗'}")

    # ================================
    # 📊 Step 6: Results Analysis
    # ================================
    print(f"\n📊 Overall Analysis:")
    print(f"Total configurations evaluated: {len(all_results)}")
    print(f"Configurations meeting hardware constraints: {len(constrained_results)}")
    
    if all_results:
        # Model type analysis
        model_type_stats = {}
        for model_type in available_models:
            type_results = [r for r in all_results if r['model_type'] == model_type]
            if type_results:
                model_type_stats[model_type] = {
                    'count': len(type_results),
                    'avg_accuracy': np.mean([r['accuracy'] for r in type_results]),
                    'max_accuracy': np.max([r['accuracy'] for r in type_results]),
                    'constrained_count': len([r for r in type_results if r['fits_constraints']])
                }
        
        print(f"\n📈 Model Type Comparison:")
        for model_type, stats in model_type_stats.items():
            print(f"{model_type.upper()}:")
            print(f"  Configurations: {stats['count']}")
            print(f"  Average Accuracy: {stats['avg_accuracy']:.4f}")
            print(f"  Best Accuracy: {stats['max_accuracy']:.4f}")
            print(f"  Hardware Compliant: {stats['constrained_count']}/{stats['count']}")
            print()

    # Show top constrained models
    if constrained_results:
        constrained_df = pd.DataFrame(constrained_results)
        constrained_df = constrained_df.sort_values(by='accuracy', ascending=False).reset_index(drop=True)

        print(f"\n🏆 Top {min(10, len(constrained_df))} Hardware-Constrained Models:")
        print("-" * 120)
        
        for idx, row in constrained_df.head(10).iterrows():
            print(f"Rank {idx + 1}: {row['model_type'].upper()}")
            print(f"  Accuracy: {row['accuracy']:.4f}")
            print(f"  Parameters: {row['params']}")
            print(f"  Hardware Usage:")
            print(f"    Flash: {row['flash_MB']:.4f} MB ({row['flash_MB']/HW_CONSTRAINTS['flash_MB']*100:.1f}%)")
            print(f"    RAM: {row['ram_MB']:.4f} MB ({row['ram_MB']/HW_CONSTRAINTS['ram_MB']*100:.1f}%)")
            print(f"    Decision Ops: {row['decision_ops']:.0f} ({row['decision_ops']/HW_CONSTRAINTS['decision_ops']*100:.1f}%)")
            print()
    else:
        print("❌ No models fit within the defined hardware constraints.")
        
        # Show top models regardless of constraints
        if all_results:
            all_df = pd.DataFrame(all_results)
            all_df = all_df.sort_values(by='accuracy', ascending=False).reset_index(drop=True)
            
            print(f"\n🔍 Top 5 Models by Accuracy (regardless of constraints):")
            for idx, row in all_df.head(5).iterrows():
                print(f"Rank {idx + 1}: {row['model_type'].upper()}")
                print(f"  Accuracy: {row['accuracy']:.4f}")
                print(f"  Parameters: {row['params']}")
                print(f"  Hardware Constraints: Flash={'✓' if row['flash_MB'] <= HW_CONSTRAINTS['flash_MB'] else '✗'}, "
                      f"RAM={'✓' if row['ram_MB'] <= HW_CONSTRAINTS['ram_MB'] else '✗'}, "
                      f"Ops={'✓' if row['decision_ops'] <= HW_CONSTRAINTS['decision_ops'] else '✗'}")
                print()

if __name__ == "__main__":
    main()