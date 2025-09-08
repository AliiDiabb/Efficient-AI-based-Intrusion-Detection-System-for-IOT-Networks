import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                           confusion_matrix, classification_report, roc_auc_score)
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Reshape, Dropout, BatchNormalization, InputLayer, MaxPooling1D, Flatten, GRU,Input,Bidirectional,LSTM
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.utils import to_categorical
import warnings
import os
warnings.filterwarnings('ignore')

print("Checking GPU availability...")
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Enable memory growth for GPU to prevent allocation errors
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU(s) available: {len(gpus)}")
        print(f"GPU Details: {[gpu.name for gpu in gpus]}")
        
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")
else:
    print("No GPU found. Using CPU.")

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# ---------------------------
# FLOPS Calculation Function
# ---------------------------
def get_flops(model, batch_size=1):
    """
    Calculate FLOPs for a given model
    """
    flops = 0
    
    for layer in model.layers:
        if isinstance(layer, Dense):
            # For Dense layers: FLOPs = 2 * input_size * output_size
            input_size = layer.input_shape[-1]
            output_size = layer.units
            flops += 2 * input_size * output_size
        # Add more layer types as needed
    
    return flops * batch_size

# ---------------------------
# Hardware Measures Function
# ---------------------------
def hw_measures(model):
    n_params = model.count_params()
    max_tens = max([np.prod(layer.output_shape[1:]) for layer in model.layers if None not in layer.output_shape[1:]], default=0)
    flops = get_flops(model, batch_size=1)
    flash_size = 4 * n_params  # 4 bytes per parameter
    ram_size = 4 * max_tens    # 4 bytes per tensor
    return n_params, max_tens, flops, flash_size, ram_size

# ---------------------------------
# Fixed Edge-IIoT Shallow DNN Model
# ---------------------------------

class DNNModels:
    def __init__(self):
        pass

    def build_simple_dnn(self, input_shape=(108,), num_classes=2, classification_type='binary'):
        """
        Very shallow DNN with minimal parameters for Edge-IIoT.
        Supports binary and multi-class classification.
        """
        model = Sequential()
        model.add(InputLayer(input_shape=input_shape))
        model.add(Dense(2, activation='relu'))  # Small size for edge deployment

        if classification_type == 'binary':
            model.add(Dense(1, activation='sigmoid'))
        else:
            model.add(Dense(num_classes, activation='softmax'))
        
        return model

    def build_deep_dnn(self, input_shape=(108,), num_classes=2, classification_type='binary'):
        """
        Deep DNN with dropout regularization.
        Hidden layer architecture: 64 -> 32 neurons.
        """
        output_units = 1 if classification_type == 'binary' else num_classes
        output_activation = 'sigmoid' if output_units == 1 else 'softmax'

        model = Sequential([
            Dense(64, activation='relu', input_shape=input_shape),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.3),
            Dense(output_units, activation=output_activation)
        ])
        return model



# ---------------------------------
# Fixed Edge-IIoT Shallow CNN Model
# ---------------------------------

class CNNModels:
    def __init__(self):
        pass

    def build_CNN_1D(input_shape=(108,), num_classes=2, classification_type='binary'):
        """
        Build a lightweight 1D CNN model for Edge-IIoT with dropout regularization.
        Uses Conv1D -> MaxPooling -> Dense architecture with fewer parameters for fast training.
        """
        # Adjust output units based on classification type
        output_units = 1 if classification_type == 'binary' else num_classes

        model = Sequential([
            Reshape((input_shape[0], 1), input_shape=input_shape),
            Conv1D(32, 3, activation='relu'),
            MaxPooling1D(2),
            Dropout(0.2),
            
            Conv1D(64, 3, activation='relu'),
            MaxPooling1D(2),
            Dropout(0.3),

            Flatten(),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(output_units, activation='sigmoid' if output_units == 1 else 'softmax')
        ])
        
        return model



# ---------------------------------
# Fixed Edge-IIoT Shallow GRU Model
# ---------------------------------

class GRUModels:
    def __init__(self):
        pass

    def build_gru_uni_false(self, input_shape=(108,), num_classes=2, classification_type='binary'):
        """Unidirectional GRU (return_sequences=False)"""
        output_units = 1 if classification_type == 'binary' else num_classes
        output_activation = 'sigmoid' if output_units == 1 else 'softmax'

        inputs = Input(shape=input_shape)
        x = Reshape((input_shape[0], 1))(inputs)
        x = GRU(64, return_sequences=False, dropout=0.3)(x)
        x = Dense(32, activation='relu')(x)
        x = Dropout(0.3)(x)
        outputs = Dense(output_units, activation=output_activation)(x)

        model = Model(inputs=inputs, outputs=outputs)
        return model

    def build_gru_uni_true(self, input_shape=(108,), num_classes=2, classification_type='binary'):
        """Unidirectional GRU (return_sequences=True)"""
        output_units = 1 if classification_type == 'binary' else num_classes
        output_activation = 'sigmoid' if output_units == 1 else 'softmax'

        inputs = Input(shape=input_shape)
        x = Reshape((input_shape[0], 1))(inputs)
        x = GRU(16, return_sequences=True, dropout=0.3)(x)
        x = GRU(8, return_sequences=False, dropout=0.3)(x)
        x = Dense(8, activation='relu')(x)
        outputs = Dense(output_units, activation=output_activation)(x)

        model = Model(inputs=inputs, outputs=outputs)
        return model

    def build_gru_bi_false(self, input_shape=(108,), num_classes=2, classification_type='binary'):
        """Bidirectional GRU (return_sequences=False)"""
        output_units = 1 if classification_type == 'binary' else num_classes
        output_activation = 'sigmoid' if output_units == 1 else 'softmax'

        model = Sequential([
            Reshape((input_shape[0], 1), input_shape=input_shape),
            Bidirectional(GRU(32, return_sequences=False, dropout=0.3)),
            Dense(16, activation='relu'),
            Dropout(0.3),
            Dense(output_units, activation=output_activation)
        ])
        return model

    def build_gru_bi_true(self, input_shape=(108,), num_classes=2, classification_type='binary'):
        """Bidirectional GRU (return_sequences=True)"""
        output_units = 1 if classification_type == 'binary' else num_classes
        output_activation = 'sigmoid' if output_units == 1 else 'softmax'

        model = Sequential([
            Reshape((input_shape[0], 1), input_shape=input_shape),
            Bidirectional(GRU(16, return_sequences=True, dropout=0.3)),
            Bidirectional(GRU(8, return_sequences=False, dropout=0.3)),
            Dense(8, activation='relu'),
            Dropout(0.3),
            Dense(output_units, activation=output_activation)
        ])
        return model



# ----------------------------------
# Fixed Edge-IIoT Shallow LSTM Model
# ----------------------------------

class LSTMModels:
    def __init__(self):
        pass

    def create_lstm_uni_false(self, input_shape=(108,), num_classes=2, classification_type='binary'):
        """Model 8: Unidirectional LSTM (return_sequences=False)"""
        
        output_units = 1 if classification_type == 'binary' else num_classes
        output_activation = 'sigmoid' if output_units == 1 else 'softmax'
        
        model = Sequential([
            Reshape((input_shape[0], 1), input_shape=input_shape),
            LSTM(64, return_sequences=False, dropout=0.3),
            Dense(32, activation='relu'),
            Dropout(0.3),
            Dense(output_units, activation=output_activation)
        ])
        return model

    def create_lstm_uni_true(self, input_shape=(108,), num_classes=2, classification_type='binary'):
        """Model 9: Unidirectional LSTM (return_sequences=True)"""
        
        output_units = 1 if classification_type == 'binary' else num_classes
        output_activation = 'sigmoid' if output_units == 1 else 'softmax'
        
        model = Sequential([
            Reshape((input_shape[0], 1), input_shape=input_shape),
            LSTM(32, return_sequences=True, dropout=0.3),
            LSTM(16, return_sequences=False, dropout=0.3),
            Dense(8, activation='relu'),
            Dropout(0.3),
            Dense(output_units, activation=output_activation)
        ])
        return model

    def create_lstm_bi_false(self, input_shape=(108,), num_classes=2, classification_type='binary'):
        """Model 10: Bidirectional LSTM (return_sequences=False)"""
        
        output_units = 1 if classification_type == 'binary' else num_classes
        output_activation = 'sigmoid' if output_units == 1 else 'softmax'
        
        model = Sequential([
            Reshape((input_shape[0], 1), input_shape=input_shape),
            Bidirectional(LSTM(16, return_sequences=False, dropout=0.3)),
            Dense(8, activation='relu'),
            Dropout(0.3),
            Dense(output_units, activation=output_activation)
        ])
        return model

    def create_lstm_bi_true(self, input_shape=(108,), num_classes=2, classification_type='binary'):
        """Model 11: Bidirectional LSTM (return_sequences=True)"""
        
        output_units = 1 if classification_type == 'binary' else num_classes
        output_activation = 'sigmoid' if output_units == 1 else 'softmax'
        
        model = Sequential([
            Reshape((input_shape[0], 1), input_shape=input_shape),
            Bidirectional(LSTM(16, return_sequences=True, dropout=0.3)),
            Bidirectional(LSTM(8, return_sequences=False, dropout=0.3)),
            Dense(8, activation='relu'),
            Dropout(0.3),
            Dense(output_units, activation=output_activation)
        ])
        return model


class DNNIDSAnalyzer:
    def __init__(self, dataset_path):
        """
        Initialize the DNN IDS Analyzer
        
        Args:
            dataset_path (str): Path to the preprocessed Edge-IIoT dataset
        """
        self.dataset_path = dataset_path
        self.data = None
        self.X = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.results = {}
        
    def load_and_prepare_data(self, max_samples=2218386):
        """
        Load and prepare the dataset for analysis
        
        Args:
            max_samples (int): Maximum number of samples to use (2,219,000)
        """
        print("Loading Edge-IIoT dataset...")
        
        # Load dataset with efficient memory usage
        try:
            self.data = pd.read_csv(self.dataset_path, low_memory=False)
            print(f"Dataset loaded successfully: {self.data.shape}")
        except FileNotFoundError:
            print(f"ERROR: Dataset file not found: {self.dataset_path}")
            print("Please ensure the dataset file exists at the specified path.")
            return False
        except Exception as e:
            print(f"ERROR loading dataset: {e}")
            return False
        
        # Use maximum specified samples
        if len(self.data) > max_samples:
            self.data = self.data.sample(n=max_samples, random_state=42)
            print(f"Using {max_samples:,} samples from the dataset")
        
        # Basic data exploration
        print("\n=== Dataset Overview ===")
        print(f"Shape: {self.data.shape}")
        print(f"Features: {self.data.shape[1]}")
        print(f"Samples: {self.data.shape[0]:,}")
        print(f"Columns: {list(self.data.columns)}")
        
        return True
    
    def prepare_targets_for_classification(self, classification_type):
        """
        Prepare target variables based on classification type using your custom logic
        
        Args:
            classification_type (str): 'binary', '6class', or '15class'
        """
        print(f"\n=== Preparing targets for {classification_type} classification ===")
        
        try:
            df = self.data.copy()
            
            # YOUR CUSTOM TARGET COLUMN SELECTION LOGIC
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
            
            elif classification_type == '6class':
                if '6_Attack' in df.columns:
                    target_col = '6_Attack'
                    print("Using '6_Attack' for 6-class classification")
                else:
                    raise ValueError("6_Attack column not found for 6-class classification")
                
            elif classification_type == '15class':
                if 'Attack_type' in df.columns:
                    target_col = 'Attack_type'
                    print("Using 'Attack_type' for 15-class classification (all attack types)")
                else:
                    raise ValueError("Attack_type column not found for 15-class classification")
            
            else:
                raise ValueError("classification_type must be 'binary', '6class', or '15class'")
            
            print(f"Using '{target_col}' as target variable for {classification_type} classification")
            
            # Separate features and target using your logic
            feature_cols = [col for col in df.columns if col not in ['Attack_label', 'Attack_type','6_Attack', '6_Attack_text']]
            X = df[feature_cols]
            y = df[target_col]
            
            # Handle non-numeric features
            for col in X.columns:
                if X[col].dtype == 'object':
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
            
            print(f"Final feature matrix shape: {X.shape}")
            print(f"Target distribution: {y.value_counts()}")
            
            # Data split: 80% train, 10% validation, 10% test
            print("\n=== Splitting Data (80%-10%-10%) ===")
            
            # First split: 90% temp, 10% test
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=0.1, random_state=42, 
                stratify=y if len(y.unique()) > 1 else None
            )
            
            # Second split: 80% train, 10% validation from temp
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=0.111, random_state=42,  # 0.111 * 0.9 ≈ 0.1
                stratify=y_temp if len(y_temp.unique()) > 1 else None
            )
            
            # Feature scaling
            print("Applying feature scaling...")
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            X_test_scaled = scaler.transform(X_test)
            
            # Prepare targets based on classification type
            if classification_type == 'binary':
                # Ensure binary targets are 0/1
                if y_train.dtype == 'object' or not set(y_train.unique()).issubset({0, 1}):
                    le = LabelEncoder()
                    y_train_processed = le.fit_transform(y_train)
                    y_val_processed = le.transform(y_val)
                    y_test_processed = le.transform(y_test)
                else:
                    y_train_processed = y_train.values
                    y_val_processed = y_val.values
                    y_test_processed = y_test.values
                
                num_classes = 2
                
            else:  # 6class or 15class
                # Use label encoder for multi-class
                le = LabelEncoder()
                y_train_processed = le.fit_transform(y_train)
                y_val_processed = le.transform(y_val)
                y_test_processed = le.transform(y_test)
                
                num_classes = len(le.classes_)
                print(f"Classes: {le.classes_}")
            
            # Print final statistics
            unique, counts = np.unique(y_train_processed, return_counts=True)
            print(f"Training class distribution: {dict(zip(unique, counts))}")
            print(f"Final data shapes - Train: {X_train_scaled.shape}, Val: {X_val_scaled.shape}, Test: {X_test_scaled.shape}")
            print(f"Target shapes - Train: {y_train_processed.shape}, Val: {y_val_processed.shape}, Test: {y_test_processed.shape}")
            
            return y_train_processed, y_val_processed, y_test_processed, num_classes, X_train_scaled, X_val_scaled, X_test_scaled
            
        except Exception as e:
            print(f"Error preparing targets for {classification_type}: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None, None, None, None, None
    
    def build_dnn_model(self, num_classes, classification_type, input_dim):
        """
        Build and compile the model for Edge-IIoT
        
        Args:
            num_classes (int): Number of output classes
            classification_type (str): Type of classification for naming
            input_dim (int): Input dimension
        """
        print(f"\n=== Building Model for {classification_type} ===")
        
        try:
            # Use the LSTM model architecture
            LSTM_Model = LSTMModels()
            
            # Choose your model architecture - modify this line to select different models:
            model = LSTM_Model.create_lstm_uni_true(input_shape=(input_dim,), num_classes=num_classes, classification_type=classification_type)
            
            # Alternative models you can use:
            # model = LSTM_Model.create_lstm_uni_false(input_shape=(input_dim,), num_classes=num_classes, classification_type=classification_type)
            # model = LSTM_Model.create_lstm_bi_false(input_shape=(input_dim,), num_classes=num_classes, classification_type=classification_type)
            # model = LSTM_Model.create_lstm_bi_true(input_shape=(input_dim,), num_classes=num_classes, classification_type=classification_type)
            
            # gru_models = GRUModels()
            # model = gru_models.build_gru_uni_false(input_shape=(input_dim,), num_classes=num_classes, classification_type=classification_type)
            
            # dnn_models = DNNModels()
            # model = dnn_models.build_deep_dnn(input_shape=(input_dim,), num_classes=num_classes, classification_type=classification_type)
            
            # Compile model with appropriate loss function
            if classification_type == 'binary':
                model.compile(
                    optimizer=Adam(learning_rate=0.001),
                    loss='binary_crossentropy',
                    metrics=['accuracy']
                )
            else:
                model.compile(
                    optimizer=Adam(learning_rate=0.001),
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy']
                )
            
            print(f"Model architecture for {classification_type}:")
            model.summary()
            
            # Calculate and display hardware measures
            print(f"\n=== Hardware Measures for {classification_type} ===")
            params, max_tens, flops, flash_size, ram_size = hw_measures(model)
            print(f"Parameters       : {params}")
            print(f"Max Tensor Size  : {max_tens}")
            print(f"FLOPs            : {flops}")
            print(f"Flash Size (B)   : {flash_size}")
            print(f"RAM Size (B)     : {ram_size}")
            
            return model
            
        except Exception as e:
            print(f"Error building model for {classification_type}: {e}")
            return None
    
    def train_dnn_model(self, model, X_train, y_train, X_val, y_val, classification_type, epochs=100, batch_size=350):
        """
        Train the DNN model with proper callbacks
        """
        print(f"\n=== Training Model for {classification_type} ===")
        
        try:
            # Force CPU usage for problematic GPU memory allocation
            with tf.device('/CPU:0'):
                # Callbacks for training optimization
                early_stopping = EarlyStopping(
                    monitor='val_loss',
                    patience=3,
                    restore_best_weights=True,
                    verbose=1
                )
                
                lr_reducer = ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=7,
                    min_lr=0.00001,
                    verbose=1
                )
                
                # Print shapes for debugging
                print(f"Training data shapes:")
                print(f"  X_train: {X_train.shape}")
                print(f"  y_train: {y_train.shape}")
                print(f"  X_val: {X_val.shape}")
                print(f"  y_val: {y_val.shape}")
                
                # Train the model
                print(f"Training with {len(X_train):,} samples on CPU...")
                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=[early_stopping, lr_reducer],
                    verbose=1
                )
            
            print(f"Model training for {classification_type} completed!")
            return history
            
        except Exception as e:
            print(f"Error training model for {classification_type}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def evaluate_model(self, model, X_test, y_test, classification_type):
        """
        Evaluate model and calculate comprehensive metrics
        """
        print(f"\n=== Evaluating {classification_type} Model ===")
        
        try:
            # Make predictions
            y_pred_prob = model.predict(X_test)
            
            # Handle predictions correctly for binary vs multi-class
            if classification_type == 'binary':
                # Binary classification: sigmoid output, threshold at 0.5
                y_pred = (y_pred_prob > 0.5).astype(int).flatten()
                y_pred_prob_for_auc = y_pred_prob.flatten()
            else:
                # Multi-class: softmax output, take argmax
                y_pred = np.argmax(y_pred_prob, axis=1)
                y_pred_prob_for_auc = y_pred_prob
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'f1_macro': f1_score(y_test, y_pred, average='macro'),
                'f1_weighted': f1_score(y_test, y_pred, average='weighted'),
                'precision_macro': precision_score(y_test, y_pred, average='macro'),
                'recall_macro': recall_score(y_test, y_pred, average='macro'),
                'confusion_matrix': confusion_matrix(y_test, y_pred)
            }
            
            # ROC-AUC for binary classification
            if classification_type == 'binary':
                try:
                    metrics['roc_auc'] = roc_auc_score(y_test, y_pred_prob_for_auc)
                except Exception as e:
                    print(f"Warning: Could not calculate ROC-AUC: {e}")
            
            self.results[classification_type] = metrics
            
            # Display results
            print(f"\n{classification_type.upper()} CLASSIFICATION RESULTS:")
            print(f"Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
            print(f"F1-Score (Macro): {metrics['f1_macro']:.4f}")
            print(f"F1-Score (Weighted): {metrics['f1_weighted']:.4f}")
            print(f"Precision (Macro): {metrics['precision_macro']:.4f}")
            print(f"Recall (Macro): {metrics['recall_macro']:.4f}")
            
            if 'roc_auc' in metrics:
                print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
            
            return metrics
            
        except Exception as e:
            print(f"Error evaluating model for {classification_type}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def run_complete_analysis(self):
        """
        Run complete analysis for all three classification types
        """
        print("="*80)
        print("DEEP NEURAL NETWORK INTRUSION DETECTION SYSTEM")
        print("EDGE-IIOT OPTIMIZED MULTI-CLASSIFICATION ANALYSIS")
        print("="*80)
        
        if not self.load_and_prepare_data():
            print("Failed to load and prepare data. Exiting.")
            return False
        
        classification_types = ['binary', '6class', '15class']
        
        for class_type in classification_types:
            print(f"\n{'='*60}")
            print(f"RUNNING {class_type.upper()} CLASSIFICATION")
            print(f"{'='*60}")
            
            try:
                # Prepare targets using your custom logic
                result = self.prepare_targets_for_classification(class_type)
                if result[0] is None:
                    print(f"Failed to prepare targets for {class_type}. Skipping.")
                    continue
                
                y_train, y_val, y_test, num_classes, X_train_scaled, X_val_scaled, X_test_scaled = result
                
                # Build model
                model = self.build_dnn_model(num_classes, class_type, X_train_scaled.shape[1])
                
                if model is None:
                    print(f"Failed to build model for {class_type}. Skipping.")
                    continue
                
                # Train model
                history = self.train_dnn_model(model, X_train_scaled, y_train, X_val_scaled, y_val, class_type, epochs=5)
                
                if history is None:
                    print(f"Failed to train model for {class_type}. Skipping.")
                    continue
                
                # Evaluate model
                metrics = self.evaluate_model(model, X_test_scaled, y_test, class_type)
                
                if metrics is None:
                    print(f"Failed to evaluate model for {class_type}. Skipping.")
                    continue
                
                # Store model and history for visualization
                setattr(self, f'model_{class_type}', model)
                setattr(self, f'history_{class_type}', history)
                
            except Exception as e:
                print(f"Error processing {class_type} classification: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        return len(self.results) > 0
    
    def visualize_results(self, save_path="./visualizations/"):
        """
        Create comprehensive visualizations for all classification types and save them
        """
        print(f"\n=== Generating and Saving Visualizations ===")
        
        # Check if we have results to visualize
        if not self.results:
            print("No results available for visualization. Please run the analysis first.")
            return
        
        # Create directory if it doesn't exist
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            print(f"Created directory: {save_path}")
        
        # Set up the plotting
        plt.style.use('default')
        
        # Get available classification types
        available_types = list(self.results.keys())
        colors = ['skyblue', 'lightgreen', 'lightcoral'][:len(available_types)]
        
        try:
            # 1. Model Performance Comparison
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Accuracy comparison
            accuracies = [self.results[ct]['accuracy'] for ct in available_types]
            bars1 = axes[0, 0].bar(available_types, accuracies, color=colors, alpha=0.7)
            axes[0, 0].set_title('Accuracy Comparison Across Classification Types')
            axes[0, 0].set_ylabel('Accuracy')
            axes[0, 0].set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, acc in zip(bars1, accuracies):
                axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               f'{acc:.3f}', ha='center', va='bottom')
            
            # F1-Score comparison
            f1_scores = [self.results[ct]['f1_weighted'] for ct in available_types]
            bars2 = axes[0, 1].bar(available_types, f1_scores, color=colors, alpha=0.7)
            axes[0, 1].set_title('F1-Score (Weighted) Comparison')
            axes[0, 1].set_ylabel('F1-Score')
            axes[0, 1].set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, f1 in zip(bars2, f1_scores):
                axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               f'{f1:.3f}', ha='center', va='bottom')
            
            # Precision vs Recall
            precisions = [self.results[ct]['precision_macro'] for ct in available_types]
            recalls = [self.results[ct]['recall_macro'] for ct in available_types]
            
            scatter = axes[1, 0].scatter(precisions, recalls, s=150, c=colors, alpha=0.7)
            for i, ct in enumerate(available_types):
                axes[1, 0].annotate(ct, (precisions[i], recalls[i]), 
                                  xytext=(5, 5), textcoords='offset points', fontsize=10)
            axes[1, 0].set_xlabel('Precision (Macro)')
            axes[1, 0].set_ylabel('Recall (Macro)')
            axes[1, 0].set_title('Precision vs Recall')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Training history comparison
            axes[1, 1].set_title('Training History Comparison')
            for i, ct in enumerate(available_types):
                if hasattr(self, f'history_{ct}'):
                    history = getattr(self, f'history_{ct}').history
                    epochs = range(1, len(history['val_accuracy']) + 1)
                    axes[1, 1].plot(epochs, history['val_accuracy'], 
                                   label=f'{ct} Validation Accuracy', color=colors[i])
            
            axes[1, 1].set_xlabel('Epochs')
            axes[1, 1].set_ylabel('Validation Accuracy')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save the performance comparison plot
            performance_plot_path = os.path.join(save_path, 'dnn_performance_comparison.png')
            plt.savefig(performance_plot_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {performance_plot_path}")
            plt.show()
            
            # 2. Confusion Matrix Visualization
            fig, axes = plt.subplots(1, len(available_types), figsize=(5*len(available_types), 4))
            if len(available_types) == 1:
                axes = [axes]
            
            for i, ct in enumerate(available_types):
                cm = self.results[ct]['confusion_matrix']
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                           ax=axes[i], cbar=True)
                axes[i].set_title(f'{ct.upper()} Confusion Matrix')
                axes[i].set_xlabel('Predicted')
                axes[i].set_ylabel('Actual')
            
            plt.tight_layout()
            confusion_plot_path = os.path.join(save_path, 'confusion_matrices.png')
            plt.savefig(confusion_plot_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {confusion_plot_path}")
            plt.show()
            
            # 3. Metrics Comparison Radar Chart
            if len(available_types) > 1:
                from math import pi
                
                fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
                
                # Metrics to compare
                metrics_to_plot = ['accuracy', 'f1_macro', 'f1_weighted', 'precision_macro', 'recall_macro']
                
                # Number of variables
                N = len(metrics_to_plot)
                
                # Compute angle for each axis
                angles = [n / float(N) * 2 * pi for n in range(N)]
                angles += angles[:1]  # Complete the circle
                
                for i, ct in enumerate(available_types):
                    values = [self.results[ct][metric] for metric in metrics_to_plot]
                    values += values[:1]  # Complete the circle
                    
                    ax.plot(angles, values, 'o-', linewidth=2, label=ct, color=colors[i])
                    ax.fill(angles, values, alpha=0.25, color=colors[i])
                
                # Add labels
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels([metric.replace('_', ' ').title() for metric in metrics_to_plot])
                ax.set_ylim(0, 1)
                ax.set_title('Performance Metrics Comparison', size=16, fontweight='bold', pad=20)
                ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
                ax.grid(True)
                
                plt.tight_layout()
                radar_plot_path = os.path.join(save_path, 'metrics_radar_chart.png')
                plt.savefig(radar_plot_path, dpi=300, bbox_inches='tight')
                print(f"Saved: {radar_plot_path}")
                plt.show()
            
            print(f"\nAll visualizations saved to: {save_path}")
            
        except Exception as e:
            print(f"Error creating visualizations: {e}")
            import traceback
            traceback.print_exc()
    
    def generate_comprehensive_report(self):
        """
        Generate comprehensive analysis report for all classification types
        """
        print(f"\n{'='*80}")
        print("COMPREHENSIVE ANALYSIS REPORT")
        print("DEEP NEURAL NETWORK INTRUSION DETECTION SYSTEM")
        print("EDGE-IIOT OPTIMIZED WITH CUSTOM TARGET SELECTION")
        print(f"{'='*80}")
        
        if not hasattr(self, 'data') or self.data is None:
            print("No data available for report generation. Please run the analysis first.")
            return
        
        print(f"\n1. DATASET OVERVIEW")
        print(f"   - Total Samples: {len(self.data):,}")
        print(f"   - Features: {self.data.shape[1]}")
        print(f"   - Available Columns: {list(self.data.columns)}")
        
        # Check available target columns
        target_columns = []
        if 'Attack_label' in self.data.columns:
            target_columns.append('Attack_label (binary)')
        if 'Attack_type' in self.data.columns:
            target_columns.append('Attack_type (15-class)')
        if '6_Attack' in self.data.columns:
            target_columns.append('6_Attack (6-class)')
        
        print(f"   - Available Target Columns: {target_columns}")
        
        if not self.results:
            print(f"\n2. RESULTS")
            print("   No results available. Analysis may have failed.")
            return
            
        print(f"\n2. PERFORMANCE SUMMARY")
        print(f"   {'Classification Type':<20} {'Accuracy':<12} {'F1-Macro':<12} {'F1-Weighted':<12} {'Precision':<12} {'Recall':<12}")
        print(f"   {'-'*80}")
        
        for ct in self.results:
            acc = self.results[ct]['accuracy']
            f1_macro = self.results[ct]['f1_macro']
            f1_weighted = self.results[ct]['f1_weighted']
            precision = self.results[ct]['precision_macro']
            recall = self.results[ct]['recall_macro']
            print(f"   {ct.upper():<20} {acc:.4f}      {f1_macro:.4f}      {f1_weighted:.4f}       {precision:.4f}      {recall:.4f}")
        
        print(f"\n3. MODEL CONFIGURATIONS")
        for ct in self.results:
            if hasattr(self, f'model_{ct}'):
                model = getattr(self, f'model_{ct}')
                params, max_tens, flops, flash_size, ram_size = hw_measures(model)
                print(f"   {ct.upper()} Model:")
                print(f"     - Parameters: {params:,}")
                print(f"     - FLOPs: {flops:,}")
                print(f"     - Flash Size: {flash_size:,} bytes ({flash_size/1024:.1f} KB)")
                print(f"     - RAM Size: {ram_size:,} bytes ({ram_size/1024:.1f} KB)")
        
        print(f"\n4. TARGET COLUMN USAGE")
        print(f"   - Binary Classification: Attack_label (Normal=0, Attack=1)")
        print(f"   - 6-Class Classification: 6_Attack column")
        print(f"   - 15-Class Classification: Attack_type column")
        
        print(f"\n5. TRAINING CONFIGURATION")
        print(f"   - Train/Validation/Test Split: 80%/10%/10%")
        print(f"   - Feature Scaling: StandardScaler")
        print(f"   - Model Architecture: LSTM-based with dropout regularization")
        print(f"   - Early Stopping: Patience=3, monitor=val_loss")
        print(f"   - Learning Rate Reduction: Factor=0.5, patience=7")


# Example usage and main execution
if __name__ == "__main__":
    # Initialize the analyzer
    # Replace with your actual dataset path
    dataset_path = "Preprocessed DataSet/Preprocessed-DNN-EdgeIIoT-dataset.csv"  # Update this path
    
    analyzer = DNNIDSAnalyzer(dataset_path)
    
    # Run complete analysis
    success = analyzer.run_complete_analysis()
    
    if success:
        # Generate visualizations only if analysis was successful
        analyzer.visualize_results()
        
        # Generate comprehensive report
        analyzer.generate_comprehensive_report()
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETED SUCCESSFULLY!")
        print("Check the './visualizations/' folder for saved plots.")
        print("="*80)
    else:
        print("\n" + "="*80)
        print("ANALYSIS FAILED!")
        print("Please check your dataset path and format.")
        print("="*80)