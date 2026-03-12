"""
Antibacterial Peptide Prediction and Generation Script (Enhanced Dialog Version) - Multi-Model Selection and Optimization Edition
Function: Read Excel data, train/save/load models, perform sequence prediction and generation.
Dependencies: pandas, scikit-learn, xgboost, lightgbm, openpyxl, numpy, joblib, os, glob
Please ensure a file named 'data.xlsx' exists in the same directory.
"""

import pandas as pd
import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from collections import defaultdict
import random
import joblib

# XGBoost and LightGBM imports
try:
    from xgboost import XGBClassifier

    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("Warning: xgboost library not found. You will not be able to select the XGBoost model.")

try:
    from lightgbm import LGBMClassifier

    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False
    print("Warning: lightgbm library not found. You will not be able to select the LightGBM model.")

# Global variables to store the current model and related information
current_model = None
current_kmer_vocab = None
current_train_df = None
current_val_df = None
current_full_df = None
current_model_eval = None
# Added: Global variable to store training data (list of antibacterial peptide sequences) for generation
current_amp_seqs_for_generation = []
# Added: Store the name of the currently loaded model
current_model_name = None
saved_models_dir = "./saved_models/"

# Create directory for saving models
if not os.path.exists(saved_models_dir):
    os.makedirs(saved_models_dir)


# ==================== 1. Data Loading and Preprocessing ====================
def load_and_prepare_data(filepath='data.xlsx', test_size=0.3, random_state=42):
    """
    Load Excel file, integrate antibacterial peptide and non-antibacterial peptide data, and add labels.
    Assumed file structure: first column ID, second column antibacterial peptide sequence, third column non-antibacterial peptide sequence.
    """
    try:
        df = pd.read_excel(filepath, engine='openpyxl', header=None)
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found in the same directory.")
        return None, None, None, None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None, None, None, None

    # Column structure: 0: ID, 1: Antibacterial peptide sequence, 2: Non-antibacterial peptide sequence
    amp_data = []  # Antibacterial peptides
    non_amp_data = []  # Non-antibacterial peptides

    for _, row in df.iterrows():
        amp_seq = str(row[1]).upper().replace(' ', '')  # Antibacterial peptide sequence
        non_amp_seq = str(row[2]).upper().replace(' ', '')  # Non-antibacterial peptide sequence

        # Keep only valid amino acid sequences
        valid_aas = set('ACDEFGHIKLMNPQRSTVWY')
        if all(aa in valid_aas for aa in amp_seq) and len(amp_seq) > 0:
            amp_data.append({'sequence': amp_seq, 'label': 1})
        if all(aa in valid_aas for aa in non_amp_seq) and len(non_amp_seq) > 0:
            non_amp_data.append({'sequence': non_amp_seq, 'label': 0})

    # Combine data
    amp_df = pd.DataFrame(amp_data)
    non_amp_df = pd.DataFrame(non_amp_data)
    full_df = pd.concat([amp_df, non_amp_df], ignore_index=True)

    if full_df.empty:
        print("Error: No valid sequence data extracted from the file.")
        return None, None, None, None

    print(f"Data loaded successfully. Total sequences: {len(full_df)}.")
    print(f"  Number of antibacterial peptides: {len(amp_df)}")
    print(f"  Number of non-antibacterial peptides: {len(non_amp_df)}")

    # Split into training and validation sets according to the specified ratio
    train_df, val_df = train_test_split(full_df, test_size=test_size, random_state=random_state,
                                        stratify=full_df['label'])
    print(f"  Training set size: {len(train_df)} ({(1 - test_size) * 100:.0f}%)")
    print(f"  Validation set size: {len(val_df)} ({test_size * 100:.0f}%)")

    # Save datasets
    train_df.to_csv('train_dataset.csv', index=False)
    val_df.to_csv('validation_dataset.csv', index=False)
    print("Training and validation sets have been saved as 'train_dataset.csv' and 'validation_dataset.csv'.")

    return train_df, val_df, full_df


# ==================== 2. Feature Engineering ====================
def sequence_to_features(sequences, k=3):
    """Convert amino acid sequences to feature dictionaries"""
    amino_acids = list('ACDEFGHIKLMNPQRSTVWY')
    all_features = []

    for seq in sequences:
        features = []
        # Amino acid composition
        aa_composition = [seq.count(aa) / len(seq) for aa in amino_acids]
        features.extend(aa_composition)

        # k-mer frequency
        kmer_counts = defaultdict(int)
        for i in range(len(seq) - k + 1):
            kmer = seq[i:i + k]
            kmer_counts[kmer] += 1

        all_features.append({'seq': seq, 'aa_comp': aa_composition,
                             'kmer_counts': kmer_counts})
    return all_features


def create_feature_vectors(feature_dicts, kmer_vocab=None):
    """Convert feature dictionaries to a numerical matrix"""
    amino_acids = list('ACDEFGHIKLMNPQRSTVWY')
    feature_vectors = []

    if kmer_vocab is None:
        all_kmers = set()
        for fd in feature_dicts:
            all_kmers.update(fd['kmer_counts'].keys())
        kmer_vocab = {kmer: idx for idx, kmer in enumerate(sorted(all_kmers))}

    for fd in feature_dicts:
        vec = []
        vec.extend(fd['aa_comp'])
        kmer_vec = [0] * len(kmer_vocab)
        for kmer, count in fd['kmer_counts'].items():
            if kmer in kmer_vocab:
                kmer_vec[kmer_vocab[kmer]] = count / len(fd['seq'])
        vec.extend(kmer_vec)
        feature_vectors.append(vec)

    return np.array(feature_vectors), kmer_vocab


# ==================== 3. Model Training and Evaluation (Includes model selection and parameter setting) ====================
def train_and_evaluate(train_df, val_df, model_type='random_forest', model_params=None):
    """
    Train the selected model and evaluate, return the model, vocabulary, and evaluation results.
    model_type: Optional 'random_forest' (default), 'svm', 'xgboost', or 'lightgbm'
    model_params: Model parameter dictionary, uses default if None
    """
    print(f"\n--- Starting Feature Engineering and Model Training (Model Type: {model_type}) ---")

    # Display used parameters
    if model_params:
        print(f"Model parameters used: {model_params}")

    train_seqs = train_df['sequence'].tolist()
    train_labels = train_df['label'].tolist()
    val_seqs = val_df['sequence'].tolist()
    val_labels = val_df['label'].tolist()

    print("Generating features for the training set...")
    train_feature_dicts = sequence_to_features(train_seqs, k=3)
    X_train, kmer_vocab = create_feature_vectors(train_feature_dicts)
    y_train = np.array(train_labels)

    print("Generating features for the validation set...")
    val_feature_dicts = sequence_to_features(val_seqs, k=3)
    X_val, _ = create_feature_vectors(val_feature_dicts, kmer_vocab)
    y_val = np.array(val_labels)

    print(f"Feature matrix dimensions -> Training set: {X_train.shape}, Validation set: {X_val.shape}")

    # --- Initialize the model based on user selection ---
    if model_type == 'random_forest':
        print("Training Random Forest model...")
        # Set default parameters
        default_params = {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': 42,
            'n_jobs': -1
        }
        # Update default parameters if custom parameters are provided
        if model_params:
            default_params.update(model_params)
        model = RandomForestClassifier(**default_params)

    elif model_type == 'svm':
        print("Training Support Vector Machine (SVM) model...")
        # Set default parameters
        default_params = {
            'C': 1.0,
            'max_iter': 10000,
            'random_state': 42
        }
        # Update default parameters if custom parameters are provided
        if model_params:
            default_params.update(model_params)
        # Use LinearSVC for better performance, and CalibratedClassifierCV for calibration to get probabilities
        base_svm = LinearSVC(**default_params)
        model = CalibratedClassifierCV(base_svm, cv=3)

    elif model_type == 'xgboost':
        if not XGB_AVAILABLE:
            print("Error: XGBoost library not installed. Please run 'pip install xgboost' to install.")
            return None, None, None
        print("Training XGBoost model...")
        # Set default parameters
        default_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'random_state': 42,
            'n_jobs': -1,
            'eval_metric': 'logloss'
        }
        # Update default parameters if custom parameters are provided
        if model_params:
            default_params.update(model_params)
        model = XGBClassifier(**default_params)

    elif model_type == 'lightgbm':
        if not LGBM_AVAILABLE:
            print("Error: LightGBM library not installed. Please run 'pip install lightgbm' to install.")
            return None, None, None
        print("Training LightGBM model...")
        # Set default parameters
        default_params = {
            'n_estimators': 100,
            'max_depth': -1,
            'learning_rate': 0.1,
            'num_leaves': 31,
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': -1
        }
        # Update default parameters if custom parameters are provided
        if model_params:
            default_params.update(model_params)
        model = LGBMClassifier(**default_params)

    else:
        print(f"Error: Unknown model type '{model_type}', will use default Random Forest.")
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

    # Train and evaluate
    model.fit(X_train, y_train)
    print("Evaluating model...")
    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)[:, 1]

    acc = accuracy_score(y_val, y_pred)
    # Generate serializable evaluation results
    eval_result = {
        'accuracy': acc,
        'classification_report': classification_report(y_val, y_pred, target_names=['Non-antibacterial peptide', 'Antibacterial peptide'], output_dict=True),
        'confusion_matrix': confusion_matrix(y_val, y_pred).tolist(),
        'model_type': model_type,
        'model_params': model_params if model_params else default_params
    }
    print(f"\nModel validation accuracy: {acc:.4f}")
    print("\nClassification report:")
    print(classification_report(y_val, y_pred, target_names=['Non-antibacterial peptide', 'Antibacterial peptide']))
    print("Confusion matrix:")
    print(confusion_matrix(y_val, y_pred))

    return model, kmer_vocab, eval_result


# ==================== 4. Sequence Prediction Function ====================
def predict_sequence(model, kmer_vocab, sequence):
    """Predict the probability of a single sequence being an antibacterial peptide"""
    seq_clean = str(sequence).upper().replace(' ', '')
    valid_aas = set('ACDEFGHIKLMNPQRSTVWY')

    if not all(aa in valid_aas for aa in seq_clean):
        print("Warning: Sequence contains non-standard amino acid characters, automatically filtered.")
        seq_clean = ''.join([aa for aa in seq_clean if aa in valid_aas])

    if len(seq_clean) < 3:
        print("Error: Sequence is too short to extract features.")
        return None

    feature_dicts = sequence_to_features([seq_clean], k=3)
    X_seq, _ = create_feature_vectors(feature_dicts, kmer_vocab)

    proba = model.predict_proba(X_seq)[0, 1]
    return proba


# ==================== 5. Antibacterial Peptide Generation Function ====================
def generate_antimicrobial_peptide(length, train_amp_seqs, order=2, max_attempts=100,
                                   target_prob=0.8, model=None, kmer_vocab=None):
    """
    Generate an antibacterial peptide of the specified length, ensuring a high probability of it being an antibacterial peptide.
    """
    if not train_amp_seqs:
        print("Error: No training antibacterial peptide sequences available for generation.")
        return None, 0.0
    if length < order + 1:
        print(f"Error: Generation length requires at least {order + 1} amino acids.")
        return None, 0.0

    # Build Markov model
    markov_model = defaultdict(lambda: defaultdict(int))
    for seq in train_amp_seqs:
        for i in range(len(seq) - order):
            prefix = seq[i:i + order]
            next_aa = seq[i + order]
            markov_model[prefix][next_aa] += 1

    if not markov_model:
        print("Error: Cannot build a valid Markov model from training sequences.")
        return None, 0.0

    # Convert to probabilities
    for prefix, counts in markov_model.items():
        total = sum(counts.values())
        for aa in counts:
            counts[aa] /= total

    # Generate sequence and verify probability
    best_peptide = None
    best_prob = 0.0
    prefixes = list(markov_model.keys())

    for attempt in range(max_attempts):
        # Randomly select a starting prefix
        start_prefix = random.choice(prefixes)
        generated = list(start_prefix)

        # Generate sequence
        while len(generated) < length:
            current_prefix = ''.join(generated[-order:])
            if current_prefix not in markov_model:
                next_aa = random.choice(list('ACDEFGHIKLMNPQRSTVWY'))
            else:
                choices, weights = zip(*markov_model[current_prefix].items())
                next_aa = random.choices(choices, weights=weights, k=1)[0]
            generated.append(next_aa)

        peptide = ''.join(generated[:length])

        # Calculate probability if a model is available
        if model is not None and kmer_vocab is not None:
            proba = predict_sequence(model, kmer_vocab, peptide)
            if proba is not None and proba > best_prob:
                best_prob = proba
                best_peptide = peptide

                # Return early if the target probability is reached
                if best_prob >= target_prob:
                    print(f"Attempt {attempt + 1}: Reached target probability {target_prob:.2f}")
                    return best_peptide, best_prob
        else:
            # If no model, return the first generated sequence
            return peptide, 0.0

    print(f"After {max_attempts} attempts, the highest probability is {best_prob:.4f}")
    return best_peptide, best_prob


# ==================== 6. Model Optimization ====================
def model_optimization_interactive():
    """Model Optimization Function - Allows users to customize model parameters and dataset split ratio"""
    global current_model, current_kmer_vocab, current_train_df, current_val_df, current_full_df, current_model_eval, current_amp_seqs_for_generation, current_model_name

    print("\n" + "=" * 60)
    print("Model Optimization Function")
    print("=" * 60)
    print("In this function, you can customize model parameters and dataset split ratio.")

    # --- 1. Select model type ---
    print("\nPlease select the model type to train:")
    print("  1. Random Forest")
    print("  2. Support Vector Machine (SVM)")
    if XGB_AVAILABLE:
        print("  3. XGBoost")
    else:
        print("  3. XGBoost (Unavailable, library not installed)")
    if LGBM_AVAILABLE:
        print("  4. LightGBM")
    else:
        print("  4. LightGBM (Unavailable, library not installed)")

    model_choice = input("Please enter an option (1-4, default is 1, enter 'q' to return to main menu): ").strip()
    if model_choice.lower() == 'q':
        print("Returning to main menu.")
        return
    model_type = 'random_forest'  # Default

    if model_choice == '2':
        model_type = 'svm'
    elif model_choice == '3' and XGB_AVAILABLE:
        model_type = 'xgboost'
    elif model_choice == '3' and not XGB_AVAILABLE:
        print("XGBoost library not installed, will use default Random Forest.")
        print("To use XGBoost, please run 'pip install xgboost' to install.")
    elif model_choice == '4' and LGBM_AVAILABLE:
        model_type = 'lightgbm'
    elif model_choice == '4' and not LGBM_AVAILABLE:
        print("LightGBM library not installed, will use default Random Forest.")
        print("To use LightGBM, please run 'pip install lightgbm' to install.")
    else:
        model_type = 'random_forest'

    print(f"\nYou have selected the {model_type} model.")

    # --- 2. Set dataset split ratio ---
    print("\n--- Dataset Split Ratio Setting ---")
    print("Current default split ratio: Training set 70%, Validation set 30%")

    while True:
        test_size_input = input("Please enter the validation set ratio (0.1-0.5, default is 0.3, enter 'q' to return to main menu): ").strip()
        if test_size_input.lower() == 'q':
            print("Returning to main menu.")
            return
        if not test_size_input:
            test_size = 0.3
            break
        try:
            test_size = float(test_size_input)
            if 0.1 <= test_size <= 0.5:
                break
            else:
                print("Validation set ratio must be between 0.1 and 0.5.")
        except ValueError:
            print("Please enter a valid number.")

    print(f"Dataset split ratio set to: Training set {(1 - test_size) * 100:.0f}%, Validation set {test_size * 100:.0f}%")

    # --- 3. Set random seed (optional) ---
    print("\n--- Random Seed Setting ---")
    random_seed_input = input("Please enter a random seed (integer, default is 42, enter -1 for no random seed, enter 'q' to return to main menu): ").strip()
    if random_seed_input.lower() == 'q':
        print("Returning to main menu.")
        return
    if not random_seed_input:
        random_state = 42
    elif random_seed_input == '-1':
        random_state = None
    else:
        try:
            random_state = int(random_seed_input)
        except ValueError:
            print("Invalid input, will use default value 42.")
            random_state = 42

    # --- 4. Set model parameters ---
    model_params = {}
    print(f"\n--- {model_type} Model Parameter Setting ---")
    print("You can set custom values for the following parameters. Press Enter to use default values. Enter 'q' at any time to return to the main menu.")

    if model_type == 'random_forest':
        # Random Forest parameters
        n_estimators = input(f"n_estimators (number of trees, default 100, enter 'q' to return to main menu): ").strip()
        if n_estimators.lower() == 'q':
            print("Returning to main menu.")
            return
        if n_estimators:
            try:
                model_params['n_estimators'] = int(n_estimators)
            except ValueError:
                print("Invalid input, will use default value.")

        max_depth = input(f"max_depth (maximum tree depth, default None for unlimited, enter 'q' to return to main menu): ").strip()
        if max_depth.lower() == 'q':
            print("Returning to main menu.")
            return
        if max_depth:
            if max_depth.lower() == 'none':
                model_params['max_depth'] = None
            else:
                try:
                    model_params['max_depth'] = int(max_depth)
                except ValueError:
                    print("Invalid input, will use default value.")

        min_samples_split = input(f"min_samples_split (minimum samples required to split an internal node, default 2, enter 'q' to return to main menu): ").strip()
        if min_samples_split.lower() == 'q':
            print("Returning to main menu.")
            return
        if min_samples_split:
            try:
                model_params['min_samples_split'] = int(min_samples_split)
            except ValueError:
                print("Invalid input, will use default value.")

        min_samples_leaf = input(f"min_samples_leaf (minimum number of samples required at a leaf node, default 1, enter 'q' to return to main menu): ").strip()
        if min_samples_leaf.lower() == 'q':
            print("Returning to main menu.")
            return
        if min_samples_leaf:
            try:
                model_params['min_samples_leaf'] = int(min_samples_leaf)
            except ValueError:
                print("Invalid input, will use default value.")

    elif model_type == 'svm':
        # SVM parameters
        C_value = input(f"C (regularization parameter, default 1.0, enter 'q' to return to main menu): ").strip()
        if C_value.lower() == 'q':
            print("Returning to main menu.")
            return
        if C_value:
            try:
                model_params['C'] = float(C_value)
            except ValueError:
                print("Invalid input, will use default value.")

        max_iter = input(f"max_iter (maximum number of iterations, default 10000, enter 'q' to return to main menu): ").strip()
        if max_iter.lower() == 'q':
            print("Returning to main menu.")
            return
        if max_iter:
            try:
                model_params['max_iter'] = int(max_iter)
            except ValueError:
                print("Invalid input, will use default value.")

    elif model_type == 'xgboost':
        # XGBoost parameters
        n_estimators = input(f"n_estimators (number of trees, default 100, enter 'q' to return to main menu): ").strip()
        if n_estimators.lower() == 'q':
            print("Returning to main menu.")
            return
        if n_estimators:
            try:
                model_params['n_estimators'] = int(n_estimators)
            except ValueError:
                print("Invalid input, will use default value.")

        max_depth = input(f"max_depth (maximum tree depth, default 6, enter 'q' to return to main menu): ").strip()
        if max_depth.lower() == 'q':
            print("Returning to main menu.")
            return
        if max_depth:
            try:
                model_params['max_depth'] = int(max_depth)
            except ValueError:
                print("Invalid input, will use default value.")

        learning_rate = input(f"learning_rate (learning rate, default 0.1, enter 'q' to return to main menu): ").strip()
        if learning_rate.lower() == 'q':
            print("Returning to main menu.")
            return
        if learning_rate:
            try:
                model_params['learning_rate'] = float(learning_rate)
            except ValueError:
                print("Invalid input, will use default value.")

    elif model_type == 'lightgbm':
        # LightGBM parameters
        n_estimators = input(f"n_estimators (number of trees, default 100, enter 'q' to return to main menu): ").strip()
        if n_estimators.lower() == 'q':
            print("Returning to main menu.")
            return
        if n_estimators:
            try:
                model_params['n_estimators'] = int(n_estimators)
            except ValueError:
                print("Invalid input, will use default value.")

        max_depth = input(f"max_depth (maximum tree depth, default -1 for unlimited, enter 'q' to return to main menu): ").strip()
        if max_depth.lower() == 'q':
            print("Returning to main menu.")
            return
        if max_depth:
            try:
                model_params['max_depth'] = int(max_depth)
            except ValueError:
                print("Invalid input, will use default value.")

        learning_rate = input(f"learning_rate (learning rate, default 0.1, enter 'q' to return to main menu): ").strip()
        if learning_rate.lower() == 'q':
            print("Returning to main menu.")
            return
        if learning_rate:
            try:
                model_params['learning_rate'] = float(learning_rate)
            except ValueError:
                print("Invalid input, will use default value.")

        num_leaves = input(f"num_leaves (number of leaves, default 31, enter 'q' to return to main menu): ").strip()
        if num_leaves.lower() == 'q':
            print("Returning to main menu.")
            return
        if num_leaves:
            try:
                model_params['num_leaves'] = int(num_leaves)
            except ValueError:
                print("Invalid input, will use default value.")

    # Display user-set parameters
    if model_params:
        print(f"\nYour custom parameters: {model_params}")
    else:
        print("\nNo custom parameters set, will use default parameters.")

    # --- 5. Confirm and start training ---
    confirm = input("\nProceed with training the model using the above settings? (y/n, enter 'q' to return to main menu): ").strip().lower()
    if confirm == 'q':
        print("Returning to main menu.")
        return
    if confirm != 'y':
        print("Training cancelled, returning to main menu.")
        return

    # Reset model name
    current_model_name = None

    # Load data
    print("\nLoading data and training model...")
    current_train_df, current_val_df, current_full_df = load_and_prepare_data(
        'data.xlsx',
        test_size=test_size,
        random_state=random_state
    )

    if current_train_df is not None:
        # Train model
        current_model, current_kmer_vocab, current_model_eval = train_and_evaluate(
            current_train_df,
            current_val_df,
            model_type=model_type,
            model_params=model_params
        )

        if current_model is not None:
            print("✅ Model training completed!")
            # Update data for generation
            current_amp_seqs_for_generation = current_train_df[current_train_df['label'] == 1][
                'sequence'].tolist()
            print(f"Generation data updated, containing {len(current_amp_seqs_for_generation)} antibacterial peptide sequences.")

            # Ask whether to save after training
            save_now = input("\nSave the model immediately? (y/n, enter 'q' to return to main menu): ").strip().lower()
            if save_now == 'q':
                print("Returning to main menu.")
                return
            if save_now == 'y':
                if current_train_df is not None:
                    train_amp_seqs = current_train_df[current_train_df['label'] == 1]['sequence'].tolist()
                    # Ask for model name
                    while True:
                        model_name = input("Please enter the model name to save (without extension, enter 'q' to return to main menu): ").strip()
                        if model_name.lower() == 'q':
                            print("Returning to main menu.")
                            return
                        if not model_name:
                            print("Model name cannot be empty, please re-enter.")
                            continue
                        # Call save_model with model type
                        if save_model(current_model, current_kmer_vocab, train_amp_seqs, current_model_eval,
                                      model_name, model_type=model_type):
                            current_model_name = model_name
                            break
                        else:
                            print("Save failed, please try again.")
                else:
                    print("Error: Training data unavailable, cannot save.")
            else:
                print("Model not saved. You can view the current model in the main menu or load it later.")


# ==================== 7. Model Management ====================
def save_model(model, kmer_vocab, train_amp_seqs, eval_result=None, model_name=None, model_type='random_forest'):
    """Save the model in .joblib format, can specify a name or interactively input"""
    print("\n" + "=" * 50)
    print("Save Model")
    print("=" * 50)

    if model is None:
        print("Error: No model to save, please train a model first.")
        return False

    # If no model name provided, ask interactively
    if model_name is None:
        while True:
            model_name = input("Please enter the model name (without extension, enter 'q' to return to main menu): ").strip()

            if model_name.lower() == 'q':
                print("Save cancelled, returning to main menu.")
                return False

            if not model_name:
                print("Model name cannot be empty, please re-enter.")
                continue
            break
    else:
        # If model name provided, use it directly
        pass

    filepath = os.path.join(saved_models_dir, f"{model_name}.joblib")

    if os.path.exists(filepath):
        overwrite = input(f"Model '{model_name}' already exists, overwrite? (y/n, enter 'q' to return to main menu): ").lower()
        if overwrite == 'q':
            print("Save cancelled, returning to main menu.")
            return False
        if overwrite != 'y':
            print("Save cancelled.")
            return False

    # Save model and related data
    to_save = {
        'model': model,
        'kmer_vocab': kmer_vocab,
        'train_amp_seqs': train_amp_seqs,  # Key: save data for generation
        'eval_result': eval_result,  # Save evaluation results
        'metadata': {
            'save_time': pd.Timestamp.now(),
            'feature_type': 'kmer+aa_composition',
            'model_type': model_type
        }
    }

    try:
        joblib.dump(to_save, filepath)
        print(f"Model saved as: {filepath}")
        print(f"Model type: {model_type}")
        print(f"Also saved {len(train_amp_seqs)} antibacterial peptide sequences for generation.")
        return True
    except Exception as e:
        print(f"Error saving model: {e}")
        return False


def list_saved_models():
    """List all saved models and display evaluation results"""
    print("\n" + "=" * 50)
    print("Saved Models List")
    print("=" * 50)

    model_files = glob.glob(os.path.join(saved_models_dir, "*.joblib"))

    if not model_files:
        print("No saved models found.")
        return []

    print("No. | Model Name      | Model Type | Accuracy | Save Time")
    print("-" * 50)

    models_info = []
    for i, filepath in enumerate(sorted(model_files), 1):
        try:
            loaded_data = joblib.load(filepath)
            save_time = loaded_data['metadata']['save_time']
            model_name = os.path.basename(filepath).replace('.joblib', '')
            model_type = loaded_data['metadata'].get('model_type', 'Unknown')

            # Get evaluation results
            eval_result = loaded_data.get('eval_result')
            if eval_result and 'accuracy' in eval_result:
                accuracy = f"{eval_result['accuracy']:.4f}"
            else:
                accuracy = "N/A"

            print(f"{i:^4} | {model_name:<15} | {model_type:<10} | {accuracy:<7} | {save_time}")
            models_info.append((filepath, model_name, save_time, accuracy, eval_result, model_type))
        except Exception as e:
            model_name = os.path.basename(filepath)
            print(f"{i:^4} | {model_name:<15} | Load Failed | Unknown | Unknown")
            models_info.append((filepath, model_name, "Unknown", "Load Failed", None, "Unknown"))

    # Ask to view detailed evaluation report for a specific model
    if models_info:
        try:
            choice = input("\nEnter the number to view detailed evaluation report for a model, or press Enter to return to main menu: ").strip()
            if choice.lower() == 'q':
                print("Returning to main menu.")
                return []
            if choice:
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(models_info):
                    _, model_name, _, _, eval_result, model_type = models_info[choice_idx]
                    if eval_result:
                        print(f"\n--- Detailed Evaluation Report for Model '{model_name}' ({model_type}) ---")
                        print(f"Accuracy: {eval_result['accuracy']:.4f}")
                        print("\nClassification Report (dictionary format):")
                        for class_name, metrics in eval_result['classification_report'].items():
                            if isinstance(metrics, dict):
                                print(f"  {class_name}: {metrics}")
                        print(f"\nConfusion Matrix: {eval_result['confusion_matrix']}")
                        if 'model_params' in eval_result:
                            print(f"\nModel Parameters: {eval_result['model_params']}")
                    else:
                        print("This model has no saved evaluation results.")
        except ValueError:
            pass  # User entered non-number, return directly

    return models_info


def load_model_interactive():
    """Interactively load a model and update data for generation"""
    global current_model, current_kmer_vocab, current_train_df, current_model_eval, current_amp_seqs_for_generation, current_model_name

    # List saved models (do not show detailed evaluation report)
    print("\n" + "=" * 50)
    print("Saved Models List")
    print("=" * 50)

    model_files = glob.glob(os.path.join(saved_models_dir, "*.joblib"))

    if not model_files:
        print("No saved models found.")
        return None, None, None, [], None

    print("No. | Model Name      | Model Type | Accuracy | Save Time")
    print("-" * 50)

    models_info = []
    for i, filepath in enumerate(sorted(model_files), 1):
        try:
            loaded_data = joblib.load(filepath)
            save_time = loaded_data['metadata']['save_time']
            model_name = os.path.basename(filepath).replace('.joblib', '')
            model_type = loaded_data['metadata'].get('model_type', 'Unknown')

            # Get evaluation results
            eval_result = loaded_data.get('eval_result')
            if eval_result and 'accuracy' in eval_result:
                accuracy = f"{eval_result['accuracy']:.4f}"
            else:
                accuracy = "N/A"

            print(f"{i:^4} | {model_name:<15} | {model_type:<10} | {accuracy:<7} | {save_time}")
            models_info.append((filepath, model_name, save_time, accuracy, eval_result, model_type, loaded_data))
        except Exception as e:
            model_name = os.path.basename(filepath)
            print(f"{i:^4} | {model_name:<15} | Load Failed | Unknown | Unknown")
            models_info.append((filepath, model_name, "Unknown", "Load Failed", None, "Unknown", None))

    if not models_info:
        print("No models available to load.")
        return None, None, None, [], None

    while True:
        try:
            choice = input("\nPlease enter the number of the model to load (enter 'q' to return to main menu): ").strip()

            if choice.lower() == 'q':
                print("Load cancelled, returning to main menu.")
                return None, None, None, [], None

            choice_idx = int(choice) - 1

            if 0 <= choice_idx < len(models_info):
                filepath, model_name, _, _, _, model_type, loaded_data = models_info[choice_idx]

                if loaded_data is None:
                    print(f"Model '{model_name}' failed to load.")
                    return None, None, None, [], None

                try:
                    current_model = loaded_data['model']
                    current_kmer_vocab = loaded_data['kmer_vocab']
                    current_model_eval = loaded_data.get('eval_result')
                    # Key: Load data for antibacterial peptide generation from the saved file
                    current_amp_seqs_for_generation = loaded_data.get('train_amp_seqs', [])
                    # Update current model name
                    current_model_name = model_name

                    print(f"\nModel '{model_name}' ({model_type}) loaded successfully!")
                    print(f"Save time: {loaded_data['metadata']['save_time']}")
                    print(f"Also loaded {len(current_amp_seqs_for_generation)} antibacterial peptide sequences for generation.")
                    print("\nReturning to main menu...")

                    return current_model, current_kmer_vocab, current_model_eval, current_amp_seqs_for_generation, current_model_name

                except Exception as e:
                    print(f"Error loading model: {e}")
                    return None, None, None, [], None
            else:
                print(f"Invalid number, please enter a number between 1 and {len(models_info)}.")

        except ValueError:
            print("Invalid input, please enter a number or 'q'.")


def predict_sequence_interactive():
    """Interactive sequence prediction"""
    global current_model, current_kmer_vocab

    if current_model is None or current_kmer_vocab is None:
        print("Error: No model loaded, please train or load a model first.")
        return

    print("\n" + "=" * 50)
    print("Sequence Prediction")
    print("=" * 50)

    while True:
        seq = input("Please enter an amino acid sequence (enter 'q' to return to main menu): ").strip()

        if seq.lower() == 'q':
            print("Returning to main menu.")
            return

        if not seq:
            print("Sequence cannot be empty, please re-enter.")
            continue

        proba = predict_sequence(current_model, current_kmer_vocab, seq)

        if proba is not None:
            print(f"\nSequence prediction result:")
            print(f"  Input sequence: {seq}")
            print(f"  Sequence length: {len(seq)}")
            print(f"  Probability of being an antibacterial peptide: {proba:.4f} ({proba * 100:.2f}%)")
            print(f"  Predicted class: {'Antibacterial peptide' if proba >= 0.5 else 'Non-antibacterial peptide'}")
            print("-" * 30)


def generate_peptide_interactive():
    """Interactive antibacterial peptide generation - automatically uses current or saved data"""
    global current_model, current_kmer_vocab, current_train_df, current_amp_seqs_for_generation

    # Determine the source of antibacterial peptide sequences for generation
    train_amp_seqs = []

    # Source 1: Current training data (if loaded)
    if current_train_df is not None:
        train_amp_seqs = current_train_df[current_train_df['label'] == 1]['sequence'].tolist()
        print("[Info] Using antibacterial peptide sequences from the current training set for generation.")
    # Source 2: From loaded model data (if a model is loaded)
    elif current_amp_seqs_for_generation:
        train_amp_seqs = current_amp_seqs_for_generation
        print(f"[Info] Using antibacterial peptide sequences obtained from the loaded model for generation.")
    else:
        print("Error: No training data available for antibacterial peptide generation.")
        print("Please perform one of the following operations first:")
        print("  1. Execute '1. Train Model' to load and process data")
        print("  2. Execute '3. Load Model' to load a model containing training data")
        return

    if not train_amp_seqs:
        print("Error: No antibacterial peptide sequences in the available training data.")
        return

    print("\n" + "=" * 50)
    print("Generate Antibacterial Peptide")
    print("=" * 50)
    print(f"Current template library for generation contains {len(train_amp_seqs)} antibacterial peptide sequences.")

    while True:
        length_input = input("Please enter the length of the antibacterial peptide to generate (10-50) (enter 'q' to return to main menu): ").strip()

        if length_input.lower() == 'q':
            print("Returning to main menu.")
            return

        try:
            length = int(length_input)

            if length < 10:
                print("Warning: Length less than 10, generation quality may be poor.")
                confirm = input("Continue? (y/n, enter 'q' to return to main menu): ").lower()
                if confirm == 'q':
                    print("Returning to main menu.")
                    return
                if confirm != 'y':
                    continue
            elif length > 50:
                print("Warning: Length exceeds 50, generation time may be long.")
                confirm = input("Continue? (y/n, enter 'q' to return to main menu): ").lower()
                if confirm == 'q':
                    print("Returning to main menu.")
                    return
                if confirm != 'y':
                    continue

            print("\nGenerating antibacterial peptide, please wait...")

            # Use the current model for high-probability generation
            peptide, proba = generate_antimicrobial_peptide(
                length=length,
                train_amp_seqs=train_amp_seqs,
                order=2,
                max_attempts=100,
                target_prob=0.8,
                model=current_model,
                kmer_vocab=current_kmer_vocab
            )

            if peptide:
                print(f"\nGenerated Antibacterial Peptide (length {length}):")
                print(f"  Sequence: {peptide}")
                print(f"  Model's predicted probability of being an antibacterial peptide: {proba:.4f} ({proba * 100:.2f}%)")

                if current_model is not None:
                    if proba >= 0.8:
                        print("  ✅ High-quality antibacterial peptide (probability >= 0.8)")
                    elif proba >= 0.5:
                        print("  ⚠️  Medium-quality antibacterial peptide (0.5 <= probability < 0.8)")
                    else:
                        print("  ❌ Low-quality antibacterial peptide (probability < 0.5)")
                print("-" * 30)
            else:
                print("Failed to generate antibacterial peptide.")

        except ValueError:
            print("Invalid input, please enter an integer or 'q'.")


# ==================== 8. Main Dialog Interface (7 options) ====================
def show_main_menu():
    """Display the main menu (7 options)"""
    print("\n" + "=" * 60)
    print("Antibacterial Peptide Prediction and Generation System - Main Menu")
    print("=" * 60)
    print("Current status:", end=" ")
    if current_model is not None:
        if current_model_name:
            print(f"✅ Model loaded ({current_model_name})", end=" | ")
        else:
            print("✅ Model loaded", end=" | ")
    else:
        print("❌ No model", end=" | ")

    if current_train_df is not None or current_amp_seqs_for_generation:
        amp_seq_count = len(current_amp_seqs_for_generation) if current_amp_seqs_for_generation else 0
        print(f"✅ Generation data available ({amp_seq_count} sequences)")
    else:
        print("❌ No generation data")

    print("-" * 60)
    print("Please select the operation to perform:")
    print("  1. Train Model (default parameters)")
    print("  2. Model Optimization (custom parameters)")
    print("  3. View Existing Models (display evaluation results)")
    print("  4. Load Model (select from saved models)")
    print("  5. Predict if a Sequence is an Antibacterial Peptide")
    print("  6. Generate Antibacterial Peptide")
    print("  7. Exit Program")
    print("")
    print("Hint: During any input step, enter 'q' to return to the main menu")
    print("=" * 60)


def main():
    """Main program"""
    global current_model, current_kmer_vocab, current_train_df, current_val_df, current_full_df, current_model_eval, current_amp_seqs_for_generation, current_model_name

    print("=" * 60)
    print("Antibacterial Peptide Prediction and Generation System (Dialog Version) - Multi-Model Selection and Optimization Edition")
    print("=" * 60)
    print("System initialization complete, entering main menu...")

    # Check for required files
    if not os.path.exists('data.xlsx'):
        print("Warning: 'data.xlsx' file not found.")
        print("Please place the data file in the current directory.")

    while True:
        show_main_menu()
        choice = input("Please enter an option (1-7): ").strip()

        if choice == '1':
            # Train Model (using default parameters)
            print("\nLoading data and training model (using default parameters)...")
            current_train_df, current_val_df, current_full_df = load_and_prepare_data('data.xlsx')

            # Reset model name because a new model is being trained
            current_model_name = None

            if current_train_df is not None:
                # --- Let user select model type ---
                print("\nPlease select the model type to train:")
                print("  1. Random Forest")
                print("  2. Support Vector Machine (SVM)")
                if XGB_AVAILABLE:
                    print("  3. XGBoost")
                else:
                    print("  3. XGBoost (Unavailable, library not installed)")
                if LGBM_AVAILABLE:
                    print("  4. LightGBM")
                else:
                    print("  4. LightGBM (Unavailable, library not installed)")

                model_choice = input("Please enter an option (1-4, default is 1, enter 'q' to return to main menu): ").strip()
                if model_choice.lower() == 'q':
                    print("Returning to main menu.")
                    continue
                model_type = 'random_forest'  # Default

                if model_choice == '2':
                    model_type = 'svm'
                elif model_choice == '3' and XGB_AVAILABLE:
                    model_type = 'xgboost'
                elif model_choice == '3' and not XGB_AVAILABLE:
                    print("XGBoost library not installed, will use default Random Forest.")
                    print("To use XGBoost, please run 'pip install xgboost' to install.")
                elif model_choice == '4' and LGBM_AVAILABLE:
                    model_type = 'lightgbm'
                elif model_choice == '4' and not LGBM_AVAILABLE:
                    print("LightGBM library not installed, will use default Random Forest.")
                    print("To use LightGBM, please run 'pip install lightgbm' to install.")
                else:
                    model_type = 'random_forest'

                print(f"\nYou have selected the {model_type} model.")
                # --- End of selection ---

                current_model, current_kmer_vocab, current_model_eval = train_and_evaluate(current_train_df,
                                                                                           current_val_df,
                                                                                           model_type=model_type)  # Pass model type
                if current_model is not None:
                    print("✅ Model training completed!")
                    # Update data for generation
                    current_amp_seqs_for_generation = current_train_df[current_train_df['label'] == 1][
                        'sequence'].tolist()
                    print(f"Generation data updated, containing {len(current_amp_seqs_for_generation)} antibacterial peptide sequences.")

                    # Ask whether to save after training
                    save_now = input("\nSave the model immediately? (y/n, enter 'q' to return to main menu): ").strip().lower()
                    if save_now == 'q':
                        print("Returning to main menu.")
                        continue
                    if save_now == 'y':
                        if current_train_df is not None:
                            train_amp_seqs = current_train_df[current_train_df['label'] == 1]['sequence'].tolist()
                            # Ask for model name
                            while True:
                                model_name = input("Please enter the model name to save (without extension, enter 'q' to return to main menu): ").strip()
                                if model_name.lower() == 'q':
                                    print("Returning to main menu.")
                                    break
                                if not model_name:
                                    print("Model name cannot be empty, please re-enter.")
                                    continue
                                # Call save_model with model type
                                if save_model(current_model, current_kmer_vocab, train_amp_seqs, current_model_eval,
                                              model_name, model_type=model_type):
                                    current_model_name = model_name
                                    break
                                else:
                                    print("Save failed, please try again.")
                        else:
                            print("Error: Training data unavailable, cannot save.")
                    else:
                        print("Model not saved. You can view the current model in the main menu or load it later.")

        elif choice == '2':
            # Model Optimization (custom parameters)
            model_optimization_interactive()

        elif choice == '3':
            # View Existing Models (enhanced version, shows evaluation results)
            list_saved_models()

        elif choice == '4':
            # Load Model
            loaded_model, loaded_vocab, loaded_eval, loaded_amp_seqs, loaded_model_name = load_model_interactive()
            if loaded_model is not None:
                current_model = loaded_model
                current_kmer_vocab = loaded_vocab
                current_model_eval = loaded_eval
                current_amp_seqs_for_generation = loaded_amp_seqs
                current_model_name = loaded_model_name

        elif choice == '5':
            # Predict Sequence
            predict_sequence_interactive()

        elif choice == '6':
            # Generate Antibacterial Peptide (automatically uses saved/loaded data)
            generate_peptide_interactive()

        elif choice == '7':
            # Exit Program
            print("\nThank you for using the Antibacterial Peptide Prediction and Generation System. Goodbye!")
            break

        else:
            print("Invalid option, please enter a number between 1 and 7.")


# ==================== 9. Program Entry Point ====================
if __name__ == "__main__":
    main()