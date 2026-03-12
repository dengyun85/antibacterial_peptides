# Manual of Antibacterial Peptide Prediction and Generation System

## Main Functions

### **Option 1: Train Model (Default Parameters)**
*   **Function**: Quickly loads `data.xlsx` and trains a model using default parameters (70% training set, 30% validation set).
*   **Process**:
    1.  Load and preprocess the data.
    2.  Select the model type (1. Random Forest, 2. SVM, 3. XGBoost, 4. LightGBM).
    3.  Train the model with default parameters, evaluate it on the validation set, and display the accuracy, classification report, and confusion matrix.
    4.  After training, you will be prompted to save the model immediately. If you choose to save, you need to provide a name for the model (e.g., `my_model`). The model will be saved in the `./saved_models/` directory.
*   **Result**: The trained model, feature vocabulary, and evaluation results are set as the "Current Model" and can be used for subsequent prediction and generation tasks.

### **Option 2: Model Optimization (Custom Parameters)**
*   **Function**: Provides finer control, allowing users to customize model parameters, dataset split ratio, and random seed.
*   **Configurable Items**:
    *   Model type selection.
    *   Validation set proportion (default: 0.3).
    *   Random seed (default: 42).
    *   Key hyperparameters for the selected model (e.g., number of trees, depth, learning rate, etc.).
*   **Process**: Similar to Option 1, but an interface for parameter input is provided at all steps. Training begins after final confirmation.

### **Option 3: View Existing Models**
*   **Function**: Lists all saved `.joblib` model files in the `./saved_models/` directory, displaying their name, type, validation accuracy, and save time.
*   **Operation**: You can select any model to view its detailed evaluation report (including the classification report dictionary and confusion matrix).

### **Option 4: Load Model**
*   **Function**: Select a saved model from the `./saved_models/` directory to load into memory as the "Current Model".
*   **Key Point**: When loading the model, **the antimicrobial peptide sequence data used during its training is also loaded**. This data will be used for the subsequent "Generate Antimicrobial Peptides" function.
*   **Result**: After loading, the "Current Status" on the main menu is updated, and you can proceed directly to prediction or generation.

### **Option 5: Predict if a Sequence is an Antimicrobial Peptide**
*   **Prerequisite**: A "Current Model" must be loaded or already trained.
*   **Function**: Enters interactive mode where the user can input a single amino acid sequence.
*   **Output**: The program outputs the predicted probability (as a value and percentage) and the final classification result (threshold: 0.5) for the sequence being an antimicrobial peptide.

### **Option 6: Generate Antimicrobial Peptides**
*   **Prerequisite**: Antimicrobial peptide sequence data must be available for generation. This data comes from two sources:
    1.  Automatically obtained after training a model via **Option 1 or 2**.
    2.  Obtained when loading a saved model via **Option 4**.
*   **Function**: Generates new sequences of a specified length based on the existing antimicrobial peptide sequence library, using a Markov model.
*   **Optimization**: If a model is currently loaded, the generation process uses this model to score the generated sequences and attempts to produce sequences with a high predicted probability (default target > 0.8).
*   **Process**: Input the desired peptide length (e.g., 20). The program will attempt to generate and output the sequence along with its model-predicted probability.

### **Option 7: Exit Program**
Exits the system.

---

## Important Notes & FAQs

1.  **First Run**: Ensure the data file named `data.xlsx` is placed in the same directory as the script before the first run.
2.  **Model-Data Binding**: The "Generate Antimicrobial Peptides" function heavily relies on the data used to train the model. **Only models trained or loaded through this program come with the data necessary for generation.** Directly training a model without saving it, or importing a model object from an external source, may prevent the use of the generation function.
3.  **Saving Models**: It is recommended to save the model after training. The model file (`.joblib`) contains the model parameters, feature vocabulary, evaluation results, and the **training antimicrobial peptide sequences**, facilitating later reuse.
4.  **Dependencies**: If you intend to use XGBoost or LightGBM, please ensure they are installed. Otherwise, the program will prompt you and automatically fall back to the Random Forest model.
5.  **Sequence Input**: During prediction, the program automatically filters invalid characters (non-standard 20 amino acid codes) from the input sequence. However, it is recommended to input a clean sequence string.
