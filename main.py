import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Import your custom classes
from model_naive_bayes import DiabetesNaiveBayes
from model_random_forest import DiabetesRandomForest

def impute_zeros(df, columns):
    """
    Replaces 0 values with the median of the non-zero values for specified columns.
    """
    df_imputed = df.copy()
    for col in columns:
        # Replace 0 with NaN so they are excluded from the median calculation
        df_imputed[col] = df_imputed[col].replace(0, np.nan)
        
        # Calculate the median of the column (Pandas .median() skips NaNs automatically)
        median_val = df_imputed[col].median()
        
        # Fill the NaN values with the calculated median
        df_imputed[col] = df_imputed[col].fillna(median_val)
        
        print(f"Column '{col}': Replaced 0s with median value {median_val}")
    
    return df_imputed

def apply_clamp_transformation(df, columns):
    """
    Clamps values to be within Mean +/- 3 Standard Deviations.
    Eq: ai = lower if ai < lower, upper if ai > upper, else ai
    """
    df_clamped = df.copy()
    
    for col in columns:
        # 1. Calculate Statistics
        mean = df_clamped[col].mean()
        std = df_clamped[col].std()
        
        # 2. Define Thresholds
        upper_limit = mean + (3 * std)
        lower_limit = mean - (3 * std)
        
        # 3. Apply Clamping
        df_clamped[col] = df_clamped[col].clip(lower=lower_limit, upper=upper_limit)
        
    return df_clamped

def load_and_preprocess_data(filepath):
    """
    Loads data and orchestrates the preprocessing pipeline:
    1. Load CSV
    2. Impute Zeros (Median Imputation)
    3. Clamp Outliers (Winsorization)
    """
    # 1. Load Data
    print(f"Reading data from {filepath}...")
    df = pd.read_csv(filepath)

    # 2. Impute Zeros
    # (Columns that logically cannot be 0)
    impute_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df = impute_zeros(df, impute_cols)

    # 3. Clamp Outliers
    # (All features excluding 'Outcome' and 'Age' as per lecture notes)
    #clamp_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
    #              'Insulin', 'BMI', 'DiabetesPedigreeFunction']
    #df = apply_clamp_transformation(df, clamp_cols)

    return df

def main():
    # --- 1. Data Preparation ---
    print("Loading and Preprocessing data...")
    # Load data using the pipeline we built (Impute -> Clamp)
    df = load_and_preprocess_data('diabetes.csv')

    # Define features (X) and target (y)
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']

    # Split data into Train and Test sets (80/20 split)
    # using random_state=42 for reproducibility
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # --- 2. Naive Bayes Execution ---
    print("\nRunning Naive Bayes Model...")
    nb_model = DiabetesNaiveBayes()
    
    nb_model.train(X_train, y_train)           # Train
    nb_predictions = nb_model.predict(X_test)  # Predict
    nb_accuracy = nb_model.evaluate(y_test, nb_predictions) # Evaluate

    # --- 3. Random Forest Execution ---
    print("\nRunning Random Forest Model...")
    rf_model = DiabetesRandomForest()
    
    rf_model.train(X_train, y_train)           # Train
    rf_predictions = rf_model.predict(X_test)  # Predict
    rf_accuracy = rf_model.evaluate(y_test, rf_predictions) # Evaluate

    # --- 4. Comparison ---
    print("\n--- Final Conclusion ---")
    print(f"Naive Bayes Accuracy:  {nb_accuracy:.4f}")
    print(f"Random Forest Accuracy: {rf_accuracy:.4f}")
    
    if rf_accuracy > nb_accuracy:
        print("Conclusion: The Random Forest model performed better on this dataset.")
        print("This suggests that the complex, non-linear relationships in diabetes indicators")
        print("were better captured by the decision tree ensemble than the probabilistic approach.")
    elif nb_accuracy > rf_accuracy:
        print("Conclusion: The Naive Bayes model performed better on this dataset.")
        print("This suggests that the data distribution is well-approximated by a Gaussian curve")
        print("and the outliers were effectively handled by our clamping preprocessing.")
    else:
        print("Conclusion: Both models performed equally well.")
    
if __name__ == "__main__":
    main()