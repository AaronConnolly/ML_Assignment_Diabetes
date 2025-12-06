import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Import your custom classes
from model_naive_bayes import DiabetesNaiveBayes
from model_random_forest import DiabetesRandomForest

def load_and_preprocess_data(filepath):
    """
    Loads data and handles the invalid zero values using Median Imputation.
    """
    # 1. Load Data
    df = pd.read_csv(filepath)

    # 2. Define columns that cannot logically be 0
    # (Based on your research: Glucose, BloodPressure, SkinThickness, Insulin, BMI)
    problematic_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

    # 3. Handling Zeros (The Imputation Logic)
    for col in problematic_cols:
        # TODO: Replace 0 with np.nan in this column
        # TODO: Calculate the median of this column (ignoring NaNs)
        # TODO: Fill the np.nan values with that median
        pass

    return df

def main():
    # --- 1. Data Preparation ---
    print("Loading and Preprocessing data...")
    df = load_and_preprocess_data('diabetes.csv')

    # TODO: Define features (X) and target (y)
    # X = df.drop('Outcome', axis=1)
    # y = df['Outcome']

    # TODO: Split data into Train and Test sets (e.g., 80/20 split)
    # X_train, X_test, y_train, y_test = train_test_split(...)

    # --- 2. Naive Bayes Execution ---
    print("\nRunning Naive Bayes Model...")
    nb_model = DiabetesNaiveBayes()
    # TODO: Call nb_model.train()
    # TODO: Call nb_model.predict()
    # TODO: Call nb_model.evaluate()

    # --- 3. Random Forest Execution ---
    print("\nRunning Random Forest Model...")
    rf_model = DiabetesRandomForest()
    # TODO: Call rf_model.train()
    # TODO: Call rf_model.predict()
    # TODO: Call rf_model.evaluate()

    # --- 4. Comparison ---
    print("\n--- Final Conclusion ---")
    # TODO: Print a statement comparing which model performed better
    
if __name__ == "__main__":
    main()