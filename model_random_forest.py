from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class DiabetesRandomForest:
    def __init__(self):
        """
        Initialize the Random Forest model. 
        """
        # We use n_estimators=100 (100 trees) which is a standard default.
        # random_state=42 ensures the trees are built the same way every time its run.
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)

    def train(self, X_train, y_train):
        """
        Train the model on the provided data.
        """
        # Fits the 100 decision trees to your training data
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        """
        Make predictions on the test set.
        """
        # Aggregates the votes from all 100 trees to decide the class
        return self.model.predict(X_test)

    def evaluate(self, y_test, y_pred):
        """
        Print accuracy and classification report.
        Returns accuracy float for comparison in main.py.
        """
        print("--- Random Forest Performance ---")
        
        # Calculate standard accuracy
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Accuracy Score: {accuracy:.4f}")
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Return this value so main.py can compare it against Naive Bayes
        return accuracy