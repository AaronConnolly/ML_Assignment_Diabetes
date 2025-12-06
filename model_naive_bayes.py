from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class DiabetesNaiveBayes:
    def __init__(self):
        """
        Initialize the Gaussian Naive Bayes model.
        """
        self.model = GaussianNB()

    def train(self, X_train, y_train):
        """
        Train the model on the provided data.
        """
        # Fitting the model to the training data
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        """
        Make predictions on the test set.
        """
        # Generating predictions
        return self.model.predict(X_test)

    def evaluate(self, y_test, y_pred):
        """
        Print accuracy and classification report.
        Returns accuracy float for comparison.
        """
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Accuracy Score: {accuracy:.4f}")
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        return accuracy