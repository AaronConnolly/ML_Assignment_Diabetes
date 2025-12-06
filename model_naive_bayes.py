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
        # TODO: Fit the self.model using X_train and y_train
        pass

    def predict(self, X_test):
        """
        Make predictions on the test set.
        """
        # TODO: Return predictions using self.model.predict(X_test)
        pass

    def evaluate(self, y_test, y_pred):
        """
        Print accuracy and classification report.
        """
        # TODO: Calculate accuracy using accuracy_score
        # TODO: Print classification_report and confusion_matrix
        print("--- Naive Bayes Performance ---")
        pass