from sklearn.metrics import accuracy_score
from sklearn.svm import SVC


def train_and_evaluate_svm(K_train, y_train, K_test, y_test):
    """
    Train an SVM with a precomputed kernel and return the test accuracy.
    """

    classifier = SVC(kernel="precomputed")
    classifier.fit(K_train, y_train)

    predictions = classifier.predict(K_test)
    return accuracy_score(y_test, predictions)
