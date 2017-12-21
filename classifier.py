from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from fastText import train_supervised

class Classifier():
    def __init__(self, classifier_type):
        """
        This is the class to create new instance of classifiers from ["SVM", "LR", "FT"]
        """
        if classifier_type == "SVM":
            config = {}
            self.model = LinearSVC(config)
        elif classifier_type == "LR":
            config = {}
            self.model = LogisticRegression(config)
        elif classifier_type == "FT":
            config = {}
            self.model = train_supervised(config)