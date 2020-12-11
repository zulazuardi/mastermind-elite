import os
import logging
import pickle

from classifier import Classifier


class IsReturningCustomerClassifier(Classifier):
    def __init__(self, model_name):
        self.classifier = self._load_classifier(model_name)

    def _load_classifier(self, filename=None):
        """
        Set and return the classifier given a filename
        Args:
            filename (str, optional): Filename of the classifier. If None, use
                self.classifier_filename
        Returns:
            The classifier
        """
        log = logging.getLogger(__name__)

        filename = os.path.join(
            "model/" + filename + ".sav")
        log.debug("Loading classifier from %s", filename)
        if not os.path.isfile(filename):
            raise FileNotFoundError("No classifier file")

        with open(filename, "rb") as f:
            self.classifier = pickle.load(f)

        return self.classifier

    def classify(self, df):
        """
        Classify the customer.
        Args:
            df (dataframe): dataframe contains user features
        Returns:
            label customer will return or not (1 or 0)
        """
        model = self.classifier()

        return model.predict(df)
