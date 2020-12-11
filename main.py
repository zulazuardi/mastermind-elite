import logging
import numpy as np
import pandas as pd

from sklearn.metrics import classification_report
from logger import setup_logging
from classifier.classifier import IsReturningCustomerClassifier as Classifier

setup_logging(
    filename="main"
)
logger = logging.getLogger(__name__)


def get_feature_importance(model):
    """
        Get feature importances
        Args:
            model  : model

        Returns:
            feature importances
    """
    importances = model.feature_important

    return importances


def get_classification_report(actual, predictions):
    """
        Get classification report, f1 score
        Args:
            actual  (list): actual result
            predictions (list): prediction result from the model

        Returns:
            summary of classification report including accuracy and f1 score
    """
    return classification_report(
        actual,
        predictions
    )


def get_test_data(filename):
    """
        Get test dataset
        Args:
            filename  (str): test dataframe

        Returns:
            features, label  : dataframe
    """
    features = pd.read_csv(
        "data/" + filename + "_feature.csv", index_col=0
    )
    label = pd.read_csv("data/" + filename + "_label.csv", index_col=0)
    return features, label


def main():
    classifier = Classifier(model_name="random_forest")

    logger.debug(
        "top 20 feature importances: {}".format(
            get_feature_importance(classifier)
        )
    )

    test_features, test_labels = get_test_data("test")

    logger.debug(
        "classification report: {}".format(
            get_classification_report(
                test_labels["is_returning_customer"].values,
                classifier.classify(test_features))
        )
    )


if __name__ == "__main__":
    main()
