import logging
import os
import numpy as np

from logger import setup_logging
from classifier.classifier import IsReturningCustomerClassifier as Classifier

setup_logging(
    level=config.log_level,
    directory=config.log_dir,
    filename="main"
)
logger = logging.getLogger(__name__)


def get_top20_feature_importance(model):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    result = []

    for f in range(21):
        result.append(
            {"feature": indices[f], "score": importances[indices[f]]}
        )
    return result


def main():
    classifier = Classifier(model_name="random_forest")

    logger.debug(
        "top 20 feature importances: {}".format(
            get_top20_feature_importance(classifier)
        )
    )


if __name__ == "__main__":
    main()
