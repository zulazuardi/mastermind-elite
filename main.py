import logging
import os

from config import config
from logger import setup_logging
from classifier.classifier import IsReturningCustomerClassifier as Classifier

setup_logging(
    level=config.log_level,
    directory=config.log_dir,
    filename="main"
)
logger = logging.getLogger(__name__)


def main():
    classifier = Classifier(model_name="random_forest")

    classifier.classify()