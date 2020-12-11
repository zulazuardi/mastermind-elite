class Classifier:

    def __init__(self):
        pass

    def classify(self, df=None):
        """
        classify the customer for returning to the platform
        Args:
            df (dataframe)
        Returns:
            label customer will return or not (1 or 0)
        Raises:
            ValueError when df is None
        """
        raise NotImplementedError
