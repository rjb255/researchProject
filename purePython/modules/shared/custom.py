import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from sklearn.metrics import mean_squared_error as mse


def split(data: DataFrame, n: int = 20):
    ## Splits data (dataframe) insto a known section, an unknown section, and a testing section

    splitBoundary = [n, int(0.9 * len(data))]
    X1 = data.iloc[: splitBoundary[0], 2:]
    Y1 = data.iloc[: splitBoundary[0], 1]
    X2 = data.iloc[splitBoundary[0] : splitBoundary[1], 2:]
    Y2 = data.iloc[splitBoundary[0] : splitBoundary[1], 1]
    X3 = data.iloc[splitBoundary[1] :, 2:]
    Y3 = data.iloc[splitBoundary[1] :, 1]
    return X1, Y1, X2, Y2, X3, Y3


def getPI(known: tuple, unknown: tuple, index: int):
    X_known, Y_known = known
    X_unknown, Y_unknown = unknown
    X_known = X_known.append(X_unknown.loc[index])
    Y_known = Y_known.append(Y_unknown.loc[index])
    X_unknown = X_unknown.drop(index)
    Y_unknown = Y_unknown.drop(index)

    return X_known, Y_known, X_unknown, Y_unknown


class Models:
    """Creating a new model as an amalgamation of other models"""

    def __init__(self, m: list):
        """Defining the underlying models used

        Args:
            m (list): A list of models to be used in concert with each other
        """
        self.models = m

    def fit(self, X, Y, *args):
        """Fitting the models collectively as seen in most models.

        Args:
            X (DataFrame): data points to be fitted
            Y (DataFrame): labels of fitted data
        """
        self.trainedModels = [m.fit(X, Y, *args) for m in self.models]

    def predict(self, X, *args):
        """Equivelent to predict seen in other models.

        Args:
            X (DataFrame): the data points to be predicted

        Returns:
            ndArray: the mean of the prediction from the other models.
        """
        return np.mean([m.predict(X, *args) for m in self.trainedModels], axis=0)

    def predict_error(self, X, *args):
        """Similar to predict but also returns the standard deviation of the models predictions

        Args:
            X (DataFrame): The data points to be predicted.

        Returns:
            (ndArray, ndArray): The mean and standard deviation of the predictions from each model.
        """
        predictions = [m.predict(X, *args) for m in self.trainedModels]
        mean = np.mean(predictions, axis=0)
        std = np.std(predictions, axis=0, ddof=1)
        return mean, std


def validate(y_true, y_false):
    """Validation of the model.

    Args:
        y_true (DataFrame): The true test labels
        y_false (ndArray): The predicted test values

    Returns:
        float: score for the model prediction
    """
    return mse(y_true, y_false, sample_weight=y_true)
