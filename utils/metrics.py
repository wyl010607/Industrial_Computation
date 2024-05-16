import numpy as np

import numpy as np


def get_acc(y_true, y_pred):
    """
    Accuracy
    Formula: Accuracy = (Number of correct predictions) / (Total number of predictions)
    """
    # Ensure both y_true and y_pred have the same shape
    if len(y_true) != len(y_pred):
        raise ValueError("Input shapes of y_true and y_pred must be the same.")

    correct_predictions = 0
    for i, j in zip(y_true, y_pred):
        if i == j:
            correct_predictions += 1

    accuracy = correct_predictions / len(y_true) * 100.0

    return accuracy


def get_mae(y_true, y_pred):
    """
    Mean Absolute Error (MAE)
    Formula: MAE = mean(|y_true - y_pred|)
    """
    return np.mean(np.abs(y_true - y_pred))


def get_mse(y_true, y_pred):
    """
    Mean Squared Error (MSE)
    Formula: MSE = mean((y_true - y_pred)^2)
    """
    return ((y_true - y_pred) ** 2).mean()


def get_rmse(y_true, y_pred):
    """
    Root Mean Squared Error (RMSE)
    Formula: RMSE = sqrt(mean((y_true - y_pred)^2))
    """
    return np.sqrt(((y_true - y_pred) ** 2).mean())


def get_mape(y_true, y_pred):
    """
    Mean Absolute Percentage Error (MAPE)
    Formula: MAPE = mean(|(y_true - y_pred) / y_true|) * 100
    """
    non_zero_mask = y_true != 0
    y_true_masked = y_true[non_zero_mask]
    y_pred_masked = y_pred[non_zero_mask]
    mape = np.mean(np.abs((y_true_masked - y_pred_masked) / y_true_masked)) * 100
    return mape


def get_rmspe(y_true, y_pred):
    """
    Root Mean Square Percentage Error (RMSPE)
    Formula: RMSPE = sqrt(mean(((y_true - y_pred) / y_true)^2)) * 100
    """
    non_zero_mask = y_true != 0
    rmspe = (
            np.sqrt(
                np.mean(
                    np.square(
                        (y_true[non_zero_mask] - y_pred[non_zero_mask])
                        / y_true[non_zero_mask]
                    )
                )
            )
            * 100
    )
    return rmspe


def get_bce(y_true, y_pred):
    """
    Binary Cross-Entropy (BCE)
    Formula: BCE = -y_true*log(y_pred) - (1 - y_true)*log(1 - y_pred)
    """
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred)


def get_cce(y_true, y_pred):
    """
    Categorical Cross-Entropy (CCE)
    Formula: CCE = -sum(y_true*log(y_pred))
    """
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.sum(y_true * np.log(y_pred), axis=1).mean()


def get_hinge(y_true, y_pred):
    """
    Hinge Loss
    Formula: Hinge = mean(max(0, 1 - y_true*y_pred))
    """
    return np.mean(np.maximum(1 - y_true * y_pred, 0))
