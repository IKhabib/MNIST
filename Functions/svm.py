import numpy as np


def hinge_loss(X, y, w, b, C):
    """
    Computes hinge loss with L2 regularization.

    Args:
        X: (n_samples, n_features)
        y: (n_samples,) labels (-1 or 1)
        w: (n_features,) weight vector
        b: scalar bias
        C: regularization strength

    Returns:
        scalar total loss
    """
    margins = 1 - y * (X @ w + b)
    hinge = np.maximum(0, margins)
    loss = 0.5 * np.dot(w, w) + C * np.sum(hinge)
    return loss


def train_linear_svm(X, y, C=0.1, learning_rate=1e-3, max_iter=1000, verbose=False):
    """
    Trains a linear SVM using gradient descent and hinge loss.

    Args:
        X: (n_samples, n_features)
        y: (n_samples,) labels (0 or 1)
        C: regularization strength
        learning_rate: step size
        max_iter: number of iterations
        verbose: whether to print loss every 100 steps

    Returns:
        w: trained weight vector
        b: trained bias term
    """
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0

    # Convert labels to -1 and +1
    y_transformed = np.where(y <= 0, -1, 1)

    for it in range(max_iter):
        for i in range(n_samples):
            xi = X[i]
            yi = y_transformed[i]
            margin = yi * (np.dot(w, xi) + b)

            if margin >= 1:
                w -= learning_rate * w
            else:
                w -= learning_rate * (w - C * yi * xi)
                b += learning_rate * C * yi

        if verbose and it % 100 == 0:
            loss = hinge_loss(X, y_transformed, w, b, C)
            print(f"Iteration {it}: Loss = {loss:.4f}")

    return w, b


def predict_linear_svm(X, w, b):
    """
    Predicts binary labels using the trained linear SVM.

    Args:
        X: (n_samples, n_features)
        w: (n_features,) weight vector
        b: bias term

    Returns:
        predictions: (n_samples,) 0 or 1
    """
    scores = X @ w + b
    return np.where(scores >= 0, 1, 0)

def one_vs_rest_svm(train_x, train_y, test_x):
    """
    Custom linear SVM for binary classification (0 vs 1).

    Args:
        train_x: (n, d)
        train_y: (n,) 0 or 1
        test_x: (m, d)

    Returns:
        pred_test_y: (m,) 0 or 1
    """
    w, b = train_linear_svm(train_x, train_y, C=0.1, learning_rate=1e-4, max_iter=1000, verbose=False)
    pred_test_y = predict_linear_svm(test_x, w, b)
    return pred_test_y

def compute_test_error_svm(test_y, pred_test_y):
    return 1 - np.mean(pred_test_y == test_y)

