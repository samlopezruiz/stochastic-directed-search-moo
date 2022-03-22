import numpy as np
from sklearn.metrics import precision_recall_fscore_support


def classification_eval_metrics(X, Y, model):
    y_pred = np.argmax(model.predict(X), axis=1)
    y_true = np.argmax(Y, axis=1)
    precision, recall, fbeta, support = precision_recall_fscore_support(y_true, y_pred)

    metrics = {}
    for key, val in zip(['precision', 'recall', 'fbeta'], [precision, recall, fbeta]):
        metrics[key] = val
    return metrics