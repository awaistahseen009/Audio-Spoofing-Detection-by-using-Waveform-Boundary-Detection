import numpy as np
from sklearn.metrics import roc_curve


def calculate_eer(labels, outputs):
    fpr, tpr, thresholds = roc_curve(labels, outputs)
    fnr = 1 - tpr
    eer_threshold = thresholds[np.nanargmin(np.abs(fnr - fpr))]
    eer = fpr[np.nanargmin(np.abs(fnr - fpr))]
    return eer, eer_threshold
