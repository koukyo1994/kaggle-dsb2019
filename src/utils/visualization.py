import itertools

import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
from typing import Sequence

from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(cm: np.ndarray,
                          classes: Sequence[str],
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          save_path: Path = Path("./")):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(save_path)


def calc_and_plot_cm(y_true: np.ndarray,
                     y_pred: np.ndarray,
                     normalize: bool = False,
                     title='Confusion matrix',
                     cmap=plt.cm.Blues,
                     save_path: Path = Path("./")):
    cm = confusion_matrix(y_true, y_pred)
    np.set_printoptions(precision=2)

    classes = [str(e) for e in np.unique(y_true).tolist()]
    plt.figure(figsize=(15, 15))
    plot_confusion_matrix(
        cm,
        classes,
        normalize=normalize,
        title=title,
        cmap=cmap,
        save_path=save_path)
