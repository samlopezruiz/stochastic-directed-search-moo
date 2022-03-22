from collections import defaultdict

import numpy as np
import pandas as pd
import plotly.io as pio
import tensorflow as tf
from sklearn.metrics import precision_recall_fscore_support

from src.moo.loss.classification import WeightedCategoricalCrossentropy
from src.utils.plot import plot_classification_results

pio.renderers.default = "browser"


def append_to_score(key, scores, precision, recall, fbeta):
    scores['precision{}'.format(key)] = precision
    scores['recall{}'.format(key)] = recall
    scores['fbeta{}'.format(key)] = fbeta


def mnist_fashion_model(weights):
    loss_fn = WeightedCategoricalCrossentropy(weights, from_logits=True).loss
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, name='moo_layer')
    ])
    model.compile(optimizer='adam',
                  loss=loss_fn,
                  metrics=['accuracy'])

    return model


def repeat_model_evaluation(weights, n_repeat, epochs=20):
    results = defaultdict(list)
    for i in range(n_repeat):
        print('evaluating model: {}/{}'.format(i + 1, n_repeat), end='\r')
        evaluate_model(weights, results, epochs=epochs, verbose=0)

    for key, val in results.items():
        results[key] = np.array(val)
    return results


def evaluate_model(weights, results, epochs=10, verbose=1):
    model = mnist_fashion_model(weights)
    model.fit(train_images, oh_train_labels, epochs=epochs, verbose=verbose)
    # test_loss, test_acc = model.evaluate(test_images, oh_test_labels, verbose=2)
    # print('\nTest accuracy:', test_acc)
    logits = model.predict(test_images)
    y_pred = np.argmax(logits, axis=1)
    precision, recall, fbeta, support = precision_recall_fscore_support(test_labels, y_pred)
    # scoresA = precision_score(test_labels, y_pred, average=None)
    for key, val in zip(['precision', 'recall', 'fbeta'], [precision, recall, fbeta]):
        results[key].append(val)


def fashion_mnist_data():
    fashion_mnist = tf.keras.datasets.fashion_mnist

    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    train_labels, test_labels = train_labels.astype(np.int32), test_labels.astype(np.int32)

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    train_images = train_images / 255.0
    test_images = test_images / 255.0

    oh_train_labels = tf.one_hot(train_labels, depth=10)
    oh_test_labels = tf.one_hot(test_labels, depth=10)

    return train_images, test_images, oh_train_labels, oh_test_labels, class_names


if __name__ == '__main__':
    scores = pd.DataFrame()

    print(tf.__version__)

    train_images, test_images, oh_train_labels, oh_test_labels, class_names = fashion_mnist_data()
    weights = np.ones(10)
    results_uw = repeat_model_evaluation(weights, n_repeat=5, epochs=30)

    # %% Weighted cross entropy
    print('-' * 20)
    weights = np.ones(10) * 0.5
    weights[6] = 2.
    weights[9] = 2.
    results_w = repeat_model_evaluation(weights, n_repeat=5, epochs=30)

    # %% Compare training of the two models
    results = [results_uw, results_w]
    labels = ['Unweighted', 'Weighted']
    keys = ['precision', 'recall', 'fbeta']

    plot_classification_results(results, labels, keys, class_names)
