import os.path
from collections import defaultdict

import numpy as np
import pandas
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import np_utils
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.moo.loss.classification import WeightedCategoricalCrossentropy
from src.utils.plot import plot_classification_results

cwp = os.path.dirname(os.path.abspath(__file__))

def repeat_model_evaluation(weights, n_repeat, X, Y, epochs=20):
    results = defaultdict(list)
    for i in range(n_repeat):
        print('evaluating model: {}/{}'.format(i + 1, n_repeat), end='\r')
        evaluate_model(weights, results, X, Y, epochs=epochs, verbose=0)

    for key, val in results.items():
        results[key] = np.array(val)
    return results


def evaluate_model(weights, results, X, Y, epochs=10, verbose=1):
    model = iris_model(weights)
    model.fit(X, Y, epochs=epochs, verbose=verbose)
    y_pred = np.argmax(model.predict(X), axis=1)
    y_true = np.argmax(Y, axis=1)
    precision, recall, fbeta, support = precision_recall_fscore_support(y_true, y_pred)

    for key, val in zip(['precision', 'recall', 'fbeta'], [precision, recall, fbeta]):
        results[key].append(val)

def iris_model(weights=None):
    loss_fn = WeightedCategoricalCrossentropy(fp_weights=weights, from_logits=False).loss
    # create model
    model = Sequential()
    model.add(Dense(24, input_dim=4, activation='relu'))
    model.add(Dense(3, activation='softmax', name='moo_layer'))
    model.compile(loss=loss_fn, optimizer='adam', metrics=['accuracy'])
    return model

def iris_data():
    dataframe = pandas.read_csv(os.path.join(cwp, '../../../examples', 'problems', 'ref', "iris.csv"), header=None)
    dataset = dataframe.values
    X = dataset[:, 0:4].astype(float)
    Y = dataset[:, 4]
    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)
    # convert integers to dummy variables (i.e. one hot encoded)
    dummy_y = np_utils.to_categorical(encoded_Y)

    X_train, X_test, y_train, y_test = train_test_split(X, dummy_y, test_size=0.01, random_state=42)
    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    # %%
    X_train, X_test, y_train, y_test = iris_data()
    weights = np.ones(3)
    results_uw = repeat_model_evaluation(weights, 5, X_train, y_train, epochs=100)

    # %% Weighted cross entropy
    print('-' * 20)
    weights = np.ones(3) * 1.
    weights[1] = 1.5
    results_w = repeat_model_evaluation(weights, 5, X_train, y_train, epochs=100)

    # %% Compare cross entropy vs weighted cross entropy weights
    results = [results_uw, results_w]
    labels = ['Unweighted', 'Weighted']
    keys = ['precision', 'recall', 'fbeta']

    plot_classification_results(results, labels, keys, list(range(3)))
