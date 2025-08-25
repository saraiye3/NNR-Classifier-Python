import time
import json
import pandas as pd
from pandas import DataFrame
from collections import Counter
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score, pairwise_distances
from typing import List

import numpy as np
from scipy.spatial import distance


def classify_with_NNR(data_trn: str, data_vld: str, df_tst: DataFrame) -> List:
    scaler = StandardScaler()

    # Load training and validation data
    dftr = pd.read_csv(data_trn)
    dfvl = pd.read_csv(data_vld)

    xtr = np.array(dftr.drop(columns='class').values)
    ytr = np.array(dftr['class'].values)

    x_valid = np.array(dfvl.drop(columns='class').values)
    y_valid = np.array(dfvl['class'].values)

    # Scale the training data and then the validation data
    xtr_scale = scaler.fit_transform(xtr)
    x_valid_scale = scaler.transform(x_valid)

    # Computing the distances between each validation point and all the training points X
    # Sorting in a matrix where each row is a validation point,
    # each column is a tuple of distance and class
    # the distances are sorted per validation point
    distances_matrix = []

    for i in range(len(x_valid_scale)):
        distances = list()
        for j in range(len(xtr_scale)):
            neighbore_distance = distance.euclidean(x_valid_scale[i], xtr_scale[j])
            distances.append((neighbore_distance, ytr[j]))
        distances.sort(key=lambda x: x[0])
        distances_matrix.append(distances)
# Finding optimal radius based on distances
    training_distances = pairwise_distances(xtr_scale)
    start_rad = np.percentile(training_distances, 5)
    end_rad = np.percentile(training_distances, 95)
    amount = int(np.ceil(np.sqrt(len(xtr))/len(xtr)))
    raddi = np.linspace(start_rad, end_rad, amount)


    best_radius = 0
    best_accuracy = 0
    for radius in raddi:
        predictions = list()
        for row in distances_matrix:
            neighbors = list()
            for col in row:
                if col[0] <= radius:
                    neighbors.append(col[1])
            if neighbors:
                most_common = Counter(neighbors).most_common(1)[0][0]
                predictions.append(most_common)
            else:
                predictions.append(-1)
        radius_accuracy = accuracy_score(y_valid, predictions)
        if radius_accuracy > best_accuracy:
            best_radius = radius


    print(f'Best radius found: {best_radius}')
    print(f'total time: {round(time.time() - start, 0)} sec')

    # Predicting on the test set
    predictions = []
    test_data = df_tst.values
    test_data_scale = scaler.transform(test_data)

    for i in range(len(test_data_scale)):
        predicted_neighbors = []
        for j in range(len(xtr_scale)):
            if distance.euclidean(test_data_scale[i], xtr_scale[j]) <= best_radius:
                predicted_neighbors.append(ytr[j])
        if predicted_neighbors:
            most_common = Counter(predicted_neighbors).most_common(1)[0][0]
            predictions.append(most_common)
        else:
            predictions.append(-1)

    return predictions

# todo: fill in your student ids
students = {'id1': '314919044'}

# Example usage
if __name__ == '__main__':
    start = time.time()

    with open('config.json', 'r', encoding='utf8') as json_file:
        config = json.load(json_file)

    df = pd.read_csv(config['data_file_test'])
    predicted = classify_with_NNR(config['data_file_train'],
                                  config['data_file_validation'],
                                  df.drop(['class'], axis=1))

    labels = df['class'].values
    if not predicted:  # empty prediction, should not happen in your implementation
        predicted = list(range(len(labels)))

    assert (len(labels) == len(predicted))  # make sure you predict label for all test instances
    print(f'test set classification accuracy: {accuracy_score(labels, predicted)}')

    print(f'total time: {round(time.time() - start, 0)} sec')