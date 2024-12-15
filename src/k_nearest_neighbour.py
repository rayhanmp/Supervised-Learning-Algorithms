import numpy as np
import pickle

# @author: Jasmine Callista Aurellie Irfan

class KNN:
    def __init__(self, k_neighbors=3, metric='euclidean', p=2):
        """
        Initialize the KNN model.

        Parameters:
        - k_neighbors: Number of neighbors to consider.
        - metric: Distance metric ('euclidean', 'manhattan', 'minkowski').
        """
        self.k_neighbors = k_neighbors
        self.metric = metric
        self.p = p
        self.x_train = None
        self.y_train = None

    def fit(self, x, y):
        self.x_train = np.array(x)
        self.y_train = np.array(y)

    def compute_distances(self, x):
        if self.metric == 'euclidean':
            distances = np.sqrt(((x[:, np.newaxis] - self.x_train) ** 2).sum(axis=2))
        elif self.metric == 'manhattan':
            distances = np.abs(x[:, np.newaxis] - self.x_train).sum(axis=2)
        elif self.metric == 'minkowski':
            distances = (((np.abs(x[:, np.newaxis] - self.x_train)) ** self.p).sum(axis=2)) ** (1 / self.p)
        return distances

    def predict(self, x):
        """Predict the class labels for the provided data."""
        x = np.array(x)
        distances = self.compute_distances(x)

        neighbors = np.argsort(distances, axis=1)[:, :self.k_neighbors]
        predictions = []

        for neighbor_indices in neighbors:
            neighbor_labels = self.y_train[neighbor_indices]
            predictions.append(np.bincount(neighbor_labels).argmax())

        return np.array(predictions)

    def save_model(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    def load_model(filename):
        with open(filename, 'rb') as file:
            model = pickle.load(file)
        return model