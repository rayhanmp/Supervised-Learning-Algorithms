import numpy as np
import pickle

# Implementation of gaussian naive bayes classifier from scratch
# @author: Rayhan Maheswara Pramanda
# @date: 2024-12-07

class GaussianNaiveBayes:
    '''
    Gaussian Naive Bayes classifier implementation from scratch
    '''
    def __init__(self, var_smoothing = 1e-9):
        self.classes = []
        self.mean = {}
        self.std = {}
        self.priors = {}

        # Regularisation to avoid division by zero when calculating likelihood, this is different from Laplace Additive Smoothing
        # Variance can be 0 if the data points are identical for a certain class
        self.var_smoothing = var_smoothing

    def fit(self, x, y):
        '''
        Fit model to the training data

        Parameters:
            x -> Training data (features)
            y -> Target values (class labels)
        '''

        if len(x) != len(y): # Check if the number of samples in the features and target are the same
            raise ValueError('Length of x and y must be the same')

        self.classes = np.unique(y) # Get unique class labels

        for c in self.classes:
            x_c = x[y == c] # Get data points for the current class
            self.mean[c] = x_c.mean(axis=0)
            self.std[c] = np.maximum(x_c.std(axis=0), self.var_smoothing) # If the standard deviation is lower than the smoothing value (could be zero), use the smoothing value instead
            self.priors[c] = len(x_c) / len(x)

    def calculate_log_likelihood(self, x, mean, std):
        '''
        Calculate log likelihood of the data given the mean and standard deviation using Gaussian Probability Density Function
        Log likelihood is used to avoid underflow when multiplying small probabilities

        Parameters:
            x -> Data point (features)
            mean -> Mean for each feature
            std -> Standard deviation for each feature
        '''
        return (-0.5 * np.sum(np.log(2 * np.pi * std**2))) - np.sum(((x - mean)**2) / (2 * std**2))


    def calculate_log_posterior(self, x):
        '''
        Calculate posterior probabilities for a single data point

        Parameters:
            x -> Data point
        '''
        log_posteriors = {}

        for c in self.classes:
            log_likelihood = self.calculate_log_likelihood(x, self.mean[c], self.std[c]) # Compute the log likelihood P(x | c)
            log_posteriors[c] = np.log(self.priors[c]) + log_likelihood

        return log_posteriors

    def predict(self, x):
        '''
        Predict class labels for the input data

        Parameters:
            x -> Test data
        '''
        x = np.array(x, dtype=np.float64)  # Ensure x is a numpy array of floats
        y_pred = []

        for item in x:
            log_posteriors = self.calculate_log_posterior(item)
            y_pred.append(max(log_posteriors, key=log_posteriors.get)) # Get the class with the highest posterior probability

        return y_pred

    def save_model(self, filename):
        '''
        Save model to a file

        Parameters:
            filename -> Name of model file
        '''
        with open(filename, 'wb') as file:
            pickle.dump(self, file)
        print(f'Model saved as {filename}')

    def load_model(filename):
        '''
        Load model from a file

        Parameters:
            filename -> Name of model file
        '''
        with open(filename, 'rb') as file:
            model = pickle.load(file)
        print(f'Model {filename} has been loaded')

        return model