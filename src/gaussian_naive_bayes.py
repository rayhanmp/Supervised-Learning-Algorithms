import numpy as np

# Implementation of gaussian naive bayes classifier from scratch
# @author: Rayhan Maheswara Pramanda
# @date: 2024-12-02

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
        self.classes = np.unique(y) # Get unique class labels

        for c in self.classes: 
            x_c = x[y == c] # Get data points for the current class
            self.mean[c] = x_c.mean(axis=0)
            self.std[c] = np.maximum(x_c.std(axis=0), self.var_smoothing) # If the standard deviation is lower than the smoothing value (could be zero), use the smoothing value instead
            self.priors[c] = len(x_c) / len(x)

    def calculate_likelihood(self, x, mean, std):
        '''
        Calculate likelihood of the data given the mean and standard deviation using Gaussian Probability Density Function

        Parameters:
            x -> Data point (features)
            mean -> Mean for each feature
            std -> Standard deviation for each feature
        '''
        return (1/(std*np.sqrt(2*np.pi)))*np.exp(-((x-mean)**2)/(2*std**2))

    def calculate_posterior(self, x):
        '''
        Calculate posterior probabilities for a single data point

        Parameters:
            x -> Data point
        '''
        posteriors = {}

        for c in self.classes: 
            likelihood = np.prod(self.calculate_likelihood(x, self.mean[c], self.std[c])) # Compute the joint likelihood P(x | c)
            posteriors[c] = self.priors[c] * likelihood

        return posteriors

    def predict(self, x):
        '''
        Predict class labels for the input data

        Parameters:
            x -> Test data
        '''
        y_pred = [] 

        for item in x:
            posteriors = self.calculate_posterior(item)
            y_pred.append(max(posteriors, key=posteriors.get)) # Get the class with the highest posterior probability

        return y_pred

