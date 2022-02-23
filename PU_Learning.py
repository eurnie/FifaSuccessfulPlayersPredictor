import numpy as np
import pandas as pd
import random
import math
from abc import ABC, abstractmethod
from typing import Tuple
from sklearn.base import BaseEstimator
from sklearn.metrics import f1_score, mean_squared_error as mse
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

class TwoStepTechnique(BaseEstimator, ABC):
    
    def __init__(self):
        self.classifier=None
    
    @abstractmethod
    def step1(self, X, s) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        
        pass
    
    @abstractmethod
    def step2(self, X, n, p)-> BaseEstimator:

        pass
    
    def fit(self, X, s):
        result_step_1 = self.step1(X, s)
        self.step2(X, result_step_1[0], result_step_1[1], result_step_1[2])
        return self
    
    def predict(self,X):
        prediction = self.final_classifier.predict(X)
        return prediction
    
    def predict_proba(self, X):
        probability = self.final_classifier.predict_proba(X)
        return probability

class SEM(TwoStepTechnique):
    
    def __init__(self,
                 tol = 1.0e-10,
                 max_iter = 100,
                 spy_prop = 0.1,
                 l = 0.15,
                 classifier = LogisticRegression(),
                 seed=331
                ):
        
        super().__init__()
        
        # Instantiate the parameters
        self.tol = tol
        self.max_iter = max_iter
        self.spy_prop = spy_prop
        self.l = l
        self.classifier = classifier
        self.seed = seed
        self.nb_spies = 0
        self.final_classifier = classifier
        
    def step1(self, X, s) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        np.random.seed(self.seed)

        # Split the dataset into P (Positive) and M (Mix of positives and negatives)
        P = []
        M = []

        for x_element, y_element in zip(X, s):
            if (y_element == 1):
                P.append(x_element)
            else:
                M.append(x_element)

        P = np.array(P)
        M = np.array(M)
        
        # Select (randomly) the spies S
        random_index_list = []
        S = []
        self.nb_spies = round(self.spy_prop * P.shape[0])
        for i in range(1, self.nb_spies):
            no_new_number = True
            while (no_new_number):
                random_index = random.randint(0, P.shape[0]-1)
                if (not random_index in random_index_list):
                    random_index_list.append(random_index)
                    S.append(P[random_index])
                    no_new_number = False
        S = np.array(S)
        
        # Update P and MS
        MS = np.concatenate((M, S), axis=0)
        count = 0
        P_no_spies = P
        for index in random_index_list:
            P_no_spies = np.delete(P_no_spies, index - count, 0)
            count += 1
        
        ### I-EM Algorithm

        # Train the classifier using P and MS:
        training_set = np.concatenate((P_no_spies, MS), axis=0)
        labels = np.array(([1] * P_no_spies.shape[0]) + ([0] * MS.shape[0]))
        self.classifier.fit(training_set, labels)
        
        # Save the model's score ''score_variation'' using model.score function
        score = self.classifier.score(training_set, labels)
        
        # Initialize iterations to 0 and the score variation
        score_variation = score
        old_score = score
        training_set = np.concatenate((training_set, MS), axis=0)
        n_iter = 0
        
        # Loop while classifier parameters change, i.e. until the score variation is >= tolerance
        while score_variation >= self.tol and n_iter < self.max_iter:
            
            # Expectation step
            probabilities = self.classifier.predict_proba(MS)
            probabilities_MS_pos = []
            probabilities_MS_neg = []
            for prob in probabilities:
                probabilities_MS_pos.append(prob[0])
                probabilities_MS_neg.append(prob[1])

            # Create the new training set with the probabilistic labels (weights)
            training_set_probabilities = ([1] * P_no_spies.shape[0]) + probabilities_MS_pos + probabilities_MS_neg
            labels = np.array(([1] * (P_no_spies.shape[0] + MS.shape[0])) + ([0] * MS.shape[0]))
            
            # Maximization step
            self.classifier.fit(training_set, labels, training_set_probabilities)
            
            # Update score variation and the old score
            score = self.classifier.score(training_set, labels)
            score_variation = abs(score - old_score)
            old_score = score
            n_iter += 1
            
        # Print the number of iterations as sanity check
        print("Number of iterations first step:", n_iter)
        
        # Select the threshold t such that l% of spies' probabilities to be positive is belot t
        spies_probabilities_sorted = probabilities_MS_pos[len(probabilities_MS_pos)-self.nb_spies-1:]
        spies_probabilities_sorted.sort()
        number = round(self.l * len(spies_probabilities_sorted))
        threshold_t = spies_probabilities_sorted[number]
        
        # Create N and U
        N = []
        U = []

        for element, proba in zip(MS[:-self.nb_spies], probabilities_MS_pos[:-self.nb_spies]):
            if (proba < threshold_t):
                N.append(element)
            else:
                U.append(element)

        N = np.array(N)
        U = np.array(U)
        
        # Return P, N, U
        return P, N, U
        
    def step2(self, X, P, N, U)->BaseEstimator:
        np.random.seed(self.seed)
        
        # Assign every document in the positive set pos the fixed class label 1
        labels_P = [1] * P.shape[0]
        
        # Assign every document in the likely negative set N the initial class label 0
        labels_N = [0] * N.shape[0]
        
        ### I-EM Algorithm

        # Train classifier using N and P:
        training_set = np.concatenate((P, N), axis=0)
        labels = np.array(labels_P + labels_N)
        self.classifier.fit(training_set, labels)

        # Compute the metrics for classifier f_i in delta_i to select the best classifier
        self.final_classifier = self.classifier

        # Initialize iterations to 0, the score variation, and whether the best classifier has been selected or not.
        score = self.classifier.score(training_set, labels)
        score_variation = score
        old_score = score
        score_final_classifier = score
        training_set = np.concatenate((training_set, U, N, U), axis=0)
        n_iter = 0
        
        # Loop until the variation is > than the tolerance
        while score_variation >= self.tol and n_iter < self.max_iter:

            # Update probabilities
            probabilities = self.classifier.predict_proba(np.concatenate((N, U), axis=0))
            probabilities_pos = []
            probabilities_neg = []
            for prob in probabilities:
                probabilities_pos.append(prob[0])
                probabilities_neg.append(prob[1])

            # Create the new training set with the probabilistic labels (weights)
            training_set_probabilities = ([1] * P.shape[0]) + probabilities_pos + probabilities_neg
            labels = np.array(([1] * (P.shape[0] + N.shape[0] + U.shape[0])) + ([0] * (N.shape[0] + U.shape[0])))
            
            # Maximization step
            self.classifier = LogisticRegression()
            self.classifier.fit(training_set, labels, training_set_probabilities)
            
            # Update parameter variation
            score = self.classifier.score(training_set, labels)
            score_variation = abs(score - old_score)
            old_score = score
            n_iter += 1

            # Select the best classifier classifier: (final_classifier)
            if (score > score_final_classifier):
                self.final_classifier = self.classifier
                score_final_classifier = score

        print("Number of iterations second step:", n_iter)
        
        return self.final_classifier

#--------------------#--------------------#--------------------#--------------------    
#-------------------- Second PU Learning Method #--------------------
    
class ModifiedLogisticRegression(BaseEstimator):

    def __init__(self,
                 tol = 1.0e-10,
                 max_iter = 100,
                 l_rate = 0.001,
                 c = 0,
                 seed = 331):
        
        # Instantiate the parameters
        self.tol = tol
        self.max_iter = max_iter
        self.l_rate = l_rate
        self.c = c
        self.seed = seed
        self.b = 1
        self.w = []
        
    def log_likelihood(self, x, y):
        # If you use the gradient ascent technique, fill in this part with the log_likelihood function
        # If you prefer to use a different technique, you can leave this empty
        count = 0
        for x_element, y_element in zip(x, y):
            dot_product = 0
            for x_value, w_value in zip(x_element, self.w):
                dot_product += x_value * w_value
            temp1 = math.log(1 / (1 + math.pow(self.b, 2) + math.exp(-1 * dot_product)))
            temp2 = math.log(1 - (1 / (1 + math.pow(self.b, 2) + math.exp(-1 * dot_product))))
            count += y_element * temp1
            count += (1 - y_element) * temp2
        return count
        
    def parameters_update(self, x, y):
        # If you use the gradient ascent technique, fill in this part with the parameter update (both w and b)
        # If you prefer to use a different technique, you can leave this empty
        gradient_w = 0
        gradient_b = 0

        if (len(self.w) != x.shape[1]):
            self.w = [1] * x.shape[1]

        for x_element, y_element in zip(x, y):
            dot_product = 0
            for x_value, w_value in zip(x_element, self.w):
                dot_product += x_value * w_value
            b_square = math.pow(self.b, 2)
            factor = math.exp(-1 * dot_product)

            temp_w_1 = y_element / (b_square + factor)
            temp_w_2 = 1 / ((1 + b_square + factor) * (b_square + factor))
            temp_w = temp_w_1 - temp_w_2
            gradient_w += x_element * factor * temp_w

            temp_b_1 = 1 - (y_element * (1 + b_square + factor))
            temp_b_2 = (1 + b_square + factor) * (b_square + factor)
            gradient_b += 2 * self.b * (temp_b_1 / temp_b_2)

        self.w += self.l_rate * gradient_w
        self.b += self.l_rate * gradient_b

    def fit(self, X, s):
        np.random.seed(self.seed)
        
        # Initialize w and b
        self.parameters_update(X, s)

        # Initialize the score (log_likelihood), the number of iterations and the score variation.
        old_score = self.log_likelihood(X, s)
        n_iter = 0
        score_variation = 1
        
        # Loop until the score variation is lower than tolerance or max_iter is reached
        while score_variation >= self.tol and n_iter < self.max_iter:
            
            # Update the parameters
            self.parameters_update(X, s)

            # Compute log_likelihood
            new_score = self.log_likelihood(X, s)
            
            # Update scores
            score_variation = new_score - old_score
            old_score = new_score
            n_iter += 1

        return self    
    
    def estimate_c(self):
        # Estimate the parameter c from b
        return (1 / (1 + math.pow(self.b, 2)))
    
    def predict(self,X):
        return [round(element) for element in self.predict_proba(X)]
    
    def predict_proba(self, X):
        probability = []
        c = self.estimate_c()
        for element in X:
            dot_product = 0
            for x_value, w_value in zip(element, self.w):
                dot_product += x_value * w_value
            b_square = math.pow(self.b, 2)
            factor = math.exp(-1 * dot_product)
            probability.append((1 / (1 + b_square + factor)) / c)
        return probability