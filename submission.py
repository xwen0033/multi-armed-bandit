import numpy as np
import csv
import os

from abc import ABC, abstractmethod
from utils.data_preprocessing import load_data, dose_class, LABEL_KEY


# Base classes
class BanditPolicy(ABC):
    @abstractmethod
    def choose(self, x):
        pass

    @abstractmethod
    def update(self, x, a, r):
        pass


class StaticPolicy(BanditPolicy):
    def update(self, x, a, r):
        pass


class RandomPolicy(StaticPolicy):
    def __init__(self, probs=None):
        self.probs = probs if probs is not None else [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]

    def choose(self, x):
        return np.random.choice(("low", "medium", "high"), p=self.probs)


############################################################
# Problem 1: Estimation of Warfarin Dose
############################################################

############################################################
# Problem 1a: baselines


class FixedDosePolicy(StaticPolicy):
    def choose(self, x):
        """
        Args:
                x (dict): dictionary containing the possible patient features.

        Returns:
                output (str): string containing one of ('low', 'medium', 'high')

        TODO:
                - Please implement the fixed dose which is to assign medium dose
                  to all patients.
        """
        ### START CODE HERE ###
        return 'medium'
        ### END CODE HERE ###


class ClinicalDosingPolicy(StaticPolicy):
    def choose(self, x):
        """
        Args:
                x (dict): Dictionary containing the possible patient features.

        Returns:
                output (str): string containing one of ('low', 'medium', 'high')

        TODO:
                - Prepare the features to be used in the clinical model
                  (consult section 1f of appx.pdf for feature definitions)
                - Create a linear model based on the values in section 1f
                  and return its output based on the input features

        Hint:
                - Look at the utils/data_preprocessing.py script to see the key values
                  of the features you can use. The age in decades is implemented for
                  you as an example.
                - You can treat Unknown race as missing or mixed race.
                - Use dose_class() implemented for you.
        """
        age_in_decades = x["Age in decades"]

        ### START CODE HERE ###
        asian = x['Asian']
        black = x['Black']
        unknown = x['Unknown race']
        height = x["Height (cm)"]
        weight = x["Weight (kg)"]
        amiodarone = x["Amiodarone (Cordarone)"]
        carbamazepine = x["Carbamazepine (Tegretol)"]
        phenytoin = x["Phenytoin (Dilantin)"]
        rifampin = x["Rifampin or Rifampicin"]
        enzyme = carbamazepine or phenytoin or rifampin

        output = 4.0376 - 0.2546 * age_in_decades + 0.0118 * height + \
                 0.0134 * weight - 0.6752 * asian + 0.4060 * black + \
                 0.0443 * unknown - 0.5695 * amiodarone + 1.2799 * enzyme
        dose = output**2

        if dose < 21:
            return 'low'
        elif dose < 49:
            return 'medium'
        else:
            return 'high'

        ### END CODE HERE ###


############################################################
# Problem 1b: upper confidence bound linear bandit


class LinUCB(BanditPolicy):
    def __init__(self, n_arms, features, alpha=1.0):
        """
        See Algorithm 1 from paper:
                "A Contextual-Bandit Approach to Personalized News Article Recommendation"

        Args:
                n_arms (int): the number of different arms/ actions the algorithm can take
                features (list of str): contains the patient features to use
                alpha (float): hyperparameter for step size.

        TODO:
                - Please initialize the following internal variables for the Disjoint Linear UCB Bandit algorithm:
                        * self.n_arms
                        * self.features
                        * self.d
                        * self.alpha
                        * self.A
                        * self.b
                  These terms align with the paper, please refer to the paper to understand what they are.
                  Feel free to add additional internal variables if you need them, but they are not necessary.

        Hint:
                Keep track of a seperate A, b for each action (this is what the Disjoint in the algorithm name means)
        """
        ### START CODE HERE ###
        self.n_arms = n_arms
        self.features = features
        self.d = len(features)
        self.alpha = alpha
        self.A = [np.identity(self.d) for _ in range(self.n_arms)]
        self.b = [np.zeros(self.d) for _ in range(self.n_arms)]
        ### END CODE HERE ###

    def choose(self, x):
        """
        See Algorithm 1 from paper:
                "A Contextual-Bandit Approach to Personalized News Article Recommendation"

        Args:
                x (dict): Dictionary containing the possible patient features.
        Returns:
                output (str): string containing one of ('low', 'medium', 'high')

        TODO:
                - Please implement the "forward pass" for Disjoint Linear Upper Confidence Bound Bandit algorithm.
        """
        xvec = np.array([x[f] for f in self.features])
        ### START CODE HERE ###
        p = np.zeros(self.n_arms)

        for a in range(self.n_arms):
            theta = np.linalg.inv(self.A[a]).dot(self.b[a])
            p[a] = theta.T.dot(xvec) + self.alpha * np.sqrt(xvec.T.dot(np.linalg.inv(self.A[a])).dot(xvec))

        action = np.argmax(p)
        output = ['low', 'medium', 'high']
        return output[action]
        ### END CODE HERE ###

    def update(self, x, a, r):
        """
        See Algorithm 1 from paper:
                "A Contextual-Bandit Approach to Personalized News Article Recommendation"

        Args:
                x (dict): Dictionary containing the possible patient features.
                a (str): string, indicating the action your algorithem chose ('low', 'medium', 'high')
                r (int): the reward you recieved for that action

        TODO:
                - Please implement the update step for Disjoint Linear Upper Confidence Bound Bandit algorithm.

        Hint:
                Which parameters should you update?
        """
        xvec = np.array([x[f] for f in self.features])
        ### START CODE HERE ###
        action = 2
        if a == 'low':
            action = 0
        if a == 'medium':
            action = 1
        self.A[action] += np.outer(xvec, xvec)
        self.b[action] += r * xvec
        ### END CODE HERE ###


############################################################
# Problem 1c: eGreedy linear bandit


class eGreedyLinB(LinUCB):
    def __init__(self, n_arms, features, alpha=1.0):
        super(eGreedyLinB, self).__init__(n_arms, features, alpha=1.0)
        self.time = 0

    def choose(self, x):
        """
        Args:
                x (dict): Dictionary containing the possible patient features.
        Returns:
                output (str): string containing one of ('low', 'medium', 'high')

        TODO:
                - Instead of using the Upper Confidence Bound to find which action to take,
                  compute the payoff of each action using a simple dot product between Theta & the input features.
                  Then use an epsilon greedy algorithm to choose the action.
                  Use the value of epsilon provided and np.random.uniform() in your implementation.
        """

        self.time += 1
        epsilon = float(1.0 / self.time) * self.alpha
        xvec = np.array([x[f] for f in self.features])
        ### START CODE HERE ###
        rewards = np.zeros(self.n_arms)
        for a in range(self.n_arms):
            sample = np.linalg.inv(self.A[a]).dot(self.b[a])
            rewards[a] = sample.T.dot(xvec)

        if np.random.uniform() < epsilon:
            action = np.random.choice(range(self.n_arms))
        else:
            action = np.argmax(rewards)

        output = ['low', 'medium', 'high']
        return output[action]
        ### END CODE HERE ###


############################################################
# Problem 1d: Thompson sampling


class ThomSampB(BanditPolicy):
    def __init__(self, n_arms, features, alpha=1.0):
        """
        See Algorithm 1 and section 2.2 from paper:
                "Thompson Sampling for Contextual Bandits with Linear Payoffs"

        Args:
                n_arms (int): the number of different arms/ actions the algorithm can take
                features (list of str): contains the patient features to use
                alpha (float): hyperparameter for step size.

        TODO:
                - Please initialize the following internal variables for the Thompson sampling bandit algorithm:
                        * self.n_arms
                        * self.features
                        * self.d
                        * self.v2 (please set this term equal to alpha)
                        * self.B
                        * self.mu
                        * self.f
                These terms align with the paper, please refer to the paper to understand what they are.
                Please feel free to add additional internal variables if you need them, but they are not necessary.

        Hints:
                - Keep track of a separate B, mu, f for each action (this is what the Disjoint in the algorithm name means)
                - Unlike in section 2.2 in the paper where they sample a single mu_tilde, we'll sample a mu_tilde for each arm
                        based on our saved B, f, and mu values for each arm. Also, when we update, we only update the B, f, and mu
                        values for the arm that we selected
                - What the paper refers to as b in our case is the medical features vector
                - The paper uses a summation (from time =0, .., t-1) to compute the model parameters at time step (t),
                        however if you can't access prior data how might one store the result from the prior time steps.

        """

        ### START CODE HERE ###
        self.n_arms = n_arms
        self.features = features
        self.d = len(features)
        self.v2 = alpha
        self.B = [np.identity(self.d) for _ in range(self.n_arms)]
        self.mu = [np.zeros(self.d)for _ in range(self.n_arms)]
        self.f = [np.zeros(self.d)for _ in range(self.n_arms)]
        ### END CODE HERE ###

    def choose(self, x):
        """
        See Algorithm 1 and section 2.2 from paper:
                "Thompson Sampling for Contextual Bandits with Linear Payoffs"

        Args:
                x (dict): Dictionary containing the possible patient features.
        Returns:
                output (str): string containing one of ('low', 'medium', 'high')

        TODO:
                - Please implement the "forward pass" for Disjoint Thompson Sampling Bandit algorithm.
                - Please use np.random.multivariate_normal to simulate the multivariate gaussian distribution in the paper.
        """
        xvec = np.array([x[f] for f in self.features])
        ### START CODE HERE ###
        rewards = np.zeros(self.n_arms)
        for a in range(self.n_arms):
            sample = np.random.multivariate_normal(self.mu[a], self.v2*np.linalg.inv(self.B[a]))
            rewards[a] = np.dot(xvec.T, sample)

        action = np.argmax(rewards)
        output = ['low', 'medium', 'high']
        return output[action]
        ### END CODE HERE ###

    def update(self, x, a, r):
        """
        See Algorithm 1 and section 2.2 from paper:
                "Thompson Sampling for Contextual Bandits with Linear Payoffs"

        Args:
                x (dict): Dictionary containing the possible patient features.
                a (str): string, indicating the action your algorithem chose ('low', 'medium', 'high')
                r (int): the reward you recieved for that action

        TODO:
                - Please implement the update step for Disjoint Thompson Sampling Bandit algorithm.

        Hint:
                Which parameters should you update?
        """
        xvec = np.array([x[f] for f in self.features])
        ### START CODE HERE ###
        action = 2
        if a == 'low':
            action = 0
        if a == 'medium':
            action = 1
        self.B[action] += np.outer(xvec, xvec)
        self.f[action] += r * xvec
        self.mu[action] = np.linalg.inv(self.B[action]).dot(self.f[action])
        ### END CODE HERE ###
