#!/usr/bin/env python3
import unittest
import random
import sys
import copy
import argparse
import inspect
import collections
import os
import pickle
import gzip
from graderUtil import graded, CourseTestRunner, GradedTestCase
import numpy as np
from itertools import product
from utils.data_preprocessing import dose_class, load_data, LABEL_KEY

# Import student submission
from submission import FixedDosePolicy
from submission import ClinicalDosingPolicy
from submission import LinUCB
from submission import eGreedyLinB
from submission import ThomSampB

# Import reference solution
if os.path.exists("./solution.py"):
    from solution import FixedDosePolicy as RefFixedDosePolicy
    from solution import ClinicalDosingPolicy as RefClinicalDosingPolicy
    from solution import LinUCB as RefLinUCB
    from solution import eGreedyLinB as RefeGreedyLinB
    from solution import ThomSampB as RefThomSampB
else:
    RefFixedDosePolicy = FixedDosePolicy
    RefClinicalDosingPolicy = ClinicalDosingPolicy
    RefLinUCB = LinUCB
    RefeGreedyLinB = eGreedyLinB
    RefThomSampB = ThomSampB


#########
# TESTS #
#########


class Test_1a(GradedTestCase):
    @graded(timeout=2, is_hidden=False)
    def test_0(self):
        """1a-0-basic: test for fixed prediction"""
        data = load_data()
        learner = FixedDosePolicy()
        prediction = learner.choose(dict(data.iloc[0]))
        predictions = []
        for t in range(10):
            x = dict(data.iloc[t])
            action = learner.choose(x)
            predictions.append(action)

        self.assertEqual([prediction], np.unique(predictions))

    @graded(timeout=2, is_hidden=False)
    def test_1(self):
        """1a-1-basic: evaluate the performance of clincal model on a single example"""
        data = load_data()
        learner = ClinicalDosingPolicy()
        x = dict(data.iloc[0])
        label = x.pop(LABEL_KEY)
        action = learner.choose(x)

        self.assertEqual(action, dose_class(label))

    ### BEGIN_HIDE ###
    ### END_HIDE ###


class Test_1b(GradedTestCase):
    @graded(timeout=2, is_hidden=False)
    def test_0(self):
        """1b-0-basic: basic test for correct initialization"""
        data = load_data()
        x = dict(data.iloc[0])
        learner = LinUCB(3, x.keys(), alpha=1)

        self.assertEqual(learner.features, x.keys())
        self.assertEqual(learner.d, len(x.keys()))
        self.assertTrue(np.array_equal(learner.A[0], np.eye(learner.d)))
        self.assertTrue(np.array_equal(learner.b[0], np.zeros(learner.d)))

    @graded(timeout=2, is_hidden=False)
    def test_1(self):
        """1b-1-basic: evaluate the choose function of LinUCB on a single example"""
        features = [
            "Age in decades",
            "Height (cm)",
            "Weight (kg)",
            "Male",
            "Female",
            "Asian",
            "Black",
            "White",
            "Unknown race",
            "Carbamazepine (Tegretol)",
            "Phenytoin (Dilantin)",
            "Rifampin or Rifampicin",
            "Amiodarone (Cordarone)",
            "VKORC1AG",
            "VKORC1AA",
            "VKORC1UN",
            "CYP2C912",
            "CYP2C913",
            "CYP2C922",
            "CYP2C923",
            "CYP2C933",
            "CYP2C9UN",
        ]
        data = load_data()
        x = dict(data.iloc[0])
        learner = LinUCB(3, features, alpha=1)
        prediction_class = "low"
        action = learner.choose(x)

        self.assertEqual(action, prediction_class)

    ### BEGIN_HIDE ###
    ### END_HIDE ###


class Test_1c(GradedTestCase):
    @graded(timeout=10, is_hidden=False)
    def test_0(self):
        """1c-0-basic: evaluate the choose function of egreedy disjoint linear UCB on single example"""
        features = [
            "Age in decades",
            "Height (cm)",
            "Weight (kg)",
            "Male",
            "Female",
            "Asian",
            "Black",
            "White",
            "Unknown race",
            "Carbamazepine (Tegretol)",
            "Phenytoin (Dilantin)",
            "Rifampin or Rifampicin",
            "Amiodarone (Cordarone)",
            "VKORC1AG",
            "VKORC1AA",
            "VKORC1UN",
            "CYP2C912",
            "CYP2C913",
            "CYP2C922",
            "CYP2C923",
            "CYP2C933",
            "CYP2C9UN",
        ]
        data = load_data()
        x = dict(data.iloc[0])
        learner = eGreedyLinB(3, features, alpha=1)

        np.random.seed(0)
        random_predictions = []
        for _ in range(1000):
            learner.time = 0
            random_predictions.append(learner.choose(x))

        np.random.seed(0)
        egreedy_predictions = []
        for _ in range(1000):
            learner.time = 1
            egreedy_predictions.append(learner.choose(x))

        np.random.seed(0)
        greedy_predictions = []
        for _ in range(1000):
            learner.time = 1e10
            greedy_predictions.append(learner.choose(x))

        random_low = collections.Counter(random_predictions)["low"] / 1000
        egreedy_low = collections.Counter(egreedy_predictions)["low"] / 1000
        greedy_low = collections.Counter(greedy_predictions)["low"] / 1000

        self.assertAlmostEqual(random_low, 0.333, delta=0.05)
        self.assertAlmostEqual(egreedy_low, 0.666, delta=0.05)
        self.assertAlmostEqual(greedy_low, 1.0, delta=0.0001)

    ### BEGIN_HIDE ###
    ### END_HIDE ###


class Test_1d(GradedTestCase):
    @graded(timeout=2, is_hidden=False)
    def test_0(self):
        """1d-0-basic: basic test for correct initialization"""
        data = load_data()
        x = dict(data.iloc[0])
        learner = ThomSampB(3, x.keys(), alpha=0.001)

        self.assertEqual(learner.features, x.keys())
        self.assertTrue(np.array_equal(learner.B[0], np.eye(learner.d)))
        self.assertTrue(np.array_equal(learner.mu[0], np.zeros((learner.d))))
        self.assertTrue(np.array_equal(learner.f[0], np.zeros((learner.d))))

    @graded(timeout=5, is_hidden=False)
    def test_1(self):
        """1d-1-basic: basic evaluation of the choose function of Thompson Sampling on a single example"""
        features = [
            "Age in decades",
            "Height (cm)",
            "Weight (kg)",
            "Male",
            "Female",
            "Asian",
            "Black",
            "White",
            "Unknown race",
            "Carbamazepine (Tegretol)",
            "Phenytoin (Dilantin)",
            "Rifampin or Rifampicin",
            "Amiodarone (Cordarone)",
            "VKORC1AG",
            "VKORC1AA",
            "VKORC1UN",
            "CYP2C912",
            "CYP2C913",
            "CYP2C922",
            "CYP2C923",
            "CYP2C933",
            "CYP2C9UN",
        ]
        data = load_data()
        learner = ThomSampB(3, features, alpha=0.001)
        prediction = "medium"
        x = dict(data.iloc[0])
        predictions = []
        np.random.seed(0)
        for _ in range(1000):
            predictions.append(learner.choose(x))

        low = collections.Counter(predictions)["low"] / 1000
        self.assertAlmostEqual(low, 0.333, delta=0.05)

    ### BEGIN_HIDE ###
    ### END_HIDE ###


def getTestCaseForTestID(test_id):
    question, part, _ = test_id.split("-")
    g = globals().copy()
    for name, obj in g.items():
        if inspect.isclass(obj) and name == ("Test_" + question):
            return obj("test_" + part)


if __name__ == "__main__":
    # Parse for a specific test
    parser = argparse.ArgumentParser()
    parser.add_argument("test_case", nargs="?", default="all")
    test_id = parser.parse_args().test_case

    assignment = unittest.TestSuite()
    if test_id != "all":
        assignment.addTest(getTestCaseForTestID(test_id))
    else:
        assignment.addTests(
            unittest.defaultTestLoader.discover(".", pattern="grader.py")
        )
    CourseTestRunner().run(assignment)
