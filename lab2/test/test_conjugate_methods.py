import unittest
import math
import numpy as np

from parameterized import parameterized

from lab2.src.methods.conjugate_method import (
    conjugate_direction_method,
    conjugate_gradient_method
)

conjugate_direction_method_testcases = [
    ([[2]], [0], [-6], [0]),
    ([[20]], [-100], [0], [5]),
    ([[4, 0], [0, 4]], [0, 0], [12, 13], [0, 0]),
    ([[4, 0], [0, 4]], [1, 0], [10, 10], [-0.25, 0])
]

conjugate_gradient_method_testcases = [
    (
        lambda x: (x[0] - 5) ** 2 + 1,
        lambda x: np.array([2 * (x[0] - 5)]),
        [0],
        [5]
    ),
    (
        lambda x: x[0] ** 2 + x[1] ** 2,
        lambda x: np.array([2 * x[0], 2 * x[1]]),
        [4, 3],
        [0, 0]
    ),
    (
        lambda x: 7 * (x[0] - 1) ** 2 + x[1] ** 2,
        lambda x: np.array([14 * (x[0] - 1), 2 * x[1]]),
        [400, -300],
        [1, 0]
    ),
    (
        lambda x: (x[0] - 2) ** 2 + (x[1] + 3) ** 2,
        lambda x: np.array([2 * (x[0] - 2), 2 * (x[1] + 3)]),
        [6, 7],
        [2, -3]
    ),
    (
        lambda x: -1 / (2 * x[0] ** 2 + x[1] ** 2 + 7),
        lambda x: np.array([
            4 * x[0] / (2 * x[0] ** 2 + x[1] ** 2 + 7) ** 2,
            2 * x[1] / (2 * x[0] ** 2 + x[1] ** 2 + 7) ** 2
        ]),
        [-0.1, 0.1],
        [0, 0]
    )
]

DELTA = 1e-3


class TestConjugateMethods(unittest.TestCase):

    @parameterized.expand(conjugate_direction_method_testcases)
    def test_conjugate_direction_method(self, Q, b, start, expected_result):
        result = conjugate_direction_method(Q=np.array(Q), b=np.array(b), start=np.array(start))
        self.check_result(result, expected_result)

    @parameterized.expand(conjugate_gradient_method_testcases)
    def test_conjugate_gradient_method(self, f, f_grad, start, expected_result):
        result = conjugate_gradient_method(f=f, f_grad=f_grad, start=np.array(start))
        self.check_result(result, expected_result)

    def check_result(self, result, expected_result):
        dist = 0.0
        for result_x_k, expected_result_x_k in list(zip(result, expected_result)):
            dist += (result_x_k - expected_result_x_k) ** 2
        dist = math.sqrt(dist)
        self.assertAlmostEqual(dist, 0, delta=DELTA)
