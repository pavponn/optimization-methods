import math
import unittest
import numpy as np
from parameterized import parameterized
from lab2.src.methods.newton_method import newton_method

newton_method_testcases = [
    (
        lambda x: x[0] ** 2 + x[1] ** 2,
        lambda x: np.array([2 * x[0], 2 * x[1]]),
        lambda x: np.array([[2, 0], [0, 2]]),
        [3, 2],
        [0, 0]
    ),
    (
        lambda x: 7 * (x[0] - 1) ** 2 + x[1] ** 2,
        lambda x: np.array([14 * (x[0] - 1), 2 * x[1]]),
        lambda x: np.array([[14, 0], [0, 2]]),
        [400, -300],
        [1, 0]
    ),
    (
        lambda x: (x[0] - 2) ** 2 + (x[1] + 3) ** 2,
        lambda x: np.array([2 * (x[0] - 2), 2 * (x[1] + 3)]),
        lambda x: np.array([[2, 0], [0, 2]]),
        [6, 7],
        [2, -3]
    ),
    (
        lambda x: x[0] ** 2 + x[1] ** 2,
        lambda x: np.array([2 * x[0], 2 * x[1]]),
        lambda x: np.array([[2, 0], [0, 2]]),
        [-0.1, 0.2],
        [0, 0]
    ),
    (
        lambda x: 7 * (x[0] - 1) ** 2 + x[1] ** 2,
        lambda x: np.array([14 * (x[0] - 1), 2 * x[1]]),
        lambda x: np.array([[14, 0], [0, 2]]),
        [413, -1210],
        [1, 0]
    ),
    (
        lambda x: (x[0] - 2) ** 2 + (x[1] + 3) ** 2,
        lambda x: np.array([2 * (x[0] - 2), 2 * (x[1] + 3)]),
        lambda x: np.array([[2, 0], [0, 2]]),
        [100, -3232],
        [2, -3]
    )
]

DELTA = 1e-3


class TestNewtonMethod(unittest.TestCase):

    @parameterized.expand(newton_method_testcases)
    def test_newton_method_with_inverse(self, f, f_grad, f_hess, start, expected_result):
        result = newton_method(f=f,
                               f_grad=f_grad,
                               f_hess=f_hess,
                               start=start,
                               use_conjugate=False)
        self.check_result(result, expected_result)

    @parameterized.expand(newton_method_testcases)
    def test_newton_method_with_conjugate_direction(self, f, f_grad, f_hess, start, expected_result):
        result = newton_method(f=f,
                               f_grad=f_grad,
                               f_hess=f_hess,
                               start=start,
                               use_conjugate=True)
        self.check_result(result, expected_result)

    def check_result(self, result, expected_result):
        dist = 0.0
        for result_x_k, expected_result_x_k in list(zip(result, expected_result)):
            dist += (result_x_k - expected_result_x_k) ** 2
        dist = math.sqrt(dist)
        self.assertAlmostEqual(dist, 0, delta=DELTA)
