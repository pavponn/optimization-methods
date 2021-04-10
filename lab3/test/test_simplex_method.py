import unittest
import scipy.optimize as opt
import numpy as np
from parameterized import parameterized
from lab3.src.methods.simplex_method import simplex_method

simplex_method_testcases = [
    (
        np.array([[1, 2, -1, 2, 4],
                  [0, -1, 2, 1, 3],
                  [1, -3, 2, 2, 0]]),
        np.array([1, 3, 4]),
        np.array([1, -3, 2, 1, 4]),
    ), (
        np.array([[1, 3, 0, 2, 1],
                  [2, -1, 1, 2, 3],
                  [1, -1, 2, 1, 0]]),
        np.array([1, 2, 4]),
        np.array([-1, -3, 2, 1, 4]),
    ), (
        np.array([[-1, 3, 0, 2, 1],
                  [2, -1, 1, 2, 3],
                  [1, -1, 2, 1, 0]]),
        np.array([1, 4, 5]),
        np.array([-1, 0, -2, 5, 4]),
    ), (
        np.array([[2, 3, 1, 2, 1],
                  [2, 1, -3, 2, 1],
                  [2, 1, 2, 1, 0]]),
        np.array([1, 3, 1]),
        np.array([-1, 1, -2, 1, 5]),
    ), (
        np.array([[2, 1, 3, 4],
                  [1, -1, 2, 1],
                  [0, 0, 1, 3]]),
        np.array([2, 4, 1]),
        np.array([-2, 3, 4, -1]),
    ), (
        np.array([[2, 3, 1, 2],
                  [2, -1, 2, 1],
                  [1, 1, 0, -1]]),
        np.array([3, 4, 1]),
        np.array([-2, 3, -3, 3]),
    ), (
        np.array([[2, 3, -1, 2],
                  [1, 1, 1, 1],
                  [2, -1, 0, 2]]),
        np.array([1, 1, 2]),
        np.array([-2, 3, 4, -1]),
    ), (
        np.array([[2, 1, 3, 4],
                  [2, -1, 2, 1],
                  [0, 0, 1, 2]]),
        np.array([1, 2, 4]),
        np.array([-2, 3, 4, -1]),
    ), (
        np.array([[1, 2, 3, 1, 2, 5],
                  [2, -3, 1, 2, 1, 4]]),
        np.array([1, 2]),
        np.array([-2, 3, 4, -1, 2, 1]),
    ), (
        np.array([[3, 2, 1, -3, 2, 1],
                  [1, 1, 0, 0, 1, 1]]),
        np.array([3, 2]),
        np.array([-2, 3, 1, 2, 0, 1]),
    ), (
        np.array([[1, 2, 3, 4, 5, 6],
                  [2, 1, -3, 2, 1, -3]]),
        np.array([1, 4]),
        np.array([1, -1, 2, 3, 1, 0]),
    ), (
        np.array([[2, 3, -1, 0, 2, 1],
                  [2, 0, 3, 0, 1, 1]]),
        np.array([1, 2]),
        np.array([-2, 3, 4, -1, 2, 1]),
    )
]

DELTA = 9


class TestSimplexMethod(unittest.TestCase):

    @parameterized.expand(simplex_method_testcases)
    def test_simplex_method(self, A, b, c):
        t = simplex_method(A, b, c)
        result = t if t is not None else (None, None)
        expected_result = opt.linprog(c=-c, A_eq=A, b_eq=b, method='simplex')
        self.check_result(result, expected_result)

    def check_result(self, result, expected_result):
        res, x = result
        if not expected_result.success:
            self.assertIsNone(x)
        else:
            np.testing.assert_array_almost_equal(res, -expected_result.fun, DELTA)
