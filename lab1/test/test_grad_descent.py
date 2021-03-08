import unittest
from typing import List, Tuple

from parameterized import parameterized
import numpy as np
from lab1.src.grad_descent import gradient_descent
import lab1.src.grad_step_strategy as st

DELTA = 1e-3

# function, gradient, start, expected result
testcases = [(lambda x: (x[0] - 3) ** 4 + 1, lambda x: np.array([4 * (x[0] - 3) ** 3]), [0], [3]),
             (lambda x: x[0] ** 2 + x[1] ** 2, lambda x: np.array([2 * x[0], 2 * x[1]]), [4, 3], [0, 0]),
             (lambda x: (x[0] - 2) ** 2 + (x[1] + 3) ** 2, lambda x: np.array([2 * (x[0] - 2), 2 * (x[1] + 3)]), [6, 7],
              [2, -3])]


# (lambda x: -1 / (2 * x[0] ** 2 + x[1] ** 2 + 7),
#               lambda x: np.array([4 * x[0] / (2 * x[0] ** 2 + x[1] ** 2 + 7) ** 2]), [-0.1, 0.1], [0, 0])


class TestGradDescent(unittest.TestCase):

    @parameterized.expand(testcases)
    def test_grad_descent_divide_strategy(self, foo, foo_grad, start, expected_result):
        result = gradient_descent(foo, foo_grad, start, step_strategy=st.StepStrategy.DIVIDE_STEP)
        self.assertEqual(len(result), len(expected_result))
        for result_x_k, expected_result_x_k in list(zip(result, expected_result)):
            self.assertAlmostEqual(result_x_k, expected_result_x_k, delta=DELTA)

    @parameterized.expand(testcases)
    def test_grad_descent_constant_strategy(self, foo, foo_grad, start, expected_result):
        result = gradient_descent(foo, foo_grad, start, step_strategy=st.StepStrategy.CONSTANT_STEP)
        self.assertEqual(len(result), len(expected_result))
        for result_x_k, expected_result_x_k in list(zip(result, expected_result)):
            self.assertAlmostEqual(result_x_k, expected_result_x_k, delta=DELTA)
