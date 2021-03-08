import unittest

from parameterized import parameterized
import numpy as np
from lab1.src.grad_descent import gradient_descent
import lab1.src.grad_step_strategy as st
from lab1.src.stop_criteria import StopCriteria

DELTA = 1e-3

# function, gradient, start, expected result
testcases = [(lambda x: (x[0] - 3) ** 4 + 1, lambda x: np.array([4 * (x[0] - 3) ** 3]), [0], [3]),
             (lambda x: x[0] ** 2 + x[1] ** 2, lambda x: np.array([2 * x[0], 2 * x[1]]), [4, 3], [0, 0]),
             (lambda x: (x[0] - 2) ** 2 + (x[1] + 3) ** 2, lambda x: np.array([2 * (x[0] - 2), 2 * (x[1] + 3)]), [6, 7],
              [2, -3])]

# TODO: check on this function
# (lambda x: -1 / (2 * x[0] ** 2 + x[1] ** 2 + 7),
#               lambda x: np.array([4 * x[0] / (2 * x[0] ** 2 + x[1] ** 2 + 7) ** 2]), [-0.1, 0.1], [0, 0])

stop_criterias = [StopCriteria.BY_GRAD, StopCriteria.BY_ARGUMENT]

testcases_with_stop_criterias = [(x[0], x[1], x[2], x[3], y) for x in testcases for y in stop_criterias]


class TestGradDescent(unittest.TestCase):

    @parameterized.expand(testcases_with_stop_criterias)
    def test_grad_descent_divide_strategy(self, foo, foo_grad, start, expected_result, stop_criteria):
        self.run_test(foo, foo_grad, start, expected_result, st.StepStrategy.DIVIDE_STEP, stop_criteria)

    @parameterized.expand(testcases_with_stop_criterias)
    def test_grad_descent_constant_strategy(self, foo, foo_grad, start, expected_result, stop_criteria):
        self.run_test(foo, foo_grad, start, expected_result, st.StepStrategy.CONSTANT_STEP, stop_criteria)

    @parameterized.expand(testcases_with_stop_criterias)
    def test_grad_descent_dichotomy_strategy(self, foo, foo_grad, start, expected_result, stop_criteria):
        self.run_test(foo, foo_grad, start, expected_result, st.StepStrategy.DICHOTOMY_STEP, stop_criteria)

    @parameterized.expand(testcases_with_stop_criterias)
    def test_grad_descent_golden_selection_strategy(self, foo, foo_grad, start, expected_result, stop_criteria):
        self.run_test(foo, foo_grad, start, expected_result, st.StepStrategy.GOLDEN_SELECTION_STEP, stop_criteria)

    @parameterized.expand(testcases_with_stop_criterias)
    def test_grad_descent_fibonacci_strategy(self, foo, foo_grad, start, expected_result, stop_criteria):
        self.run_test(foo, foo_grad, start, expected_result, st.StepStrategy.FIBONACCI_STEP, stop_criteria)

    def run_test(self, foo, foo_grad, start, expected_result, step_strategy, stop_criteria):
        result = gradient_descent(foo, foo_grad, start, step_strategy=step_strategy, stop_criteria=stop_criteria)
        self.assertEqual(len(result), len(expected_result))
        for result_x_k, expected_result_x_k in list(zip(result, expected_result)):
            self.assertAlmostEqual(result_x_k, expected_result_x_k, delta=DELTA)
