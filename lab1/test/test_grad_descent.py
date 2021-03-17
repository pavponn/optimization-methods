import unittest

from parameterized import parameterized
import numpy as np

from lab1.src.grad_step_strategy import StepStrategy
from lab1.src.stop_criteria import StopCriteria
from lab1.src.grad_descent import gradient_descent


DELTA = 1e-3

# function, gradient, start, expected result
TEST_CASES = [
    (
        lambda x: (x[0] - 3) ** 4 + 1,
        lambda x: np.array([4 * (x[0] - 3) ** 3]),
        [0],
        [3]
    ),
    (
        lambda x: x[0] ** 2 + x[1] ** 2,
        lambda x: np.array([2 * x[0], 2 * x[1]]),
        [4, 3],
        [0, 0]
    ),
    (
        lambda x: (x[0] - 2) ** 2 + (x[1] + 3) ** 2,
        lambda x: np.array([2 * (x[0] - 2), 2 * (x[1] + 3)]),
        [6, 7],
        [2, -3]
    ),
]

# TODO: test on this function
# (lambda x: -1 / (2 * x[0] ** 2 + x[1] ** 2 + 7),
#               lambda x: np.array([4 * x[0] / (2 * x[0] ** 2 + x[1] ** 2 + 7) ** 2]), [-0.1, 0.1], [0, 0])

STOP_CRITERIA = [StopCriteria.BY_GRAD, StopCriteria.BY_ARGUMENT]

TEST_CASES = [x + (c,) for x in TEST_CASES for c in STOP_CRITERIA]


class TestGradDescent(unittest.TestCase):

    @parameterized.expand(TEST_CASES)
    def test_grad_descent_divide_strategy(self, f, f_grad, start, expected_result, stop_criteria):
        self.run_test(f, f_grad, start, expected_result, StepStrategy.DIVIDE_STEP, stop_criteria)

    @parameterized.expand(TEST_CASES)
    def test_grad_descent_constant_strategy(self, f, f_grad, start, expected_result, stop_criteria):
        self.run_test(f, f_grad, start, expected_result, StepStrategy.CONSTANT_STEP, stop_criteria)

    @parameterized.expand(TEST_CASES)
    def test_grad_descent_dichotomy_strategy(self, f, f_grad, start, expected_result, stop_criteria):
        self.run_test(f, f_grad, start, expected_result, StepStrategy.DICHOTOMY_STEP, stop_criteria)

    @parameterized.expand(TEST_CASES)
    def test_grad_descent_golden_selection_strategy(self, f, f_grad, start, expected_result, stop_criteria):
        self.run_test(f, f_grad, start, expected_result, StepStrategy.GOLDEN_SELECTION_STEP, stop_criteria)

    @parameterized.expand(TEST_CASES)
    def test_grad_descent_fibonacci_strategy(self, f, f_grad, start, expected_result, stop_criteria):
        self.run_test(f, f_grad, start, expected_result, StepStrategy.FIBONACCI_STEP, stop_criteria)

    def run_test(self, f, f_grad, start, expected_result, step_strategy, stop_criteria):
        result = gradient_descent(f, f_grad, start, step_strategy=step_strategy, stop_criteria=stop_criteria)

        self.assertEqual(len(result), len(expected_result))
        for result_x_k, expected_result_x_k in list(zip(result, expected_result)):
            self.assertAlmostEqual(result_x_k, expected_result_x_k, delta=DELTA)
