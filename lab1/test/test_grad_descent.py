import unittest
import numpy as np
import math
import logging

from parameterized import parameterized
from lab1.src.grad.grad_step_strategy import StepStrategy
from lab1.src.grad.stop_criteria import StopCriteria
from lab1.src.grad.grad_descent import gradient_descent

DELTA = 6.3e-5

EPSILON_STRATEGY = 1e-5
EPSILON_STOP_CRITERIA = 1e-10

DEFAULT_MAX_ITERS_STRATEGY = 40
DEFAULT_MAX_ITERS_CRITERIA = 100

# function, gradient, start, expected result
TEST_CASES = [
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

STOP_CRITERIA = [StopCriteria.BY_FUNCTION, StopCriteria.BY_GRAD, StopCriteria.BY_ARGUMENT]

TEST_CASES = [x + (c,) for x in TEST_CASES for c in STOP_CRITERIA]


class TestGradDescent(unittest.TestCase):
    max_ans_dist = 0
    min_ans_dist = 1e20

    def __init__(self, *args, **kwargs):
        logging.basicConfig(format='%(asctime)s    |    %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S%p',
                            level=logging.DEBUG)
        super(TestGradDescent, self).__init__(*args, **kwargs)

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
        result, _ = gradient_descent(f=f,
                                     f_grad=f_grad,
                                     start=start,
                                     step_strategy=step_strategy,
                                     stop_criteria=stop_criteria,
                                     eps_stop_criteria=EPSILON_STOP_CRITERIA,
                                     eps_strategy=EPSILON_STRATEGY,
                                     max_iterations_strategy=DEFAULT_MAX_ITERS_STRATEGY,
                                     max_iterations_criteria=DEFAULT_MAX_ITERS_CRITERIA
                                     )
        self.assertEqual(len(result), len(expected_result))
        dist = 0.0
        for result_x_k, expected_result_x_k in list(zip(result, expected_result)):
            dist += (result_x_k - expected_result_x_k) ** 2
        dist = math.sqrt(dist)
        logging.info(f'Distance: {dist}')
        TestGradDescent.max_ans_dist = max(TestGradDescent.max_ans_dist, dist)
        TestGradDescent.min_ans_dist = min(TestGradDescent.min_ans_dist, dist)
        self.assertAlmostEqual(dist, 0, delta=DELTA)

    @classmethod
    def tearDownClass(cls):
        logging.info(f'Max distance: {cls.max_ans_dist}')
        logging.info(f'Min distance: {cls.min_ans_dist}')
