import unittest
from parameterized import parameterized
import logging

from lab1.src.onedim.one_dim_search import (
    dichotomy_method,
    golden_selection_method,
    fibonacci_method
)

DELTA = 1e-6

TEST_CASES = [
    (lambda x: x + 1, -3, 4, -3),
    (lambda x: -x - 1, -50, 10, 10),
    (lambda x: x ** 2, -4, 6, 0),
    (lambda x: x ** 2 - 4, -100, 100, 0),
    (lambda x: (x - 3) ** 2 + 5, -15, 100, 3),
    (lambda x: (x - 1) * (x - 23), -300, 400, 12),
    (lambda x: (x - 3) ** 2 + 8, -1e4, 1e5, 3),
    (lambda x: (x ** 2 - 2) / (4 - x ** 4), -111, 124, 0),
    (lambda x: (x ** 2 - 2) / (4 - x ** 4), -1, 1, 0)
]


class TestOneDimSearch(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        logging.basicConfig(format='%(asctime)s    |    %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S%p',
                            level=logging.DEBUG)
        super(TestOneDimSearch, self).__init__(*args, **kwargs)

    @parameterized.expand(TEST_CASES)
    def test_dichotomy_method(self, f, a, b, expected_result):
        res = dichotomy_method(f, a, b, enable_logging=True)[0]
        self.assertAlmostEqual(res, expected_result, delta=DELTA)

    @parameterized.expand(TEST_CASES)
    def test_golden_selection_method(self, f, a, b, expected_result):
        res = golden_selection_method(f, a, b, enable_logging=True)[0]
        self.assertAlmostEqual(res, expected_result, delta=DELTA)

    @parameterized.expand(TEST_CASES)
    def test_fibonacci_method(self, f, a, b, expected_result):
        res = fibonacci_method(f, a, b, enable_logging=True)[0]
        self.assertAlmostEqual(res, expected_result, delta=DELTA)


if __name__ == '__main__':
    unittest.main()
