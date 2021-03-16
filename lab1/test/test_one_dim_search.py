import unittest
from parameterized import parameterized
import logging

from lab1.src.one_dim_search import (
    dichotomy_method,
    golden_selection_method,
    fibonacci_method
)

DELTA = 1e-6

testcases = [
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

    @parameterized.expand(testcases)
    def test_dichotomy_method(self, foo, a, b, expected_result):
        res, _, _ = dichotomy_method(foo, a, b, enable_logging=True)
        self.assertAlmostEqual(res, expected_result, delta=DELTA)

    @parameterized.expand(testcases)
    def test_golden_selection_method(self, foo, a, b, expected_result):
        res, _, _ = golden_selection_method(foo, a, b, enable_logging=True)
        self.assertAlmostEqual(res, expected_result, delta=DELTA)

    @parameterized.expand(testcases)
    def test_fibonacci_method(self, foo, a, b, expected_result):
        res, _, _ = fibonacci_method(foo, a, b, enable_logging=True)
        self.assertAlmostEqual(res, expected_result, delta=DELTA)


if __name__ == '__main__':
    unittest.main()
