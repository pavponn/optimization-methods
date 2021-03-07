import unittest
from parameterized import parameterized, parameterized_class

from src.one_dim_search import (
    dichotomy_method,
    golden_selection_method,
    fibonacci_method
)

DELTA = 1e-6

testcases = [
    (lambda x: x ** 2, -4, 6, 0),
    (lambda x: x ** 2 - 4, -23, 11, 0),
    (lambda x: (x - 3) ** 2 + 5, -1.5, 6, 3),
    (lambda x: (x - 1) * (x - 23), 2, 28, 12)
]


class TestOneDimSearch(unittest.TestCase):

    @parameterized.expand(testcases)
    def test_dichotomy_method(self, foo, a, b, expected_result):
        res, _ = dichotomy_method(foo, a, b)
        self.assertAlmostEqual(res, expected_result, delta=DELTA)

    @parameterized.expand(testcases)
    def test_golden_selection_method(self, foo, a, b, expected_result):
        res, _ = golden_selection_method(foo, a, b)
        self.assertAlmostEqual(res, expected_result, delta=DELTA)

    @parameterized.expand(testcases)
    def test_fibonacci_method(self, foo, a, b, expected_result):
        res, _ = fibonacci_method(foo, a, b)
        self.assertAlmostEqual(res, expected_result, delta=DELTA)


if __name__ == '__main__':
    unittest.main()
