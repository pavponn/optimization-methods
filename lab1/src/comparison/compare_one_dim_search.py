import pandas as pd
from math import sqrt
from lab1.src.one_dim_search import (
    dichotomy_method,
    golden_selection_method,
    fibonacci_method
)
import lab1.src.utils.utils as ut

COMPARISON_DIR = '../../comparison_results'

precisions = [0.1, 0.01, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]


cases_for_comparison = [
    (lambda x: x + 1, -3, 4, -3),
    (lambda x: sqrt(x + sqrt(27)), sqrt(23), 100, sqrt(23)),
    (lambda x: x ** 2 - 4, -100, 100, 0),
    (lambda x: (x - 3) ** 2 + 5, -15, 100, 3),
    (lambda x: (x - 1) * (x - 23), -300, 400, 12),
    (lambda x: (x ** 2 - 2) / (4 - x ** 4), -111, 124, 0),
    (lambda x: (x ** 2 - 2) / (4 - x ** 4), -1, 1, 0)
]

MAX_ITER = 1e5


def compare():
    iters_data_frames = []
    calls_data_frames = []
    for counter, case in enumerate(cases_for_comparison):
        foo, a, b, _ = case
        dichotomy_iterations = []
        dichotomy_foo_calls = []
        golden_selection_iterations = []
        golden_selection_foo_calls = []
        fibonacci_iterations = []
        fibonacci_foo_calls = []
        for eps in precisions:
            _, iterations, foo_calls = dichotomy_method(foo, a, b, eps=eps, max_iter=MAX_ITER)
            dichotomy_iterations.append(iterations)
            dichotomy_foo_calls.append(foo_calls)
            _, iterations, foo_calls = golden_selection_method(foo, a, b, eps=eps, max_iter=MAX_ITER)
            golden_selection_iterations.append(iterations)
            golden_selection_foo_calls.append(foo_calls)
            _, iterations, foo_calls = fibonacci_method(foo, a, b, eps=eps)
            fibonacci_iterations.append(iterations)
            fibonacci_foo_calls.append(foo_calls)
        d_iters = \
            {
                'precision': precisions,
                'dich_iters': dichotomy_iterations,
                'gold_iters': golden_selection_iterations,
                'fib_iters': fibonacci_iterations
            }

        d_calls = \
            {
                'precision': precisions,
                'dich_calls': dichotomy_foo_calls,
                'gold_calls': golden_selection_foo_calls,
                'fib_calls': fibonacci_foo_calls
            }
        # foo_str = ut.get_function_string_representation(foo)
        iters_data_frames.append((counter, pd.DataFrame(data=d_iters)))
        calls_data_frames.append((counter, pd.DataFrame(data=d_calls)))
    return iters_data_frames, calls_data_frames


def compare_and_write_to_files():
    iteration_frames, calls_frames = compare()
    for name, iter_frame in iteration_frames:
        iter_frame.to_csv(f'{COMPARISON_DIR}/{name}_iters.csv', index=False)
    for name, call_frame in calls_frames:
        call_frame.to_csv(f'{COMPARISON_DIR}/{name}_calls.csv', index=False)


if __name__ == '__main__':
    compare_and_write_to_files()
