import pandas as pd
from math import sqrt

from lab1.src.onedim.one_dim_search import (
    dichotomy_method,
    golden_selection_method,
    fibonacci_method
)


COMPARISON_DIR = '../../comparison_results/onedim'

PRECISIONS = [0.1, 0.01, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]

CASES_FOR_COMPARISON = [
    (lambda x: x + 1, -3, 4),
    (lambda x: sqrt(x + sqrt(27)), sqrt(23), 100),
    (lambda x: x ** 2 - 4, -100, 100),
    (lambda x: (x - 3) ** 2 + 5, -15, 100),
    (lambda x: (x - 1) * (x - 23), -300, 400),
    (lambda x: (x ** 2 - 2) / (4 - x ** 4), -111, 124),
    (lambda x: (x ** 2 - 2) / (4 - x ** 4), -1, 1),
    (lambda x: (x - 1) / (1 - x ** 3), -500, 500)
]

MAX_ITER = 1e5


def compare():
    iters_data_frames = []
    calls_data_frames = []

    for i, (f, a, b) in enumerate(CASES_FOR_COMPARISON):
        dichotomy_iterations, dichotomy_f_calls = [], []
        golden_selection_iterations, golden_selection_f_calls = [], []
        fibonacci_iterations, fibonacci_f_calls = [], []

        for eps in PRECISIONS:
            _, dich_iterations, dich_f_calls = dichotomy_method(f, a, b, eps=eps, max_iter=MAX_ITER)
            dichotomy_iterations.append(dich_iterations)
            dichotomy_f_calls.append(dich_f_calls)

            _, gold_iterations, gold_f_calls = golden_selection_method(f, a, b, eps=eps, max_iter=MAX_ITER)
            golden_selection_iterations.append(gold_iterations)
            golden_selection_f_calls.append(gold_f_calls)

            _, fib_iterations, fib_f_calls = fibonacci_method(f, a, b, eps=eps)
            fibonacci_iterations.append(fib_iterations)
            fibonacci_f_calls.append(fib_f_calls)

        d_iters = \
            {
                'precision': PRECISIONS,
                'dich_iters': dichotomy_iterations,
                'gold_iters': golden_selection_iterations,
                'fib_iters': fibonacci_iterations
            }
        iters_data_frames.append((i, pd.DataFrame(data=d_iters)))

        d_calls = \
            {
                'precision': PRECISIONS,
                'dich_calls': dichotomy_f_calls,
                'gold_calls': golden_selection_f_calls,
                'fib_calls': fibonacci_f_calls
            }
        calls_data_frames.append((i, pd.DataFrame(data=d_calls)))

    return iters_data_frames, calls_data_frames


def compare_and_write_to_files():
    iteration_frames, calls_frames = compare()
    for name, iter_frame in iteration_frames:
        iter_frame.to_csv(f'{COMPARISON_DIR}/{name}_iters.csv', index=False)
    for name, call_frame in calls_frames:
        call_frame.to_csv(f'{COMPARISON_DIR}/{name}_calls.csv', index=False)


if __name__ == '__main__':
    compare_and_write_to_files()
