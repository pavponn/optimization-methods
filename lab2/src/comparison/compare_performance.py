import numpy as np

from lab1.src.grad.grad_step_strategy import StepStrategy
from lab1.src.grad.grad_descent import gradient_descent
from lab1.src.grad.stop_criteria import StopCriteria

from lab2.src.methods.newton_method import newton_method
from lab2.src.methods.conjugate_method import conjugate_direction_method
from lab2.src.comparison.compare_methods import (
    quadratic_function,
    quadratic_function_grad,
    quadratic_function_hess,
)


EPS = 1e-9

FUNCTIONS = [
    (
        "100(y - x)² + (1 - x)²",  # f_str
        quadratic_function,  # f
        quadratic_function_grad,  # f_grad
        quadratic_function_hess,  # f_hess
        np.array([-2, 0], dtype="float64"),  # b
        (1e12, -1e12),  # start
    )
]


def run_grad_descent(f, f_grad, _f_hess, _b, start):
    trajectory = []
    gradient_descent(f, f_grad, np.array(start),
                     step_strategy=StepStrategy.FIBONACCI_STEP,
                     stop_criteria=StopCriteria.BY_ARGUMENT,
                     trajectory=trajectory,
                     eps_strategy=EPS,
                     eps_stop_criteria=EPS)
    iters = len(trajectory) - 1
    return iters


def run_newton_method(f, f_grad, f_hess, _b, start):
    iters = newton_method(f, f_grad, f_hess, np.array(start), eps=EPS)[1]
    return iters


def run_conj_dir_method(_f, _f_grad, f_hess, b, start):
    iters = conjugate_direction_method(
        f_hess(np.array([])), b,
        np.array(start), eps=EPS
    )[1]
    return iters


METHODS = {
    'Gradient descent': run_grad_descent,
    'Newton method': run_newton_method,
    'Conjugate direction': run_conj_dir_method,
}


def main():
    for f_str, f, f_grad, f_hess, b, start in FUNCTIONS:
        for (method_name, method) in METHODS.items():
            print(f"Running {method_name}...")
            iters = method(f, f_grad, f_hess, b, start)
            print(f"Finished in {iters} iterations")


if __name__ == '__main__':
    main()
