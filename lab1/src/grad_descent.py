from typing import Callable

import numpy as np
import lab1.src.grad_step_strategy as st

DEFAULT_EPSILON = 1e-9
DEFAULT_MAX_ITERATIONS = 1e5


def gradient_descent(f: Callable[[np.ndarray], float],
                     f_grad: Callable[[np.ndarray], np.ndarray],
                     start: np.ndarray,
                     eps: float = DEFAULT_EPSILON,
                     step_strategy: st.StepStrategy = st.StepStrategy.CONSTANT_STEP,
                     max_iterations=DEFAULT_MAX_ITERATIONS):
    strategy = st.get_step_strategy(step_strategy, f, f_grad, eps)
    cur_x = start
    iters = 0
    while True:
        iters += 1
        cur_grad = f_grad(cur_x)
        step = strategy.next_step(cur_x)
        cur_x = cur_x - step * cur_grad

        # TODO: stop criteria
        if np.linalg.norm(cur_grad) < eps:
            return cur_x

        if iters == max_iterations:
            return cur_x


if __name__ == '__main__':
    def foo(p):
        return p[0] ** 2 + p[1] ** 2


    def foo_grad(p):
        x, y = p[0], p[1]
        return np.array([2 * x, 2 * y])


    res = gradient_descent(foo, foo_grad, start=np.array([3, 4]))
    print(res)
