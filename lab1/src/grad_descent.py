from typing import Callable

import numpy as np
import lab1.src.grad_step_strategy as st
import lab1.src.stop_criteria as sc

DEFAULT_EPSILON = 1e-9
DEFAULT_MAX_ITERATIONS = 1e5


def gradient_descent(f: Callable[[np.ndarray], float],
                     f_grad: Callable[[np.ndarray], np.ndarray],
                     start: np.ndarray,
                     eps: float = DEFAULT_EPSILON,
                     step_strategy: st.StepStrategy = st.StepStrategy.CONSTANT_STEP,
                     stop_criteria: sc.StopCriteria = sc.StopCriteria.BY_GRAD,
                     max_iterations=DEFAULT_MAX_ITERATIONS):
    strategy = st.get_step_strategy(step_strategy, f, f_grad, eps)
    criteria = sc.get_stop_criteria(stop_criteria, f, f_grad, eps, max_iterations)
    cur_x = start
    iters = 0
    while True:
        iters += 1
        cur_grad = f_grad(cur_x)
        step = strategy.next_step(cur_x)
        next_x = cur_x - step * cur_grad

        if criteria.should_stop(cur_x, next_x):
            return cur_x

        cur_x = next_x

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
