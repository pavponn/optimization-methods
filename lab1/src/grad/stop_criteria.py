from enum import Enum
import numpy as np


DEFAULT_EPS = 1e-9
DEFAULT_MAX_ITERATIONS = 1e9


class StopCriteria(Enum):
    BY_ARGUMENT = 1,
    BY_FUNCTION = 2,
    BY_GRAD = 3,
    BY_ITERATION = 4,


def get_stop_criteria(criteria, f, f_grad, eps, max_iterations):
    if criteria == StopCriteria.BY_ARGUMENT:
        return ArgumentStopCriteria(f, eps)
    elif criteria == StopCriteria.BY_FUNCTION:
        return FunctionStopCriteria(f, eps)
    elif criteria == StopCriteria.BY_ITERATION:
        return MaxIterationCriteria(f, eps, max_iterations)
    elif criteria == StopCriteria.BY_GRAD:
        return GradStopCriteria(f, f_grad, eps)
    else:
        raise TypeError("Unknown stop criteria")


class BaseStopCriteria(object):
    def __init__(self, f, eps=DEFAULT_EPS):
        self.f = f
        self.eps = eps

    def should_stop(self, x_cur, x_prev):
        raise NotImplemented


class ArgumentStopCriteria(BaseStopCriteria):
    def should_stop(self, x_cur, x_prev):
        return np.linalg.norm(x_cur - x_prev) < self.eps


class FunctionStopCriteria(BaseStopCriteria):
    def should_stop(self, x_cur, x_prev):
        return np.linalg.norm(self.f(x_cur) - self.f(x_prev)) < self.eps


class GradStopCriteria(BaseStopCriteria):
    def __init__(self, f, f_grad, eps=DEFAULT_EPS):
        super().__init__(f, eps)
        self.f_grad = f_grad

    def should_stop(self, x_cur, x_prev):
        return np.linalg.norm(self.f_grad(x_cur)) < self.eps


class MaxIterationCriteria(BaseStopCriteria):
    def __init__(self, f, eps=DEFAULT_EPS, max_iterations=DEFAULT_MAX_ITERATIONS):
        super().__init__(f, eps)
        self.max_iterations = max_iterations
        self.cur_iter = 0

    def should_stop(self, x_cur, x_prev):
        self.cur_iter += 1
        return self.cur_iter >= self.max_iterations
