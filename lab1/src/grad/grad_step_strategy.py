from enum import Enum
import numpy as np

from lab1.src.onedim.one_dim_search import (
    dichotomy_method,
    golden_selection_method,
    fibonacci_method
)

DEFAULT_MAX_ITERS = 100
DEFAULT_MAX_STEP = 100
DEFAULT_START_ALPHA = 10.0


class StepStrategy(Enum):
    DIVIDE_STEP = 1,
    CONSTANT_STEP = 2
    DICHOTOMY_STEP = 3,
    GOLDEN_SELECTION_STEP = 4,
    FIBONACCI_STEP = 5,


def step_to_str(strategy: StepStrategy):
    names = {
        StepStrategy.DIVIDE_STEP: "Divide",
        StepStrategy.CONSTANT_STEP: "Constant",
        StepStrategy.DICHOTOMY_STEP: "Dichotomy",
        StepStrategy.GOLDEN_SELECTION_STEP: "Golden selection",
        StepStrategy.FIBONACCI_STEP: "Fibonacci",
    }
    return names[strategy] + " step"


def get_step_strategy(strategy, f, f_grad, eps, max_iters=DEFAULT_MAX_ITERS):
    strategies = {
        StepStrategy.CONSTANT_STEP: ConstantStepStrategy,
        StepStrategy.DIVIDE_STEP: DivideStepStrategy,
        StepStrategy.DICHOTOMY_STEP: DichotomyStepStrategy,
        StepStrategy.GOLDEN_SELECTION_STEP: GoldenSelectionStepStrategy,
        StepStrategy.FIBONACCI_STEP: FibonacciStepStrategy,
    }
    if strategy in strategies:
        return strategies[strategy](f, f_grad, eps, max_iters)
    else:
        raise TypeError("Unknown strategy")


class BaseStepStrategy(object):
    def __init__(self, f, f_grad, eps, max_iters=DEFAULT_MAX_ITERS):
        self.f = f
        self.f_grad = f_grad
        self.eps = eps
        self.max_iters = max_iters

    def next_step(self, x):
        raise NotImplemented


class ConstantStepStrategy(BaseStepStrategy):
    def __init__(self, f, f_grad, eps, max_iters=DEFAULT_MAX_ITERS, start_alpha=DEFAULT_START_ALPHA):
        super().__init__(f, f_grad, eps, max_iters)
        self.cur_alpha = start_alpha
        self.iter = 0

    def next_step(self, x):
        if self.iter % 100 == 0:
            self.cur_alpha *= 1e5
        fx = self.f(x)
        while True:
            self.iter += 1
            new_x = x - self.cur_alpha * self.f_grad(x)
            if self.f(new_x) <= fx:
                break
            else:
                self.cur_alpha /= 2

        return self.cur_alpha


class DivideStepStrategy(BaseStepStrategy):
    def __init__(self, f, f_grad, eps, max_iters=DEFAULT_MAX_ITERS, alpha=100.0, delta=0.8, max_power=1e5):
        super().__init__(f, f_grad, eps, max_iters)
        self.alpha = alpha
        self.delta = delta
        self.max_power = max_power

    def next_step(self, x):
        cur_lambda = self.alpha
        iters = 0
        while iters < self.max_power:
            iters += 1
            new_x = x - cur_lambda * self.f_grad(x)
            if self.f(new_x) > self.f(x) - self.eps * cur_lambda * np.linalg.norm(self.f_grad(x)) ** 2:
                cur_lambda *= self.delta
            else:
                break

        return cur_lambda


class DichotomyStepStrategy(BaseStepStrategy):
    def __init__(self, f, f_grad, eps, max_iters=DEFAULT_MAX_ITERS, max_step=DEFAULT_MAX_STEP):
        super().__init__(f, f_grad, eps, max_iters)
        self.max_step = max_step

    def next_step(self, x):
        return dichotomy_method(lambda step: self.f(x - step * self.f_grad(x)),
                                0, self.max_step, self.eps, self.max_iters)[0]


class GoldenSelectionStepStrategy(BaseStepStrategy):
    def __init__(self, f, f_grad, eps, max_iters=DEFAULT_MAX_ITERS, max_step=DEFAULT_MAX_STEP):
        super().__init__(f, f_grad, eps, max_iters)
        self.max_step = max_step

    def next_step(self, x):
        return golden_selection_method(lambda step: self.f(x - step * self.f_grad(x)),
                                       0, self.max_step, self.eps, self.max_iters)[0]


class FibonacciStepStrategy(BaseStepStrategy):
    def __init__(self, f, f_grad, eps, max_iters=DEFAULT_MAX_ITERS, max_step=DEFAULT_MAX_STEP):
        super().__init__(f, f_grad, eps, max_iters)
        self.max_step = max_step

    def next_step(self, x):
        return fibonacci_method(lambda step: self.f(x - step * self.f_grad(x)),
                                0, self.max_step, self.eps)[0]
