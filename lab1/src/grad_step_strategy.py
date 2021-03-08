from enum import Enum
import numpy as np


class StepStrategy(Enum):
    DIVIDE_STEP = 1,
    CONSTANT_STEP = 2


def get_step_strategy(strategy, f, f_grad, eps):
    if strategy == StepStrategy.CONSTANT_STEP:
        return ConstantStepStrategy(f, f_grad, eps)
    elif strategy == StepStrategy.DIVIDE_STEP:
        return DivideStepStrategy(f, f_grad, eps)
    else:
        raise TypeError("unknown strategy")


class BaseStepStrategy(object):
    def __init__(self, f, f_grad, eps):
        self.f = f
        self.f_grad = f_grad
        self.eps = eps

    def next_step(self, x):
        raise NotImplemented


class ConstantStepStrategy(BaseStepStrategy):
    def __init__(self, f, f_grad, eps, start_alpha=10.0):
        super().__init__(f, f_grad, eps)
        self.cur_alpha = start_alpha
        self.iter = 0

    def next_step(self, x):
        if self.iter % 100 == 0:
            self.cur_alpha *= 1e4
        fx = self.f(x)
        while True:
            new_x = x - self.cur_alpha * self.f_grad(x)
            if self.f(new_x) <= fx:
                break
            else:
                self.cur_alpha /= 2

        return self.cur_alpha


class DivideStepStrategy(BaseStepStrategy):
    def __init__(self, f, f_grad, eps, alpha=10.0, delta=0.8, max_power=1e4):
        super().__init__(f, f_grad, eps)
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

# TODO: one dim strategies
