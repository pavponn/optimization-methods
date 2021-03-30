DEFAULT_EPS = 1e-5
DEFAULT_MAX_ITERS = 1000


class BaseStepStrategy(object):
    def __init__(self, f, eps, max_iters):
        self.f = f
        self.eps = eps
        self.max_iters = max_iters

    def next_step(self, x, x_wave):
        raise NotImplemented


class ConstantStepStrategy(BaseStepStrategy):
    def __init__(self, f, eps=DEFAULT_EPS, max_iters=DEFAULT_MAX_ITERS):
        super().__init__(f, eps, max_iters)
        self.step = 10
        self.iters = 0

    def next_step(self, x, x_wave):
        if self.iters % 100 == 0:
            self.step *= 1e5

        fx = self.f(x)

        while True:
            self.iters += 1
            x_new = x + self.step * x_wave
            if self.f(x_new) < fx:
                return self.step
            if self.step < self.eps:
                return self.step
            self.step = self.step / 2
