import dark.tensor as dt

class Optimizer():
    def __init__(self, parameters):
        self.parameters = parameters

    def step(self):
        raise NotImplementedError()

    def zero_grad(self):
        for p in self.parameters:
            p.zero_grad()


class SGD(Optimizer):
    def __init__(self, parameters, lr=1e-3, momentum = 0):
        assert lr > 0 and lr <= 1
        assert momentum >= 0 and momentum <= 1

        super().__init__(parameters)
        self.lr = lr
        self.momentum = momentum
        self.prev_update = [0] * len(self.parameters)

    def step(self):
        for i, p in enumerate(self.parameters):
            if p.requires_grad == False: 
                continue
            
            update = self.lr * p.grad
            update = self.momentum * self.prev_update[i] + update
            self.prev_update[i] = update

            p.data = p.data - update


# https://gist.github.com/hrayrhar/3b809c5ae778485a9ea9d253c4bfc90a
class Adam(Optimizer):
    def __init__(self, parameters, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        self.lr = lr
        self.beta_1, self.beta_2 = betas
        self.eps = eps

        assert self.lr > 0 and self.lr <= 1
        assert self.beta_1 >= 0 and self.beta_1 <= 1
        assert self.beta_2 >= 0 and self.beta_2 <= 1
        assert self.eps >= 0

        super().__init__(parameters)
        self.iter = 0
        self.ms = None
        self.vs = None

    def step(self):
        if self.ms is None:
            self.ms = [dt.zeros(p.data.shape) for p in self.parameters]
            self.vs = [dt.zeros(p.data.shape) for p in self.parameters]

        self.iter += 1
        lr_t = self.lr * (dt.sqrt(1. - dt.power(self.beta_2, self.iter)) / (1. - dt.power(self.beta_1, self.iter)))

        for i in range(len(self.parameters)):
            p = self.parameters[i]
            if p.requires_grad == False: 
                continue

            m_t = (self.beta_1 * self.ms[i]) + (1. - self.beta_1) * p.grad
            v_t = (self.beta_2 * self.vs[i]) + (1. - self.beta_2) * dt.square(p.grad)
            
            p.data = p.data - lr_t * m_t / (dt.sqrt(v_t) + self.eps)
            self.ms[i] = m_t
            self.vs[i] = v_t