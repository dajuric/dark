
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
            update = -self.lr * p.grad
            update = self.momentum * self.prev_update[i] + update
            self.prev_update[i] = update

            p.value = p.value + update

