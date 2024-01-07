from dark import Parameter, Node, Constant

class Module():
    def __init__(self):
        self._modules = {}
        self._parameters = {}

    def forward(self, *x):
        pass

    def __call__(self, *x):
        x = [(Constant(p) if not isinstance(p, Node) else p) for p in x]
        return self.forward(*x)

    def modules(self):
        return self._modules.values()

    def parameters(self):
        params = []

        def get_params(module):
           for p in module._parameters.values():
               params.append(p)

           for m in module._modules.values():    
               get_params(m)

        get_params(self)
        return params

    def train(self):
        for p in self.parameters():
            p.requires_grad = True
        
    def eval(self):
       for p in self.parameters():
            p.requires_grad = False

    def apply(self, apply_func):
        for m in self._all_modules():
            apply_func(m)

    def _all_modules(self):
        modules = []

        def get_modules(module):
            modules.append(module)
            for m in module._modules.values():
                get_modules(m)

        get_modules(self)
        return modules

    def __setattr__(self, key, value) -> None: #TODO: introduce getattr ??
        if isinstance(value, Parameter):
            self._parameters[key] = value
        elif isinstance(value, Module):
            self._modules[key] = value

        super().__setattr__(key, value)

class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        for i, m in enumerate(modules):
            self._modules['Seq-' + str(i)] = m

    def forward(self, input):
        result = input
        for m in self.modules():
            result = m(result)

        return result

class ModuleList(Module):
    def __init__(self):
        super().__init__()
        self.list = []

    def __getitem__(self, idx):
        return self.list[idx]

    def __len__(self):
        return len(self.list)

    def __iter__(self):
        return iter(self.list)

    def append(self, module):
        key = 'Module-' + str(len(self.list))
        self._modules[key] = module
        
        self.list.append(module)
        return self