class A:
    def __init__(self):
        self.A = None

    def compute(self):
        self.A = 10


class B:
    def __init__(self, function_A: A):
        self.function_a = function_A

    def compute(self):
        self.function_a.compute()


if __name__ == '__main__':
    func_a = A()
    func_b = B(function_A=func_a)
    func_b.compute()
    print(func_a.A)
