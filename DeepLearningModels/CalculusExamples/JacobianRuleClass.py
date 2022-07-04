import numpy as np
class ODEModel(object):

    def __init__(self, eps = 1e-10):

        self.eps = eps
        self.farray = []

    def add_function(self, f):

        self.farray.append(f)

    def f(self, t, x):

        return np.array([fi(t, x) for fi in self.farray], dtype = float )

    def df(self, t, x):

        J = np.zeros([len(x), len(x)], dtype = float )

        for i in range(len(x)):
            x1 = x.copy()
            x2 = x.copy()

            x1[i] += self.eps
            x2[i] -= self.eps

            f1 = self.f(t, x1)
            f2 = self.f(t, x2)

            J[ : , i] = (f1 - f2) / (2 * self.eps)

        return J


F = ODEModel(eps = 1e-12)

eq1 = lambda t,u : u[1]
eq2 = lambda t,u : u[2]
eq3 = lambda t,u : u[3]
eq4 = lambda t,u : -8 * u[0] + np.sin(t) * u[1] - 3 * u[3] + t**2

F.add_function(eq1)
F.add_function(eq2)
F.add_function(eq3)
F.add_function(eq4)


t = 0
x = np.array([1, 2, 3, 4], dtype = float )
print (F.df(t, x))