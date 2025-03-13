import numpy as np

class ObjectiveFunction:
    def __init__(self, name, range_of_domain, d=20, true_minimum=0.0, true_minimizer=None):
        self.name = name
        self.range_of_domain = range_of_domain
        self.d = d
        self.true_minimum = true_minimum
        self.true_minimizer = true_minimizer

    def __call__(self, x):
        """Each subclass must implement this method."""
        raise NotImplementedError("This method must be implemented by subclasses.")


class Ackley(ObjectiveFunction):
    def __init__(self, d=20):
        super().__init__(
            name="Ackley",
            range_of_domain=[-5, 5],
            d=d,
            true_minimum=0.0,
            true_minimizer=np.zeros(d)
        )

    def __call__(self, x):
        """Compute Ackley function in a vectorized way.
        - x: (N, d) array where N is the number of particles.
        - Returns: (N,) array with function values.
        """
        if x.ndim == 1:  # Ensure 2D input for single evaluations
            x = x[np.newaxis, :]
        
        self.d = x.shape[1]  # Set dimensionality
        a, b, c = 20, 0.2, 2 * np.pi
        sum_sq_term = -a * np.exp(-b * np.sqrt(np.sum(x ** 2, axis=1) / self.d))
        cos_term = -np.exp(np.sum(np.cos(c * x), axis=1) / self.d)
        return a + np.exp(1) + sum_sq_term + cos_term

class Rastrigin(ObjectiveFunction):
    def __init__(self, d=20):
        super().__init__(
            name="Rastrigin",
            range_of_domain=[-5.12, 5.12],
            d=d,
            true_minimum=0.0,
            true_minimizer=np.zeros(d)
        )

    def __call__(self, x):
        if x.ndim == 1:
            x = x[np.newaxis, :]
        
        self.d = x.shape[1]
        A = 10
        return A * self.d + np.sum(x ** 2 - A * np.cos(2 * np.pi * x), axis=1)

class Griewank(ObjectiveFunction):
    def __init__(self, d=20):
        super().__init__(
            name="Griewank",
            range_of_domain=[-600, 600],
            d=d,
            true_minimum=0.0,
            true_minimizer=np.zeros(d)
        )

    def __call__(self, x):
        if x.ndim == 1:
            x = x[np.newaxis, :]
        j = np.arange(1., x.shape[1] + 1)
        s = np.sum(x**2, axis=1) / 4000
        p = np.prod(np.cos(x / np.sqrt(j)), axis=1)
        return 1 + s - p  # Returns (N,)
    
class Rosenbrock(ObjectiveFunction):
    def __init__(self, d=20):
        super().__init__(
            name="Rosenbrock",
            range_of_domain=[-5, 10],
            d=d,
            true_minimum=0.0,
            true_minimizer=np.ones(d)
        )

    def __call__(self, x):
        if x.ndim == 1:
            x = x[np.newaxis, :]

        self.d = x.shape[1]
        return np.sum(100 * (x[:, 1:] - x[:, :-1]**2)**2 + (1 - x[:, :-1])**2, axis=1)

class Salomon(ObjectiveFunction):
    def __init__(self, d=20):
        super().__init__(
            name="Salomon",
            range_of_domain=[-100, 100],
            d=d,
            true_minimum=0.0,
            true_minimizer=np.zeros(d)
        )

    def __call__(self, x):
        if x.ndim == 1:
            x = x[np.newaxis, :]
        r = np.sqrt(np.sum(x**2, axis=1))
        return 1 - np.cos(2 * np.pi * r) + 0.1 * r  # Returns (N,)

class Schwefel220(ObjectiveFunction):
    def __init__(self, d=20):
        super().__init__(
            name="Schwefel 2.20",
            range_of_domain=[-100, 100],
            d=d,
            true_minimum=0.0,
            true_minimizer=np.zeros(d)
        )

    def __call__(self, x):
        x = np.array(x)
        return np.sum(np.abs(x), axis=1)

class XSYRandom(ObjectiveFunction):
    def __init__(self, d=20):
        super().__init__(
            name="XSY random",
            range_of_domain=[-5, 5],
            d=d,
            true_minimum=0.0,
            true_minimizer=np.zeros(d)
        )

    def __call__(self, x):
        if x.ndim == 1:
            x = x[np.newaxis, :]
        eta = np.random.uniform(0, 1, x.shape)  # Random eta for each element
        i = np.arange(1, x.shape[1] + 1)  # Index 1 to d
        return np.sum(np.abs(x) ** i * eta, axis=1)  # Returns (N,)
    
    
