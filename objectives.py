import numpy as np

class ObjectiveFunction:
    def __init__(self, name, range_of_domain, d=20, true_minimum=0.0, true_minimizer=None):
        self.name = name
        self.range_of_domain = range_of_domain
        self.d = d
        self.true_minimum = true_minimum
        self.true_minimizer = true_minimizer

    def clip_inputs(self, x):
        """Clip inputs to ensure they are within the domain range."""
        return np.clip(x, self.range_of_domain[0], self.range_of_domain[1])

    def handle_numerical_issues(self, value):
        """Handle NaN, infinity, or invalid values."""
        return np.nan_to_num(value, nan=1e-10, posinf=1e6, neginf=1e6)

    def __call__(self, x):
        """Each subclass must implement this method."""
        raise NotImplementedError("This method must be implemented by subclasses.")


class Ackley(ObjectiveFunction):
    def __init__(self, d=20):
        super().__init__(
            name="Ackley",
            range_of_domain=(-5, 5),
            d=d,
            true_minimum=0.0,
            true_minimizer=np.zeros(d)
        )
        self.a = 20
        self.b = 0.2
        self.c = 2 * np.pi
        
    def __call__(self, x):
        x = np.array(x)
        #x = self.clip_inputs(x) 
        n = len(x) 
        try:        
            sum1 = np.sum(x**2)
            sum2 = np.sum(np.cos(self.c * x))
            term1 = -self.a * np.exp(-self.b * np.sqrt(sum1 / n))
            term2 = -np.exp(sum2 / n)
            value = term1 + term2 + self.a + np.exp(1)
        except RuntimeWarning:
            value = float("inf")
        return self.handle_numerical_issues(value)

class Rastrigin(ObjectiveFunction):
    def __init__(self, d=20):
        super().__init__(
            name="Rastrigin",
            range_of_domain=(-5.12, 5.12),
            d=d,
            true_minimum=0.0,
            true_minimizer=np.zeros(d)
        )
        self.A = 10

    def  __call__(self, x):
        x = np.array(x)
        #x = self.clip_inputs(x)  
        n = len(x)        
        try:
            value =  self.A * n + np.sum(x**2 - self.A * np.cos(2 * np.pi * x))
        except RuntimeWarning:
            value = float("inf")
        return self.handle_numerical_issues(value)

class Griewank(ObjectiveFunction):
    def __init__(self, d=20):
        super().__init__(
            name="Griewank",
            range_of_domain=(-600, 600),
            d=d,
            true_minimum=0.0,
            true_minimizer=np.zeros(d)
        )

    def __call__(self, x):
        x = np.array(x)
        #x = self.clip_inputs(x)
        n = len(x) 
        try:          
            j = np.arange(1., n+1)
            s = np.sum(x**2)
            p = np.prod(np.cos(x / np.sqrt(j)))
            value = 1 + s/4000 - p
        except RuntimeWarning:
            value = float("inf")
        return self.handle_numerical_issues(value)
    
class Rosenbrock(ObjectiveFunction):
    def __init__(self, d=20):
        super().__init__(
            name="Rosenbrock",
            range_of_domain=(-5, 10),
            d=d,
            true_minimum=0.0,
            true_minimizer=np.ones(d)
        )

    def __call__(self, x):
        x = np.array(x)
        #x = self.clip_inputs(x)
        try:
            x0 = x[:-1]
            x1 = x[1:]
            value = np.sum((1-x0)**2) + 100*np.sum((x1-x0**2)**2)
        except RuntimeWarning:
            value = float("inf")
        return self.handle_numerical_issues(value)

class Salomon(ObjectiveFunction):
    def __init__(self, d=20):
        super().__init__(
            name="Salomon",
            range_of_domain=(-100, 100),
            d=d,
            true_minimum=0.0,
            true_minimizer=np.zeros(d)
        )

    def __call__(self, x):
        x = np.array(x)
        #x = self.clip_inputs(x)
        try:
            r = np.sqrt(np.sum(x**2))
            value = 1-np.cos(2*np.pi*r)+0.1*r
        except RuntimeWarning:
            value = float("inf")
        return self.handle_numerical_issues(value)

class Schwefel220(ObjectiveFunction):
    def __init__(self, d=20):
        super().__init__(
            name="Schwefel 2.20",
            range_of_domain=(-100, 100),
            d=d,
            true_minimum=0.0,
            true_minimizer=np.zeros(d)
        )

    def __call__(self, x):
        x = np.array(x)
        return np.sum(np.abs(x))

class XSYRandom(ObjectiveFunction):
    def __init__(self, d=20):
        super().__init__(
            name="XSY random",
            range_of_domain=(-5, 5),
            d=d,
            true_minimum=0.0,
            true_minimizer=np.zeros(d)
        )

    def __call__(self, x):
        x = np.array(x)
        #x = self.clip_inputs(x)
        try:
            eta = np.random.uniform(0,1,len(x))
            value = sum(np.abs(x[i])**(i+1)*eta[i] for i in range(len(x))) 
        except RuntimeWarning:
            value = float("inf")
        return self.handle_numerical_issues(value)
    
    