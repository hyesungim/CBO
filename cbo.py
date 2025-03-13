import numpy as np
import multiprocessing as mp
import pandas as pd
# import dask
# from dask import delayed, compute

def run_once(params):
    """Runs a single optimization with explicitly passed parameters."""
    (objective, N, max_iter, sigma, lamda, alpha, dt) = params  # Unpack parameters
    cbo_instance = CBO(objective=objective, N=N, max_iter=max_iter, sigma=sigma, 
                       lamda=lamda, alpha=alpha, dt=dt)
    
    consensus_point = cbo_instance.optimize()
    L_infty_error, L2_error, fitness = cbo_instance.errors(consensus_point)
    
    return L_infty_error, L2_error, fitness

class CBO:
    def __init__(self, 
                 objective,  
                 N=100, 
                 max_iter=1000, 
                 sigma=1.0, 
                 lamda=1.0, 
                 alpha=10000.0, 
                 dt=0.1):
        """
        Args:
            objective: Objective function to be minimized.
            d: dimension of the problem. This is defined in the objective. 
            original_domain: Range of the search space.
            N: Number of particles.
            max_iter: Maximum number of iterations. The default is 1,000.
            sigma: Diffusion parameter. The default is 1.0.
            lamda: Drift parameter. The default is 1.0.
            alpha: Strength of the weight function. The default is 10,000.
            dt: Time step size. The default is 0.1.
            delta: Convergence threshold. The default is 1e-4.
            reference_domain: Initialisation of particles.
        """
        self.objective = objective
        self.d = objective.d
        self.original_domain = objective.range_of_domain
        self.N = N
        self.max_iter = max_iter
        self.sigma = sigma
        self.lamda = lamda
        self.alpha = alpha
        self.dt = dt
        self.true_minimizer = objective.true_minimizer
        self.reference_domain = [-1,1]

        # Initialize particles from uniform distribution 
        self.particles_position = np.random.uniform(self.reference_domain[0], self.reference_domain[1], (self.N, self.d))
        
        # Initialiaze consensus point
        self.consensus_point = None
        
        # Store history
        self.consensus_history = []
        self.variance_history = []
        self.particles_history = []
        
        # For repeated simulation
        self.L_infty_errors = []
        self.L2_errors = []
        self.fitness_values = []
        self.success_rate = 0.0 

        # Early stopping
        self.iter_count = 0
        self.delta = 1e-4 
        self.n_stall = 500

    def rescale_to_original_domain(self, particles):
        """Rescale particles from the reference domain [-1, 1]^d to the original domain [a, b]^d."""
        a = self.original_domain[0]
        b = self.original_domain[1]
        return a + (particles - self.reference_domain[0]) * (b - a) / (self.reference_domain[1] - self.reference_domain[0])

    def rescale_to_reference_domain(self, particles):
        """Rescale particles from the original domain [a, b]^d back to the reference domain [-1, 1]^d."""
        a = self.original_domain[0]
        b = self.original_domain[1]
        return self.reference_domain[0] + (particles - a) * (self.reference_domain[1] - self.reference_domain[0]) / (b - a)    
    
    def compute_consensus_point(self):
        rescaled_particles = self.rescale_to_original_domain(self.particles_position)
        objective_values = self.objective(rescaled_particles)       
        adjusted_weights = np.exp(-self.alpha * (objective_values - np.min(objective_values)))
        consensus_point = np.sum(self.particles_position * adjusted_weights[:, np.newaxis], axis=0) / np.sum(adjusted_weights)
        return consensus_point
        
    def update_particles(self):
        normal_values = np.random.normal(0, 1, (self.N, self.d))
        Delta = self.particles_position-self.consensus_point
        self.particles_position += -(self.lamda) * self.dt * (Delta) + (self.sigma) * np.sqrt(self.dt) * (Delta) * normal_values
        #self.particles_position = np.clip(self.particles_position, self.reference_domain[0], self.reference_domain[1])
        
        '''# Compute variance of particles position
        position_mean = np.mean(self.particles_position, axis=0)
        position_variance = np.mean(np.linalg.norm(self.particles_position - position_mean, axis=1))
        self.variance_history.append(position_variance)'''
    
    def errors(self, consensus_point):
        L_infty_error = np.max(np.abs(consensus_point - self.true_minimizer))
        L2_error = np.linalg.norm(consensus_point - self.true_minimizer)
        fitness = self.objective(consensus_point)
        f = float(fitness[0])
        return L_infty_error, L2_error, f
            
    def fitness(self, consensus_point):
        return self.objective(consensus_point)

    def optimize(self):
        n = 0 # Counter for consecutive small updates
        self.consensus_point = self.compute_consensus_point() # Initialize 

        for i in range(self.max_iter):
            prev_consensus = self.consensus_point # Store previous consensus
            self.consensus_point = self.compute_consensus_point() # Compute new consensus
            self.update_particles() # Update particles
            self.consensus_history.append(self.consensus_point)
            self.particles_history.append(self.particles_position.copy())
        
            stopping_criterion = np.linalg.norm(self.consensus_point - prev_consensus)
            
            #print(f"X_alpha (n+1) - X_alpha (n) (L2) = {self.consensus_point-prev_consensus}={stopping_criterion}")
            if stopping_criterion < self.delta:
                n +=1
            else:
                n = 0 
            self.iter_count += 1
            
            if n >= self.n_stall:
                break
          
        final_consensus_point = self.consensus_history[-1]
        return final_consensus_point
       
    def success(self, num_runs=100, radius=0.25):
        success_count = 0
        params = (self.objective, self.N, self.max_iter, self.sigma, self.lamda, self.alpha, self.dt)  # Pack parameters

        with mp.Pool(processes=mp.cpu_count()) as pool:
            results = pool.map(run_once, [params] * num_runs)  # Pass the same params for each run

        # Unpack results
        L_infty_errors, L2_errors, fitness_values = zip(*results)

        self.L_infty_errors =L_infty_errors
        self.L2_errors = L2_errors
        self.fitness_values = fitness_values

        # Compute success rate
        success_count = sum(1 for err in L_infty_errors if err < radius)
        success_rate = success_count / num_runs

        return success_rate
