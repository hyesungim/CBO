import numpy as np

class CBO:
    def __init__(self, 
                 objective,  
                 N=500, 
                 max_iter=1000, 
                 sigma=1.0, 
                 lamda=1.0, 
                 alpha=10000.0, 
                 dt=0.1, 
                 delta=1e-4):
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
        self.delta = delta
        self.true_minimizer = objective.true_minimizer
        self.reference_domain = [-1,1]

        # Initialize particles
        self.particles_position = np.random.uniform(self.reference_domain[0], self.reference_domain[1], (self.N, self.d))
        
        # Initialiaze consensus point
        self.consensus_point = np.random.randn(self.d)
        
        # Store history
        self.consensus_history = []
        self.variance_history = []
        self.particles_history = []
        
        # For repeated simulation
        self.L_infty_errors = []
        self.L2_errors = []
        self.fitness_values = []
        self.success_rate = 0.0
 
    
    def rescale_to_original_domain(self, particles):
        """Rescale particles from the reference domain [-1, 1]^d to the original domain [a, b]^d."""
        a, b = self.original_domain
        return a + (particles - self.reference_domain[0]) * (b - a) / (
            self.reference_domain[1] - self.reference_domain[0]
        )

    def rescale_to_reference_domain(self, particles):
        """Rescale particles from the original domain [a, b]^d back to the reference domain [-1, 1]^d."""
        a, b = self.original_domain
        return self.reference_domain[0] + (particles - a) * (
            self.reference_domain[1] - self.reference_domain[0]
        ) / (b - a)
        
    def compute_consensus_point(self):
        rescaled_particles = self.rescale_to_original_domain(self.particles_position)
        objective_values = np.array([self.objective(p) for p in rescaled_particles])
        
        adjusted_weights = np.exp(-self.alpha * (objective_values - np.min(objective_values)))
        consensus_point = np.sum(self.particles_position * adjusted_weights[:, np.newaxis], axis=0) / np.sum(adjusted_weights)
        return consensus_point
        
    def update_particles(self):
        normal_values = np.random.normal(0, 1, (self.N, self.d))
        
        self.consensus_point = self.compute_consensus_point()
        self.particles_position += -(self.lamda) * self.dt * (self.particles_position-self.consensus_point) 
        + (self.sigma) * np.sqrt(self.dt) * (self.particles_position-self.consensus_point) * normal_values
        self.particles_position = np.clip(self.particles_position, self.reference_domain[0], self.reference_domain[1])
        
        '''# Compute variance of particles position
        position_mean = np.mean(self.particles_position, axis=0)
        position_variance = np.mean(np.linalg.norm(self.particles_position - position_mean, axis=1))
        self.variance_history.append(position_variance)'''
    
    def compute_errors(self, consensus_point):
        # Calculate L_infty and L2 errors for final consensus point
        L_infty_error = np.max(np.abs(consensus_point - self.true_minimizer))
        L2_error = np.linalg.norm(consensus_point - self.true_minimizer)**2
        return L_infty_error, L2_error
            
    def optimize(self):
        # Compute Initial consensus point
        self.consensus_point = self.compute_consensus_point()
     
        for i in range(self.max_iter):
            self.update_particles()
            self.consensus_history.append(self.consensus_point)
            self.particles_history.append(self.particles_position.copy())

        return self.consensus_point
    
    def get_success_rate(self, num_runs=20, radius=0.25):
        success_count = 0
        
        for _ in range(num_runs):
            # Run a single optimization
            consensus_point = self.optimize()
            L_infty_error, L2_error = self.compute_errors(consensus_point)
            fitness = self.objective(consensus_point)
            
            self.L_infty_errors.append(L_infty_error)
            self.L2_errors.append(L2_error)
            self.fitness_values.append(fitness)
            
            if L_infty_error < radius:
                success_count += 1

        self.success_rate = (success_count / num_runs)
        return self.success_rate