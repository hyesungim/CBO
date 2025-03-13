import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
import multiprocessing as mp
from functools import partial
import numpy as np
from cbo import CBO
from bgkcbo import BGKCBO

def run_cbo(params):
    """Runs a single CBO optimization and returns error statistics."""
    objective, N, max_iter, sigma, lamda, alpha, dt = params
    
    # Create an independent CBO instance
    cbo_instance = CBO(objective=objective, N=N, max_iter=max_iter, sigma=sigma, lamda=lamda, alpha=alpha, dt=dt)   
    consensus_point = cbo_instance.optimize()  # Run optimization
    L_infty_error, L2_error, fitness = cbo_instance.errors(consensus_point)  # Compute errors
    return L_infty_error, L2_error, fitness

def cbo_experiments_sigmas(objectives, N, max_iter, sigmas, lamda=1.0, alpha=10000, dt=0.1, num_runs=100, save_path="cbo_results.pkl"):
    """Runs multiple independent CBO optimizations in parallel and computes statistics."""

    # Generate all (objective, sigma) combinations repeated num_runs times
    task_list = [(obj, N, max_iter, sigma, lamda, alpha, dt)  
                 for obj in objectives  
                 for sigma in sigmas  
                 for _ in range(num_runs)]

    # Run experiments in parallel
    with mp.Pool(processes=min(mp.cpu_count(), 4)) as pool:
        results = pool.map(run_cbo, task_list)

    # Convert results to a structured DataFrame, storing full lists
    df = pd.DataFrame([
        {
            "objective": objective.name,  # Extract objective name
            "N": N, "max_iter": max_iter, "sigma": sigma, "lamda": lamda, "alpha": alpha,
            "dt": dt, "L_infty_error": le, "L2_error": l2, "fitness": f
        }
        for (objective, N, max_iter, sigma, lamda, alpha, dt), (le, l2, f) in zip(task_list, results)
    ])

    # Group and store full error lists for each parameter setting
    df_errors = df.groupby(["objective", "N", "sigma", "lamda", "alpha", "dt"]).agg(
        L_infty_errors=("L_infty_error", list),
        L2_errors=("L2_error", list),
        fitness_values=("fitness", list)
    ).reset_index()

    # Compute summary statistics
    df_summary = df.groupby(["objective", "N", "sigma", "lamda", "alpha", "dt"]).agg(
        success_rate=("L_infty_error", lambda x: np.mean(np.array(x) < 0.25)),  # Success rate with threshold 0.25
        mean_error=("L_infty_error", np.nanmean),
        median_error=("L_infty_error", np.nanmedian),
        mean_L2_error=("L2_error", np.nanmean),
        mean_fitness=("fitness", np.nanmean)
    ).reset_index()

    # Merge summary with full error lists
    df_combined = pd.merge(df_summary, df_errors, on=["objective", "N", "sigma", "lamda", "alpha", "dt"], how="left")

    # Compute 0.1 and 0.9 quantiles for L_infty_errors
    df_combined["Q0.1_error"] = df_combined["L_infty_errors"].apply(lambda x: np.quantile(x, 0.1))
    df_combined["Q0.9_error"] = df_combined["L_infty_errors"].apply(lambda x: np.quantile(x, 0.9))

    # Save combined results to Pickle
    df_combined.to_pickle(save_path)

    print(f"Combined summary and error list results saved to {save_path}")

    return df_combined
