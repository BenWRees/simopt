"""
    A Python script to generate a plot that shows how many unique design points are sampled for 
    each existing trust-region algorithm over the course of a single optimization run.
    The y-axis represents the cumulative total number of unique design points sampled,
    while the x-axis represents the number of iterations.
"""
import matplotlib.pyplot as plt
import numpy as np

def ASTRO_DF_refined_design_pts(dimension:int) -> list[tuple[int, int]] : 
    """
    Returns a list of total design points sampled per iteration at each dimension
    in the refined ASTRO-DF that samples 2n+2 design points per iteration but some iterations 
    sample 2n design points (when reuse is possible).
    """
    iterations = list(range(1, 101))
    best_sum_design_pts_iter = lambda d : 2*d+2 + sum([2*d for _ in iterations[1:]])   
    worst_sum_design_pts_iter = lambda d : sum([2*d + 2 for _ in iterations]) 
    best_case = [best_sum_design_pts_iter(n) for n in range(1,dimension+1)]
    worst_case = [worst_sum_design_pts_iter(n) for n in range(1,dimension+1)]

    
    worst_and_best_case = [(a,b) for a, b in zip(worst_case, best_case)]
    return worst_and_best_case
     

def STORM_design_pts(dimension:int) -> list[int] : 
    """
    Returns a list of total design points sampled per-iteration for each dimension.
    in the STORM algorithm that samples n+1 or 1/Delta^2 design points per iteration.
    """
    iterations = list(range(1, 101))
    sum_design_pts_iter = lambda d : (d**2+3*d+2)/2 + sum([1 for _ in iterations[1:]])
    design_pts = [sum_design_pts_iter(n) for n in range(1,dimension+1)]
    return design_pts

def Powell_design_pts(dimension:int) -> list[int] : 
    """
    Returns a list of total design points sampled per-iteration for each dimension.
    in the Powell algorithm that samples 2n+1 design points per iteration.
    """
    iterations = list(range(1, 101))
    sum_design_pts_iter = lambda d : (d**2+3*d+2)/2 + sum([1 for _ in iterations[1:]])   
    design_pts = [sum_design_pts_iter(n) for n in range(1,dimension+1)]
    return design_pts


def ASTRO_DF_design_pts(dimension:int) -> list[int] : 
    """
    Returns a list of total design points sampled per-iteration for each dimension.
    in the ASTRO-DF algorithm that samples 2n^2+2 design points per iteration.
    """
    iterations = list(range(1, 101))
    sum_design_pts_iter = lambda d : sum([(d**2+3*d+2)/2 for _ in iterations])   
    design_pts = [sum_design_pts_iter(n) for n in range(1,dimension+1)]
    return design_pts

def STRONG_design_pts(dimension:int) -> list[int] : 
    """
    Returns a list of total design points sampled per-iteration for each dimension.
    in the STRONG algorithm that samples __ design points per iteration 
    """
    iterations = list(range(1, 101))
    sum_design_pts_iter = lambda d : sum([d*(d+1)/2 for _ in iterations])   
    design_pts = [sum_design_pts_iter(n) for n in range(1,dimension+1)]
    return design_pts

def plot_unique_design_points():
    """
    Plots the cumulative unique design points sampled for each algorithm over the iterations.
    For astro_df_refined we have two sets of design points for the best and worst case 
    """
    dimensions = np.arange(1, 100)  # Example: 100 iterations
    largest_dimension = dimensions[-1]

    astro_df_refined = ASTRO_DF_refined_design_pts(largest_dimension)
    astro_df_worst = [a for a, _ in astro_df_refined]
    astro_df_best = [b for _, b in astro_df_refined]
    astro_df_mean = [(a + b) / 2 for a, b in astro_df_refined]
    storm = STORM_design_pts(largest_dimension)
    powell = Powell_design_pts(largest_dimension)
    astro_df = ASTRO_DF_design_pts(largest_dimension)
    strong = STRONG_design_pts(largest_dimension)

    plt.figure(figsize=(10, 6))
    plt.plot(dimensions, astro_df_worst, color='red', linestyle='--')
    plt.plot(dimensions, astro_df_best, color='red', linestyle='--')
    plt.fill_between(dimensions, astro_df_worst, astro_df_best, color='red', alpha=0.2)
    plt.plot(dimensions, astro_df_mean, label='ASTRO-DF-C', color='red')
    # plt.plot(dimensions, powell, label='UOBYQA/BOBYQA/NEWUOA', color='purple')
    plt.plot(dimensions, storm, label='STORM', color='orange')
    plt.plot(dimensions, astro_df, label='ASTRO-DF', color='green')
    plt.plot(dimensions, strong, label='STRONG', color='blue')

    plt.title('Cumulative Unique Design Points Sampled per Iteration')
    plt.xlabel('Dimension of Problem')
    plt.xlim(1, largest_dimension)
    plt.ylabel('Cumulative Unique Design Points')
    plt.legend()
    plt.yscale('log')
    plt.tight_layout()
    plt.grid(True)
    plt.show()

def main():
    plot_unique_design_points()

if __name__ == "__main__":
    main()