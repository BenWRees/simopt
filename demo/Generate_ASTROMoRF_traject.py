"""
    This script runs ASTROMoRF on ROSENBROCK-1 and then plots the recommended solutions and associated trust-region radii at each iteration on a response surface contour plot of the 
    simulation problem.
"""

from copy import deepcopy
import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from simopt.experiment_base import ProblemSolver
from simopt.base import Problem, Solution
from simopt.solvers.astromorf import ASTROMORF

def plot_response_surface(problem: Problem, recommended_solutions: list[tuple], ax=None, levels=50):
    """Plot the response surface of a 2D problem.
    Args:
        problem (Problem): The simulation problem to plot
        ax (matplotlib.axes.Axes, optional): Axes to plot on. Defaults to None.
        levels (int, optional): Number of contour levels. Defaults to 50.
    Returns:
        matplotlib.axes.Axes: Axes with the response surface plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    

    # Create grid for surface plot which is the max of the 
    #grid should be [min(x[0]-5, max(x[0])+5)]x[min(x[1]-5, max(x[1])+5)]
    x1 = np.linspace(min([sol[0] for sol in recommended_solutions]) - 5, max([sol[0] for sol in recommended_solutions]) + 5, 100)
    x2 = np.linspace(min([sol[1] for sol in recommended_solutions]) - 5, max([sol[1] for sol in recommended_solutions]) + 5, 100)
    X1, X2 = np.meshgrid(x1, x2)

    
    # Evaluate function on grid
    Z = np.zeros_like(X1)
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            point = (X1[i, j], X2[i, j])
            # sol = deepcopy(problem.factors['initial_solution'])
            # sol[:] = point
            # Use deterministic evaluation or create solution
            Z[i, j] = problem.model.fn(point)
    
    # Create contour plot
    contour = ax.contour(X1, X2, Z, levels=levels, cmap='viridis', alpha=0.6)
    ax.contourf(X1, X2, Z, levels=levels, cmap='viridis', alpha=0.3)
    # plt.colorbar(contour, ax=ax, label='Objective Value')
    
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    # ax.set_title('Response Surface with ASTROMoRF Trajectory')
    
    return ax


def plot_line_recommended_solutions(recommended_solutions: list[tuple], ax=None):
    """Plot the trajectory of recommended solutions and trust-region radii on the response surface.
    Args:
        recommended_solutions (list[tuple]): List of iterations with recommended solutions and trust-region
        radii
        ax (matplotlib.axes.Axes, optional): Axes to plot on. Defaults to None.
    Returns:
        matplotlib.axes.Axes: Axes with the trajectory plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    # Extract x and y coordinates
    x_coords = [sol[0] for sol in recommended_solutions]
    y_coords = [sol[1] for sol in recommended_solutions]
    
    # Plot points
    ax.scatter(x_coords, y_coords, c='red', s=100, zorder=5, 
               edgecolors='black', linewidths=1.5, label='Recommended Solutions')
    
    # Plot arrows connecting successive points
    for i in range(len(recommended_solutions) - 1):
        ax.annotate('', xy=(x_coords[i+1], y_coords[i+1]), 
                    xytext=(x_coords[i], y_coords[i]),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2, 
                                    shrinkA=5, shrinkB=5))
    
    # Label start and end points
    ax.plot(x_coords[0], y_coords[0], 'go', markersize=12, 
            label='Start', zorder=6)
    ax.plot(x_coords[-1], y_coords[-1], 'r*', markersize=15, 
            label='End', zorder=6)
    
    # Add iteration numbers
    # for i, (x, y) in enumerate(zip(x_coords, y_coords)):
    #     ax.text(x, y, f'{i}', fontsize=8, ha='center', va='bottom', 
    #             color='white', weight='bold',
    #             bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
    
    return ax


def plot_trust_regions(deltas: list[float], recommended_solutions: list[tuple], ax=None):
    """Plot trust regions around recommended solutions.
    Args:
        deltas (list[float]): List of trust-region radii
        recommended_solutions (list[tuple]): List of recommended solutions
        ax (matplotlib.axes.Axes, optional): Axes to plot on. Defaults to None.
    Returns:
        matplotlib.axes.Axes: Axes with the trust-region plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot trust region circles
    for i, (sol, delta) in enumerate(zip(recommended_solutions, deltas)):
        circle = Circle((sol[0], sol[1]), delta, 
                       fill=False, edgecolor='blue', 
                       linestyle='-', linewidth=2, 
                       alpha=0.6, label='Trust Region' if i == 0 else '')
        ax.add_patch(circle)
    
    return ax

    
def get_fn_estimations(xs: list[tuple], ps: ProblemSolver, problem: Problem) -> list[float]:
    fn_ests = []
    for x in xs:
        sol = Solution(x, problem)
        sol.attach_rngs(ps.solver.rng_list)
        problem.simulate(sol, 10)
        fx = sol.objectives_mean.item()
        fn_ests.append(fx)
    return fn_ests


def main(): 
    """Main function to run ASTROMoRF and plot results.
    """
    # Construct the problem and run ASTROMoRF on problem 
    ps = ProblemSolver(solver_name="ASTROMORF", problem_name="EXAMPLE-1", problem_fixed_factors={"budget": 5000})

    print(f'Running ASTROMoRF on EXAMPLE-1...')
    ps.run(n_macroreps=1)

    # Generate figure and axes from matplotlib 
    fig, ax = plt.subplots(figsize=(12, 10))

    # Extract the first experiment (only one macroreplication)
    solver = ps.solver
    problem = ps.problem
    recommended_solutions = ps.all_recommended_xs[0]  # list of x_recommended
    print(f'The recommended solutions at each iteration are: \n {recommended_solutions} \n')
    #read deltas from JSON file and store in variable deltas 
    with open('deltas.json', 'r') as f:
        deltas = json.load(f)
    print(f'The trust-region radii at each iteration are: \n {deltas} \n')

    print(f'Number of iterations: {len(recommended_solutions)}')
    print(f'Number of deltas: {len(deltas)}')

    # Plot everything on the same axes
    plot_response_surface(problem, recommended_solutions, ax=ax)
    plot_trust_regions(deltas, recommended_solutions, ax=ax)
    plot_line_recommended_solutions(recommended_solutions, ax=ax)
    
    ax.legend(loc='best', fontsize=20)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('astromorf_trajectory.png', dpi=300, bbox_inches='tight')
    print('Plot saved as astromorf_trajectory.png')
    plt.show()

if __name__ == "__main__":
    main()