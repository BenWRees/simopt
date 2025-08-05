"""
    The Data Analysis class allows for taking results logs of an experiment and undergoing additional visualisation experiments including plotting the recomended solutions on the response surface 
    and undergoing statistical analysis of the results. 
"""

from simopt.experiment_base import ProblemSolver, Curve, read_experiment_results
import pandas as pd
import re
import ast
import numpy as np 
from simopt.linear_algebra_base import create_new_solution
from simopt.directory import problem_directory
from mrg32k3a.mrg32k3a import MRG32k3a
import matplotlib.pyplot as plt 

class DataAnalysis : 
    def __init__(self, experiment: ProblemSolver, pickle_file_path: None | str = None, file_path: None | str = None):
        if pickle_file_path is not None : 
            self.data : list[pd.DataFrame] = self.load_results_pickle(pickle_file_path)
        elif file_path is not None  : 
            self.data: list[pd.DataFrame] = self.load_experiment_results(file_path)   
        elif experiment.all_recommended_xs is not None : 
            self.data: list[pd.DataFrame] = self.load_results_problem_solver(experiment)
        else : 
            raise ValueError("Cannot load in appropriate experiment")
        
        self.experiment = experiment


    def load_results_pickle(self, file_path: str) -> list[pd.DataFrame] :
        """Loads in the problem solver from a pickle file

        Args:
            file_path (str): _description_

        Returns:
            pd.DataFrame: The data loaded from the pickle file
        """               
        #load problemsolver object from pickle_file 
        myexperiment = read_experiment_results(file_path) 
        return self.load_results_problem_solver(myexperiment)

    def load_results_problem_solver(self, experiment:ProblemSolver) -> list[pd.DataFrame] :
        """Takes a problem Solver object and constructs the fi

        Args:
            experiment (ProblemSolver): _description_

        Returns:
            pd.DataFrame: _description_
        """
        df_list = experiment.log_experiments_csv(save_to_excel=False)
        return df_list 

    def read_line_text_file(self,line: str) -> tuple[list, list] : 
        # Extract keys (text before colons)
        keys = re.findall(r'([\w\s]+):', line)

        # Extract values (text after colons)
        values = re.findall(r': ([^:]+)(?=\t|$)', line)

        # Convert numeric values properly
        parsed_values = []
        for value in values:
            value = value.strip()
            if value.startswith("(") and value.endswith(")"):  # Convert tuple-like strings to actual tuples
                if "np.float64" in value : #case with numpy float 
                    node = ast.parse(value, mode='eval')
                    # Extract numbers by visiting the AST nodes
                    float_tuple = tuple(float(arg.value) for arg in node.body.elts)
                    parsed_values.append(float_tuple)
                else :
                    parsed_values.append(tuple(map(int, value[1:-1].split(', '))))
            elif value.replace('.', '', 1).isdigit():  # Convert numbers
                parsed_values.append(float(value) if '.' in value else int(value))
            else:
                parsed_values.append(value)
        return keys, parsed_values

    def load_experiment_results(self, file_path: str) -> list[pd.DataFrame] : 
        """Loads in data from a experiment log text file. This has no optimality gap however 

        Args:
            file_path (str): the file path of the experiment log text file

        Returns:
            list[pd.DataFrame]: The data loaded from the experiment log
        """        
        #open the text file and at the start of every macroreplication create a new pandas dataframe and start populating row by row 
        capturing = False 
        df_list = []
        with open(file_path, 'r') as file : 
            for line in file : 
                line = line.strip() 
                match = re.match(r"Macroreplication (\d+):",line)
                if match : #if we are at the start of a macroreplication, we start collecting data on the next line
                    capturing = True 
                    df = pd.DataFrame()
                    idx = 0
                    continue 
                elif capturing == True and line == '' : #if we reach a new line then we stop capturing and continue reading lines
                    df_list.append(df)
                    df = pd.DataFrame()
                    capturing = False
                    continue  

                #here we populate the dataframe 
                if capturing :
                    #if it's a new macroreplication, we have to create column names for the dataframe 
                    col_names, values = self.read_line_text_file(line)
                    if df.empty() : 
                        # col_names.append('Optimality Gap')
                        # col_names = [a.upper() for a in col_names]
                        df.columns = col_names 

                    # values.append()
                    df.loc[idx] = values 
                    idx += 1
        return df_list
    

    
    def plot_recommended_solutions(self) -> tuple[list, list] : 
        """Plot the recommended Solutions as a vector curve

        Returns:
            tuple[list, list]: The x and y coordinates for the recommended solutions
            The x-values are the recommended solutions
            The y-values are the objectives_mean of the simulated recommended solutions
        """
        unpacked_dfs = []
        # sol_len = len(self.data[0]['Recommended Solution'].iloc[0]) #get the length of the recommended solution
        sol_len = self.experiment.problem.dim
        solution_names = ['solution_'+str(i) for i in range(1,sol_len+1)]
        
        for df in self.data : 
            unpacked_dfs.append(pd.DataFrame(df['Recommended Solution'].to_list(), columns=solution_names))

        max_rows = max(df.shape[0] for df in unpacked_dfs)
        aligned_dfs = [df.reindex(range(max_rows)) for df in unpacked_dfs]
        combined_sol = pd.concat(aligned_dfs).groupby(level=0).mean()

        mean_rec_sol = combined_sol.to_numpy().tolist()


        combined_obj = pd.concat([a['Estimated Objective'] for a in self.data], axis=1)
        combined_obj.columns = [f'Macroreplication {i}' for i in range(1,combined_obj.shape[1]+1)]
        combined_obj['Estimated Objective'] = combined_obj.mean(axis=1)

        return mean_rec_sol, combined_obj['Estimated Objective'].to_numpy()


    def plot_response_surface(self, no_pts: int) -> tuple[list,list] : 
        """Plot the experiment's problem response surface

        Returns:
            tuple[list, list]: The x and y coordinates for the problem response surface
            The x-values are a range of possible solutions
            The y-values are all the objectives_mean of the simulated possible solutions
        """
        #generate a grid of possible solution points based
        dim = self.experiment.problem.dim 
        min_bounds = self.experiment.problem.lower_bounds
        max_bounds = self.experiment.problem.upper_bounds 
        if dim < 2 : 
            grid_points = np.linspace(min_bounds[0], max_bounds[0], no_pts)
            sols = [] 

            for pt in grid_points : 
                pt = (pt)
                sol= create_new_solution(pt, self.experiment.problem)
                sols.append(sol)


            #simulate each solution 
            for sol in sols : 
                #! At the moment just get an accurate SAA - will change to having the problem deterministic for this example 
                self.experiment.problem.simulate(sol,1)


            obj_fn_vals = [] 
            for sol in sols : 
                fn = -1 * self.experiment.problem.minmax[0] * sol.objectives_mean
                obj_fn_vals.append(fn)
            
            return grid_points, obj_fn_vals
        
        else :
            x,y = np.linspace(min_bounds[0], max_bounds[0], no_pts), np.linspace(min_bounds[1], max_bounds[1], no_pts)
            X, Y = np.meshgrid(x,y)
            
            # step = (max_bounds[0]-min_bounds[0])/no_pts
            # grid_points = [(round(min_bounds[0] + i * step, 2), round(min_bounds[0] + j * step, 2)) for i in range(100) for j in range(100)]


            grid_points = [(a,b) for a,b in zip(X,Y)]
            if dim > 2 :
                add_grid_pts = []
                #add the other dimensions 
                for pt in grid_points : 
                    extra_dim = dim - 2
                    add_grid_pts.append(pt + (0.0,) * extra_dim)
                grid_points = add_grid_pts

            # Take all the grid points and evaluate a list of solutions 
            sols = [] 
            # if dim > 2 : 
            #     for pt in add_grid_pts : 
            #         sol = create_new_solution(pt, self.experiment.problem)
            #         sols.append(sol)
            # else :
            for pt in grid_points : 
                sol= create_new_solution(pt, self.experiment.problem)
                # sol.attach_rngs([MRG32k3a()]) #need to make all the randomness deterministic 
                sols.append(sol)


            #simulate each solution 
            for sol in sols : 
                #! At the moment just get an accurate SAA - will change to having the problem deterministic for this example 
                self.experiment.problem.simulate(sol,1)



            obj_fn_vals = [] 
            for sol in sols : 
                fn = -1 * self.experiment.problem.minmax[0] * sol.objectives_mean
                obj_fn_vals.append(fn)
        

            return X,Y, np.array(obj_fn_vals)



    #TODO: Get plots of the mean recommended solution over all macroreplications against the function plot 
    def show_plots(self, plot_type: str, no_pts: int) -> None : 
        """Show the plot for the designated plot type

        Args:
            plot_type (str): The type of plot to show. These are the following: 
                'recommended solutions map' - the vector line of the recommended solutions against the response surface
        """
        if plot_type.lower() ==  'recommended solutions map' : 
            if self.experiment.problem.dim < 2 : 
                rs_x, rs_y = self.plot_response_surface(no_pts)
                plt.plot(rs_x,rs_y)
                sol_x, sol_y = self.plot_recommended_solutions()
                plt.plot(sol_x,sol_y, color='black', marker='o')
                plt.show()
            else : 
                rs_x, rs_y, rs_z = self.plot_response_surface(no_pts)
                sol_x, sol_y = self.plot_recommended_solutions()
                fig = plt.figure(figsize=(10,10))
                ax = fig.add_subplot(projection='3d')
                
                # rs_x_1 = np.array([a[0] for a in rs_x])
                # rs_x_2 = np.array([a[1] for a in rs_x])

                sol_x_1 = np.array([a[0] for a in sol_x])
                sol_x_2 = np.array([a[1] for a in sol_x])

                # Plot the surface
                ax.plot_surface(rs_x, rs_y, rs_z, cmap='RdPu', label='Response Surface') #THIS ISN'T BEING OVERLAID 
                # fig.colorbar(surf, shrink=1.3, aspect=4, pad=0.1) 
                
                
                ax.plot(sol_x_1, sol_x_2, sol_y, color='black', marker='o', label='Mean Solution Path')
                ax.view_init(40,30)
                title = "Solution Path of " +self.experiment.solver.name + " for Experiment " + self.experiment.problem.name
                plt.legend()
                plt.title(title)
                plt.show()

                
                 


    def regression_analysis(self) : 
        """
            Undergoes Regression Analysis on the Space 
        """
        pass 