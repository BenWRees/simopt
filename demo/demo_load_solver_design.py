# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 12:45:43 2024

@author: Owner
"""

import sys
import os.path as o
sys.path.append(o.abspath(o.join(o.dirname(sys.modules[__name__].__file__), "..")))



from simopt.experiment_base import  ProblemsSolvers
import pandas as pd

# run this script in terminal from simopt folder

solver_name = 'RNDSRCH' # name of solver that design was created on
problem_names = ['SSCONT-1', 'SAN-1'] # list of problem names for solver design to be run on (if more than one version of same problem, repeat name)

# name of file containing design points (csv or excel): column headers must exactly match names of solver factors w/ each row representing a design point (can also use csv's generated by GUI)
design_filename = ".\data_farming_experiments\RNDSRCH_design.csv" 

# list of dictionaries that provide fixed factors for problems when you don't want to use the default values, if you want to use all default values use empty dictionary (first problem factors then model), order must match problem 
problem_fixed_factors = [[{},{}],[{},{}]]

solver_fixed_factors = {} # use this dictionary to change any default solver factors that were not included in the design

n_macroreps = 2 # number of macroreplication to run and each solver design point
n_postreps = 100 # number of post replications to run on each macro replication
n_postreps_init_opt = 200 # number of normalization postreplications to run at initial solution and optimal solution

# turn design file into df & retrive dp information
design_table = pd.read_csv(design_filename)

design_factor_names = design_table.columns.tolist() 

#remove GUI columns from list if present
design_factor_names.remove('Design #')
design_factor_names.remove('Solver Name')
design_factor_names.remove('Design Type')
design_factor_names.remove('Number Stacks')

dp_list = [] # list of all design points

for index, row in design_table.iterrows():
    dp = {} #dictionary of current dp
    for factor in design_factor_names:
        dp[factor] = row[factor] 
    dp_list.append(dp)

# add fixed solver factors to dps
for fixed_factor in solver_fixed_factors:
    for dp in dp_list:
        dp[fixed_factor] = solver_fixed_factors[fixed_factor]
    
n_dp = len(dp_list)
solver_names = []
for i in range(n_dp):
    solver_names.append(solver_name)
    
experiment = ProblemsSolvers(solver_factors= dp_list,
                             problem_factors = problem_fixed_factors,
                             solver_names = solver_names,
                             problem_names = problem_names)

experiment.run(n_macroreps)
experiment.post_replicate(n_postreps)
experiment.post_normalize(n_postreps_init_opt)
experiment.record_group_experiment_results()
experiment.log_group_experiment_results()
experiment.report_group_statistics()
