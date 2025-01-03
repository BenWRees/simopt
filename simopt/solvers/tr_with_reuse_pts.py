from numpy.linalg import norm
import numpy as np
from math import ceil
import warnings
warnings.filterwarnings("ignore")
import copy

from simopt.base import (
    Solution,
)

from simopt.solvers.trust_region_class import (random_model, trust_region_geometry)

"""
	Class for a probabilistic trust region with design points able to be reused
"""
# class trust_region_reuse_points(trust_region) :
# 	def __init__(self, name="TRUSTREGION_REUSE", fixed_factors=None) :
# 		if fixed_factors is None:
# 			fixed_factors = {'random_model type': 'random_model_reuse', }
# 		self.name = name
# 		self.objective_type = "single"
# 		self.constraint_type = "box"
# 		self.variable_type = "continuous"
# 		self.gradient_needed = False
# 		self.specifications = {
# 			"reuse_points": {
# 				"description": "reuse the previously visited points",
# 				"datatype": bool,
# 				"default": True
# 			},
# 			"ps_sufficient_reduction": {
# 				"description": "use pattern search if with sufficient reduction, 0 always allows it, large value never does",
# 				"datatype": float,
# 				"default": 0.1
# 			}
			
# 		}
# 		self.check_factor_list = {
# 			"crn_across_solns": self.check_crn_across_solns,
# 			"ps_sufficient_reduction": self.check_ps_sufficient_reduction
# 		}

# 		super().__init__(name,fixed_factors)

# 	def check_ps_sufficient_reduction(self):
# 			return self.factors["ps_sufficient_reduction"] >= 0



#This is the only change, where we need to deal with cases of reuse in the model construction
class random_model_reuse(random_model) :
	def __init__(self, geometry_instance, tr_instance, poly_basis, problem, sampling_instance, model_construction_parameters) :
		super().__init__(geometry_instance, tr_instance, poly_basis, problem, sampling_instance, model_construction_parameters)
	
	def construct_model(self, current_solution, delta, k, expended_budget, visited_pts_list) -> tuple[
        Solution,
        float,
        int,
        list[Solution],
        list[Solution],
		int
    ]:
		interpolation_solns = []
		x_k = current_solution.x
		reuse_points = True
		lambda_min: int = self.model_construction_parameters["lambda_min"]
		
		j = 0
		budget: int = self.problem.factors["budget"]
		lambda_max = budget - expended_budget
		pilot_run = ceil(max(lambda_min, min(.5 * self.problem.dim, lambda_max)) - 1)
		# lambda_max = budget / (15 * sqrt(problem.dim))
		
		if len(visited_pts_list) == 0 :
			visited_pts_list.append(current_solution)

		while True:
			fval = []
			j = j + 1
			delta_k = delta * self.model_construction_parameters['w'] ** (j - 1)

			#calculate kappa - model construction happens once per iteration, so this will only happen once per iteration
			if hasattr(self.sampling_instance.sampling_rule, 'calculate_kappa') and k==1 :
				#only calculate if the sampling instance has the class 'calculate_kappa' defined
				self.problem.simulate(current_solution, pilot_run)
				expended_budget += pilot_run
				sample_size = pilot_run
				expended_budget = self.sampling_instance.sampling_rule.calculate_kappa(self.problem, current_solution, delta_k, k, expended_budget, sample_size)
			else : 
				self.problem.simulate(current_solution, 2) 
				expended_budget += 2

			# Calculate the distance between the center point and other design points
			Dist = []
			for i in range(len(visited_pts_list)):
				Dist.append(norm(np.array(visited_pts_list[i].x) - np.array(x_k))-delta_k)
				# If the design point is outside the trust region, we will not reuse it (distance = -big M)
				if Dist[i] > 0:
					Dist[i] = -delta_k*10000

			# Find the index of visited design points list for reusing points
			# The reused point will be the farthest point from the center point among the design points within the trust region
			f_index = Dist.index(max(Dist))

			# If it is the first iteration or there is no design point we can reuse within the trust region, use the coordinate basis
			if (k == 1) or (norm(np.array(x_k) - np.array(visited_pts_list[f_index].x))==0) or not reuse_points :
				# Construct the interpolation set


				empty_geometry = copy.deepcopy(self.geometry_instance)
			
				Z = empty_geometry.interpolation_points(np.zeros(self.problem.dim), delta_k)
				Y = self.geometry_instance.interpolation_points(np.array(current_solution.x), delta_k)

			# Else if we will reuse one design point
			elif k > 1:
				first_basis = (np.array(visited_pts_list[f_index].x)-np.array(x_k)) / norm(np.array(visited_pts_list[f_index].x)-np.array(x_k))
				# if first_basis has some non-zero components, use rotated basis for those dimensions
				rotate_list = np.nonzero(first_basis)[0]
				rotate_matrix = self.geometry_instance.get_rotated_basis(first_basis, rotate_list)

				# if first_basis has some zero components, use coordinate basis for those dimensions
				for i in range(self.problem.dim):
					if first_basis[i] == 0:
						rotate_matrix = np.vstack((rotate_matrix, self.geometry_instance.get_coordinate_vector(i)))

				# construct the interpolation set
				Y = self.geometry_instance.get_rotated_basis_interpolation_points(np.array(x_k), delta_k, rotate_matrix, visited_pts_list[f_index].x)
				
				empty_geometry = copy.deepcopy(self.geometry_instance)
				Z = empty_geometry.get_rotated_basis_interpolation_points(np.zeros(self.problem.dim), delta_k, rotate_matrix, np.array(visited_pts_list[f_index].x) - np.array(x_k))
			else:
				error_msg = "Error in constructing the interpolation set"
				raise ValueError(error_msg)
	
			# Evaluate the function estimate for the interpolation points
			for i in range(2 * self.problem.dim + 1):
				# for x_0, we don't need to simulate the new solution
				if (k == 1) and (i == 0):
					# self.problem.simulate(current_solution,1) #no need to simulate the new solution
					fval.append(-1 * self.problem.minmax[0] * current_solution.objectives_mean)
					interpolation_solns.append(current_solution)
				# reuse the replications for x_k (center point, i.e., the incumbent solution)
				elif (i == 0):
					#SAMPLING STRAT 1
					init_sample_size = current_solution.n_reps
					sig2 = current_solution.objectives_var

					current_solution, sampling_budget = self.sampling_instance(self.problem, current_solution,\
																  k, delta_k, expended_budget, init_sample_size, sig2, False)
					expended_budget = sampling_budget
					fval.append(-1 * self.problem.minmax[0] * current_solution.objectives_mean)
					interpolation_solns.append(current_solution)

				# else if reuse one design point, reuse the replications
				elif (i == 1) and (norm(np.array(x_k) - np.array(visited_pts_list[f_index].x)) != 0) and reuse_points :
					reuse_solution = visited_pts_list[f_index]
					#SAMPLING STRAT 2
					init_sample_size = reuse_solution.n_reps
					# sig2 = self.sampling_instance.sampling_rule.get_sig_2(visited_pts_list[f_index])
					sig2 = reuse_solution.objectives_var
					
					reuse_solution, sampling_budget = self.sampling_instance(self.problem, reuse_solution,\
																  k, delta_k, expended_budget, init_sample_size, sig2, False)
					expended_budget = sampling_budget
					fval.append(-1 * self.problem.minmax[0] * reuse_solution.objectives_mean)
					interpolation_solns.append(reuse_solution)

				# for new points, run the simulation with pilot run
				else:
					#SAMPLING STRAT 3
					interpolation_pt_solution = self.tr_instance.create_new_solution(tuple(Y[i]), self.problem)
					visited_pts_list.append(interpolation_pt_solution)
					self.problem.simulate(interpolation_pt_solution, pilot_run)
					expended_budget += pilot_run 
					init_sample_size = pilot_run
		
					interpolation_pt_solution, sampling_budget = self.sampling_instance(self.problem, interpolation_pt_solution, k, delta_k, expended_budget, init_sample_size,0)
					expended_budget = sampling_budget
					fval.append(-1 * self.problem.minmax[0] * interpolation_pt_solution.objectives_mean)
					interpolation_solns.append(interpolation_pt_solution)

			# get the current model coefficients
			q, grad, Hessian = self.coefficient(Z, fval)

			if not self.model_construction_parameters['skip_criticality']:
				# check the condition and break
				if norm(grad) > self.model_construction_parameters['criticality_threshold']:
					break

			if delta_k <= self.model_construction_parameters['mu'] * norm(grad):
				break

			# If a model gradient norm is zero, there is a possibility that the code stuck in this while loop
			if norm(grad) == 0:
				break
		
		#save the final coefficients and function values
		self.coefficients = [q, grad, Hessian]
		self.fval = fval
		delta_k = min(max(self.model_construction_parameters['beta'] * norm(grad), delta_k), delta)

		return current_solution, delta_k, expended_budget, interpolation_solns, visited_pts_list, pilot_run






#adaptation to the trust_region_geometry to include the fixed geometry needed to reduce the interpolation set 
class astrodf_geometry(trust_region_geometry) :
	def __init__(self, problem) :
		super().__init__(problem)

	# generate the basis (rotated coordinate) (the first vector comes from the visited design points (origin basis)
	def get_rotated_basis(self, first_basis, rotate_index):
		rotate_matrix = np.array(first_basis)
		rotation = np.matrix([[0, -1], [1, 0]])

		# rotate the coordinate basis based on the first basis vector (first_basis)
		# choose two dimensions which we use for the rotation (0,i)
		for i in range(1,len(rotate_index)):
			v1 = np.array([[first_basis[rotate_index[0]]],  [first_basis[rotate_index[i]]]])
			v2 = np.dot(rotation, v1)
			rotated_basis = np.copy(first_basis)
			rotated_basis[rotate_index[0]] = v2[0][0]
			rotated_basis[rotate_index[i]] = v2[1][0]
			# stack the rotated vector
			rotate_matrix = np.vstack((rotate_matrix,rotated_basis))
		return rotate_matrix

	# compute the interpolation points (2d+1) using the rotated coordinate basis (reuse one design point)
	def get_rotated_basis_interpolation_points(self, x_k, delta, rotate_matrix, reused_x):
		Y = [x_k]
		epsilon = 0.01
		for i in range(self.problem.dim):
			if i == 0:
				plus = np.array(reused_x)
			else:
				plus = Y[0] + delta * rotate_matrix[i]
			minus = Y[0] - delta * rotate_matrix[i]

			if sum(x_k) != 0:
				# block constraints
				for j in range(self.problem.dim):
					if minus[j] <= self.problem.lower_bounds[j]:
						minus[j] = self.problem.lower_bounds[j] + epsilon
					elif minus[j] >= self.problem.upper_bounds[j]:
						minus[j] = self.problem.upper_bounds[j] - epsilon
					if plus[j] <= self.problem.lower_bounds[j]:
						plus[j] = self.problem.lower_bounds[j] + epsilon
					elif plus[j] >= self.problem.upper_bounds[j]:
						plus[j] = self.problem.upper_bounds[j] - epsilon

			Y.append(plus)
			Y.append(minus)
		return Y
		