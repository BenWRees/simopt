from math import log, ceil
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from numpy.linalg import norm

__all__ = ['SamplingRule','BasicSampling', 'AdaptiveSampling']

class SamplingRule :
	def __init__(self, tr_instance, sampling_rule) :
		self.tr_instance = tr_instance
		self.sampling_rule = sampling_rule
		self.kappa = None

	#When sampling_rule is called is the behaviour of the sampling rule
	def __call__(self, problem, current_solution, n, delta_k, used_budget, init_sample_size, sig2, sample_after=True) :
		current_solution, budget = self.sampling_rule(problem, current_solution, n, delta_k, used_budget, init_sample_size, sig2, sample_after)
		return current_solution, budget

	# def __call__(self, *params) :
	# 	current_solution, budget = self.sampling_instance(*params)
	# 	return current_solution, budget
	
#This is a basic dynamic sampling rule - samples the objective fuction more 

class BasicSampling(SamplingRule) :
	def __init__(self, tr_instance) :
		self.tr_instance = tr_instance
		super().__init__(tr_instance, self.sampling)

	def sampling(self, problem, current_solution, n, delta_k, used_budget, init_sample_size, sig2, sample_after=True) : 
		#sample 10 times 
		sample_number = 10
		problem.simulate(current_solution,sample_number)
		used_budget += sample_number
		return current_solution, used_budget
		


class AdaptiveSampling(SamplingRule) : 
	def __init__(self, tr_instance) :
		self.kappa = 1
		self.pilot_run = 2
		super().__init__(tr_instance, self.sampling_strategy)
		# self.tr_instance = tr_instance
		# self.expended_budget = 0
		self.delta_power = 2 if tr_instance.factors['crn_across_solns'] else 4


	def calculate_pilot_run(self, k, problem, expended_budget, current_solution, delta_k) :
		lambda_min = self.tr_instance.factors['lambda_min']
		lambda_max = problem.factors['budget'] - expended_budget
		self.pilot_run = ceil(max(lambda_min * log(10 + k, 10) ** 1.1, min(0.5 * problem.dim, lambda_max))-1)

		if k == 1 :
			problem.simulate(current_solution, self.pilot_run)
			expended_budget += self.pilot_run
			sample_size = self.pilot_run

			current_solution, expended_budget = self.__calculate_kappa(problem, current_solution, k, delta_k, expended_budget, self.pilot_run)
		elif self.tr_instance.factors['crn_across_solns'] :
			# since incument was only evaluated with the sample size of previous incumbent, here we compute its adaptive sample size
			sample_size = current_solution.n_reps
			sig2 = current_solution.objectives_var[0]
			# adaptive sampling
			current_solution, sampling_budget = self.adaptive_sampling_1(problem, current_solution, k, delta_k, expended_budget, sample_size, sig2)
			expended_budget = sampling_budget

		return current_solution, expended_budget

	def get_stopping_time(self, pilot_run: int, sig2: float, delta: float, dim: int,) -> int:
		"""
		Compute the sample size based on adaptive sampling stopping rule using the optimality gap
		"""
		if self.kappa == 0:
			self.kappa = 1
		# lambda_k = max(
		#     self.factors["lambda_min"], 2 * log(dim + 0.5, 10)
		# ) * max(log(k + 0.1, 10) ** (1.01), 1)

		# compute sample size
		raw_sample_size = pilot_run * max(1, sig2 / (self.kappa**2 * delta**self.delta_power))
		# Convert out of ndarray if it is
		if isinstance(raw_sample_size, np.ndarray):
			raw_sample_size = raw_sample_size[0]
		# round up to the nearest integer
		sample_size: int = ceil(raw_sample_size)
		return sample_size
	
	def __calculate_kappa(self, problem, current_solution, k, delta_k, used_budget, sample_size) :

		lambda_max = problem.factors['budget'] - used_budget
		while True:
			rhs_for_kappa = current_solution.objectives_mean
			sig2 = current_solution.objectives_var[0]
			if self.delta_power == 0:
				sig2 = max(sig2, np.trace(current_solution.objectives_gradients_var),)
			self.kappa = rhs_for_kappa * np.sqrt(self.pilot_run) / (delta_k ** (self.delta_power / 2))
			stopping = self.get_stopping_time(self.pilot_run, sig2, delta_k, problem.dim,)
			if (sample_size >= min(stopping, lambda_max) or used_budget >= problem.factors['budget']):
				# calculate kappa
				self.kappa = (rhs_for_kappa * np.sqrt(self.pilot_run)/ (delta_k ** (self.delta_power / 2)))
				# print("kappa "+str(kappa))
				break
			problem.simulate(current_solution, 1)
			used_budget += 1
			sample_size += 1

		return current_solution, used_budget


	def get_sig_2(self, solution) :
		return solution.objectives_var[0]
	
	#this is sample_type = 'conditional after'
	def adaptive_sampling_1(self, problem, new_solution, k, delta_k, used_budget, sample_size, sig2) :
		lambda_max = problem.factors['budget'] - used_budget
		
		problem.simulate(new_solution, self.pilot_run)
		used_budget += self.pilot_run
		sample_size = self.pilot_run

		# adaptive sampling
		while True:
			sig2 = new_solution.objectives_var[0]
			stopping = self.get_stopping_time(self.pilot_run, sig2, delta_k, problem.dim,)
			if (sample_size >= min(stopping, lambda_max) or used_budget >= problem.factors['budget']):
				break
			problem.simulate(new_solution, 1)
			used_budget += 1
			sample_size += 1

		return new_solution, used_budget
	
	#this is sample_type = 'conditional before'	
	def adaptive_sampling_2(self, problem, new_solution, k, delta_k, used_budget, sample_size, sig2) : 
		lambda_max = problem.factors['budget'] - used_budget
		while True:
			stopping = self.get_stopping_time(self.pilot_run, sig2, delta_k, problem.dim,)
			if (sample_size >= min(stopping, lambda_max) or used_budget >= problem.factors['budget']):
				break
			problem.simulate(new_solution, 1)
			used_budget += 1
			sample_size += 1
			sig2 = new_solution.objectives_var[0]
		return new_solution, used_budget


	def sampling_strategy(self, problem, new_solution, k, delta_k, used_budget, sample_size, sig2, sample_after=True) :
		if sample_after :
			return self.adaptive_sampling_1(problem, new_solution, k, delta_k, used_budget, 0, sig2)
		
		return self.adaptive_sampling_2(problem, new_solution, k, delta_k, used_budget, sample_size, sig2)