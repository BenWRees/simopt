from math import log, ceil
import warnings
warnings.filterwarnings("ignore")

from math import floor, log

import numpy as np
from numpy.linalg import norm

from simopt.base import Problem, Solution

__all__ = ['SamplingRule','BasicSampling', 'AdaptiveSampling', 'OriginalAdaptiveSampling', 'ASTROMoRFSampling']

class SamplingRule :
	def __init__(self, tr_instance, sampling_rule) :
		self.tr_instance = tr_instance
		self.sampling_rule = sampling_rule
		self.kappa = None

	#When sampling_rule is called is the behaviour of the sampling rule
	def __call__(self, problem, k, current_solution, delta_k, used_budget, sample_after=True) :
		current_solution, budget = self.sampling_rule(problem, k, current_solution, delta_k, used_budget, sample_after)
		return current_solution, budget

	# def __call__(self, *params) :
	# 	current_solution, budget = self.sampling_instance(*params)
	# 	return current_solution, budget
	
#This is a basic dynamic sampling rule - samples the objective fuction more 

class BasicSampling(SamplingRule) :
	def __init__(self, tr_instance) :
		self.tr_instance = tr_instance
		super().__init__(tr_instance, self.sampling)

	def sampling(self, problem, k, current_solution, delta_k, used_budget, sample_after=True) : 
		#sample 10 times 
		sample_number = 10
		problem.simulate(current_solution,sample_number)
		used_budget += sample_number
		return current_solution, used_budget
	

#ASTRODF originial adaptive sampling rule
class OriginalAdaptiveSampling(SamplingRule) :
	def __init__(self, tr_instance) :
		self.kappa = 1
		self.tr_instance = tr_instance
		super().__init__(tr_instance, self.sampling)		

	def sample_size(self, k, sig, delta_k) :
		alpha_k = 1
		lambda_k = 10*log(k,10)**1.5
		kappa = 10**2
		S_k = floor(max(5,lambda_k,(lambda_k*sig)/((kappa^2)*delta_k**(2*(1+1/alpha_k)))))
		#S_k = math.floor(max(lambda_k,(lambda_k*sig)/((kappa^2)*delta**(2*(1+1/alpha_k)))))
		return S_k

	def sampling(self, problem, k, current_solution, delta_k, used_budget, sample_after=True) :
		# need to check there is existing result
		problem.simulate(current_solution, 1)
		expended_budget += 1
		sample_size = 1
		
		# Adaptive sampling
		while True:
			problem.simulate(current_solution, 1)
			used_budget += 1
			sample_size += 1
			sig = current_solution.objectives_var
			if sample_size >= self.samplesize(k,sig,delta_k):
				break

		return current_solution, used_budget
		


class AdaptiveSampling(SamplingRule) : 
	def __init__(self, tr_instance) :
		self.kappa = 1
		# self.pilot_run = None
		super().__init__(tr_instance, self.sampling_strategy)
		# self.tr_instance = tr_instance
		# self.expended_budget = 0
		self.delta_power = 2 if tr_instance.factors['crn_across_solns'] else 4


	def calculate_pilot_run(self, k, problem, expended_budget) : 
		lambda_min = self.tr_instance.factors['lambda_min']
		lambda_max = problem.factors['budget'] - expended_budget
		return ceil(max(lambda_min * log(10 + k, 10) ** 1.1, min(0.5 * problem.dim, lambda_max))-1)

	def calculate_kappa(self, k, problem, expended_budget, current_solution, delta_k) :
		lambda_max = problem.factors['budget'] - expended_budget
		pilot_run = self.calculate_pilot_run(k, problem, expended_budget)

		#calculate kappa
		problem.simulate(current_solution, pilot_run)
		expended_budget += pilot_run

		# current_solution, expended_budget = self.__calculate_kappa(problem, current_solution, delta_k, expended_budget)
		sample_size = pilot_run
		
		while True:
			rhs_for_kappa = current_solution.objectives_mean
			sig2 = current_solution.objectives_var[0]

			self.kappa = rhs_for_kappa * np.sqrt(pilot_run) / (delta_k ** (self.delta_power / 2))
			stopping = self.get_stopping_time(sig2, delta_k, k, problem, expended_budget)
			if (sample_size >= min(stopping, lambda_max) or expended_budget >= problem.factors['budget']):
				# calculate kappa
				self.kappa = (rhs_for_kappa * np.sqrt(pilot_run)/ (delta_k ** (self.delta_power / 2)))
				# print("kappa "+str(kappa))
				break
			problem.simulate(current_solution, 1)
			expended_budget += 1
			sample_size += 1

		return current_solution, expended_budget

	def get_stopping_time(self, sig2: float, delta: float, k: int, problem: Problem, expended_budget: int) -> int:
		"""
		Compute the sample size based on adaptive sampling stopping rule using the optimality gap
		"""
		pilot_run = self.calculate_pilot_run(k, problem, expended_budget)
		if self.kappa == 0:
			self.kappa = 1

		# compute sample size
		raw_sample_size = pilot_run * max(1, sig2 / (self.kappa**2 * delta**self.delta_power))
		# Convert out of ndarray if it is
		if isinstance(raw_sample_size, np.ndarray):
			raw_sample_size = raw_sample_size[0]
		# round up to the nearest integer
		sample_size: int = ceil(raw_sample_size)
		return sample_size
	
	#this is sample_type = 'conditional after'
	def adaptive_sampling_1(self, problem, k, new_solution, delta_k, used_budget) :
		lambda_max = problem.factors['budget'] - used_budget
		pilot_run = self.calculate_pilot_run(k, problem, used_budget)

		problem.simulate(new_solution, pilot_run)
		used_budget += pilot_run
		sample_size = pilot_run

		# adaptive sampling
		while True:
			sig2 = new_solution.objectives_var[0]
			stopping = self.get_stopping_time(sig2, delta_k, k, problem, used_budget)
			if (sample_size >= min(stopping, lambda_max) or used_budget >= problem.factors['budget']):
				break
			problem.simulate(new_solution, 1)
			used_budget += 1
			sample_size += 1

		return new_solution, used_budget
	
	#this is sample_type = 'conditional before'	
	def adaptive_sampling_2(self, problem, k, new_solution, delta_k, used_budget) : 
		lambda_max = problem.factors['budget'] - used_budget
		sample_size = new_solution.n_reps 
		sig2 = new_solution.objectives_var[0]

		while True:
			stopping = self.get_stopping_time(sig2, delta_k, k, problem, used_budget)
			if (sample_size >= min(stopping, lambda_max) or used_budget >= problem.factors['budget']):
				break
			problem.simulate(new_solution, 1)
			used_budget += 1
			sample_size += 1
			sig2 = new_solution.objectives_var[0]
		return new_solution, used_budget


	def sampling_strategy(self, problem, k, new_solution, delta_k, used_budget, sample_after=True) :
		if sample_after :
			return self.adaptive_sampling_1(problem, k, new_solution, delta_k, used_budget)
		
		return self.adaptive_sampling_2(problem, k, new_solution, delta_k, used_budget)
	


class ASTROMoRFSampling(AdaptiveSampling) :
	"""
	ASTRO-MoRF sampling rule
	"""
	def __init__(self, tr_instance) :
		self.kappa = 1
		super().__init__(tr_instance)
		self.delta_power = 2 if tr_instance.factors['crn_across_solns'] else 4
		self.tr_instance = tr_instance

	