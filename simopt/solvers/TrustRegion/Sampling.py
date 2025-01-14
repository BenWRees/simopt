from math import log, ceil
import warnings
warnings.filterwarnings("ignore")


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
		self.kappa = 0
		super().__init__(tr_instance, self.sampling_strategy)
		# self.tr_instance = tr_instance
		# self.expended_budget = 0
	

	# compute the sample size based on adaptive sampling stopping rule using the optimality gap
	def get_stopping_time(self, k, sig2, delta, kappa, dim):
		if kappa == 0: 
			kappa = 1
		lambda_k = max(self.tr_instance.factors["lambda_min"], 2 * log(dim + .5, 10)) * max(log(k + 0.1, 10) ** (1.01), 1)
		# compute sample size
		N_k = ceil(max(lambda_k, lambda_k * sig2 / (kappa ** 2 * delta ** 4)))
		return N_k

	#this is calculated once per iteration
	def calculate_kappa(self, problem, new_solution, delta_k, k, used_budget, sample_size) :
		lambda_max = problem.factors['budget'] - used_budget
		while True:
			problem.simulate(new_solution, 1)
			used_budget += 1
			sample_size += 1
			fn = new_solution.objectives_mean
			sig2 = new_solution.objectives_var
			if sample_size >= self.get_stopping_time(k, sig2, delta_k, fn / (delta_k ** 2), problem.dim) or \
				sample_size >= lambda_max or used_budget >= problem.factors['budget']:
				# calculate kappa
				self.kappa = fn / (delta_k ** 2)
				return used_budget
	
	def get_sig_2(self, solution) :
		return solution.objectives_var[0]
	
	#this is sample_type = 'conditional after'
	def adaptive_sampling_1(self, problem, new_solution, k, delta_k, used_budget, sample_size) :
		lambda_max = problem.factors['budget'] - used_budget
		problem.simulate(new_solution,1)
		used_budget += 1
		while True:
			problem.simulate(new_solution, 1)
			used_budget += 1
			sample_size += 1
			sig2 = self.get_sig_2(new_solution)
			if sample_size >= self.get_stopping_time(k, sig2, delta_k, self.kappa, problem.dim) or \
				sample_size >= lambda_max or used_budget >= problem.factors['budget']:
				break
		return new_solution, used_budget
	
	#this is sample_type = 'conditional before'
	def adaptive_sampling_2(self, problem, new_solution, k, delta_k, used_budget, sample_size, sig2) :
		lambda_max = problem.factors['budget'] - used_budget
		while True:
			if sample_size >= self.get_stopping_time(k, sig2, delta_k, self.kappa, problem.dim) or \
				sample_size >= lambda_max or used_budget >= problem.factors['budget']:
				break
			problem.simulate(new_solution, 1)
			used_budget += 1
			sample_size += 1
			sig2 = self.get_sig_2(new_solution)
		return new_solution, used_budget


	def sampling_strategy(self, problem, new_solution, k, delta_k, used_budget, sample_size, sig2, sample_after=True) :
		if sample_after :
			return self.adaptive_sampling_1(problem, new_solution, k, delta_k, used_budget, sample_size)
		
		return self.adaptive_sampling_2(problem, new_solution, k, delta_k, used_budget, sample_size, sig2)