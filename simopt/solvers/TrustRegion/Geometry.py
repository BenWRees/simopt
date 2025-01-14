import numpy as np 
from math import comb
from numpy.linalg import norm
from ...base import (
	Problem
)

__all__ = ['TrustRegionGeometry', 'AstroDFGeometry', 'OMoRFGeometry']

class TrustRegionGeometry :
	def __init__(self, problem: Problem):
		self.problem = problem

	def standard_basis(self, index: int) -> list[float]:
		"""
		Creates a standard basis vector e_i in the space of dimension equal to the problem dimension. Where i is at the index of the index parameter
		Args:
			index (int): the location of the value 1 in the standard basis vector
		
		Returns:
			np.array: a standard basis vector of the form (0,0,...,0,1,0,...,0), where the 1 is in the location of index
		"""
		arr = np.zeros(self.problem.dim)
		arr[index] = 1.0
		return arr

	def interpolation_points(self, current_solution: np.ndarray, delta: float) -> list[np.ndarray]:
		"""
		Constructs an interpolation set of 
		
		Args:
			delta (TYPE): Description
		
		Returns:
			[np.array]: Description
		"""
		x_k = current_solution
		d = self.problem.dim

		Y = [x_k]
		epsilon = 0.01
		for i in range(0, d):
			plus = Y[0] + delta * self.standard_basis(i)
			minus = Y[0] - delta * self.standard_basis(i)

			if sum(x_k) != 0: #check if x_k is not the origin
				# block constraints
				if minus[i] <= self.problem.lower_bounds[i]:
					minus[i] = self.problem.lower_bounds[i] + epsilon
				if plus[i] >= self.problem.upper_bounds[i]:
					plus[i] = self.problem.upper_bounds[i] - epsilon

			Y.append(plus)
			Y.append(minus)
		return Y

class AstroDFGeometry(TrustRegionGeometry) :
	def __init__(self, problem: Problem) -> None :
		super().__init__(problem)

	# generate the basis (rotated coordinate) (the first vector comes from the visited design points (origin basis)
	def get_rotated_basis(self, first_basis: np.ndarray, rotate_index: np.ndarray) -> np.ndarray:
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
	def get_rotated_basis_interpolation_points(self, x_k: np.ndarray, delta: float, rotate_matrix: np.ndarray, reused_x: np.ndarray) -> list[np.ndarray]:
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
	
"""
	Class that represents the geometry of the solution space. It is able to construct an interpolation set and handle geometric behaviours of the space
"""
class OMoRFGeometry(TrustRegionGeometry) :
	def __init__(self, problem: Problem):
		self.problem = problem
		# self.current_val = current_val

	def assign_current_val(self, current_val) :
		"""
			Assigns the current iteration solution to the class member current_val

		Args:
			current_val (np.array): the current iteration solution 
		"""
		self.current_val = current_val

	def interpolation_points(self, current_val, delta, degree, sub_dim):
		"""
		Samples 0.5*(d+1)*(d+2) points within the trust region
		Args:
			delta (float): the trust-region radius
		
		Returns:
			[np.array]: the trust_region set
		"""

		# size = int(0.5*(self.problem.dim + 1)*(self.problem.dim + 2))
		# size = comb(sub_dim + degree, degree)
		dim = len(current_val)
		size = comb(sub_dim + degree, degree) + dim*sub_dim #- (sub_dim*(sub_dim+1))//2
		print('number of interpolation points: ', size)
		x_k = current_val 
		Y = [x_k]
		for i in range(1,size) : 
			random_vector = np.random.normal(size=len(x_k))
			random_vector /= norm(random_vector) #normalise the vector
			random_vector *= delta #scale the vector to lie on the surface of the trust region 
			random_vector += x_k #translate vector to be in trust region
			Y.append(random_vector)

		#maybe make half of the vectors stored in Y reflections by -rand_vect?
		return Y 