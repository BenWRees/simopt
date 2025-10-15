"""
    Defines a Network Airline Revenue Management Model and Problem as presented in 
    'Simulation-Based Booking Limits for Airline Revenue Management' 
    by Dimitri Bertsimas and Sanne de Boer.

    The simulation model captures the dynamics of booking requests across multiple fare classes,
    for a single time window 
"""
from __future__ import annotations
from typing import Callable, Optional, Final

import numpy as np
from scipy.stats import beta
from scipy.special import gamma

from mrg32k3a.mrg32k3a import MRG32k3a
from simopt.base import ConstraintType, Model, Problem, VariableType
from simopt.utils import classproperty, override

NUM_FARE_CLASSES: Final[int] = 3  # Number of fare classes
NUM_ODF: Final[int] = 4  # Number of origin-destination fare classes

class AirlineRevenueModel(Model) :
    """A model for airline revenue management with multiple fare classes and stochastic demand."""

    @classproperty
    @override
    def class_name_abbr(cls) -> str:
        return "AIRLINE_REV"
    
    @classproperty
    @override
    def class_name(cls) -> str:
        return "Airline Revenue Management"
    
    @classproperty
    @override
    def n_rngs(cls) -> int:
        return 1
    
    @classproperty
    @override
    def n_responses(cls) -> int:
        return 1
    
    @classproperty
    @override
    def specifications(cls) -> dict[str, dict]: 
        return {
            "num_classes": {
                "description": "number of fare classes",
                "datatype": int,
                "default": 2,
            },
            "ODF_leg_matrix": {
                "description": "Matrix of odf classses served by each leg",
                "datatype": list,
                "default": [[1,0],[1,0],[0,1],[0,1],[1,1],[1,1]],
            },
            "prices": {
                "description": "prices for each origin-destination fare class",
                "datatype": tuple,
                "default": (300, 100, 150, 50, 100, 25),
            },
            "capacity": {
                "description": "total capacity for the flight",
                "datatype": tuple,
                "default": (100,200),
            },
            "booking limits": {
                "description": "booking limits for each fare class and each origin-destination pair",
                "datatype": tuple,
                "default": (3,3,3,3,3,3),
            },
            'alpha': {
                'description': 'parameters for the Polya process governing booking requests for each odf ',
                'datatype': tuple,
                'default': (2.0,2.0,2.0,2.0, 2.0, 2.0),
            },
            'beta': {
                'description': 'parameters for the Polya process governing booking requests for each odf',
                'datatype': tuple,
                'default': (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
            },
            'gamma_shape' : {
                'description': 'shape parameters for the gamma distribution governing expected demand for each odf',
                'datatype': tuple,
                'default': (2.0, 2.0, 2.0, 2.0, 2.0, 2.0),
            },
            'gamma_scale' : {
                'description': 'scale parameters for the gamma distribution governing expected demand for each odf',
                'datatype': tuple,
                'default': (50.0, 50.0, 50.0, 50.0, 50.0, 50.0),
            },
            'time steps' : {
                'description': 'number of time steps in the booking horizon',
                'datatype': int,
                'default': 1,
            },
            'tau' : {
                'description': 'length of booking period for each time step',
                'datatype': tuple,
                'default': (1,),
            },
        }
    
    @property
    @override
    def check_factor_list(self) -> dict[str, Callable]:
        return {
            'num_classes': self.check_num_classes,
            'ODF_leg_matrix': self.check_odf_leg_matrix,
            'prices': self.check_prices,
            'capacity': self.check_capacity,
            'booking limits': self.check_booking_limits,
            'alpha': self.check_alpha,
            'beta': self.check_beta,
            'gamma_shape': self.check_gamma_shape,
            'gamma_scale': self.check_gamma_scale,
            'time steps': self.check_time_steps,
            'tau': self.check_tau,
        }
    
    def check_num_classes(self) -> None:
        if not isinstance(self.factors['num_classes'], int) and self.factors['num_classes'] <= 0:
            raise ValueError("num_classes must be a positive integer.")
    
    def check_odf_leg_matrix(self) -> None:
        adjacency_matrix = np.array(self.factors['ODF_leg_matrix'])
        if len(adjacency_matrix.shape) != 2 or adjacency_matrix.shape[0] == 0 or adjacency_matrix.shape[1] == 0:
            raise ValueError("ODF_leg_matrix must be a non-empty 2D list or array.")
        if not np.all(np.isin(adjacency_matrix, [0, 1])):
            raise ValueError("ODF_leg_matrix must contain only 0s and 1s.")
        if not np.any(np.sum(adjacency_matrix, axis=1) > 0):
            raise ValueError("Each leg must serve at least one fare class.")
    
    def check_prices(self) -> None:
        odf,legs = np.array(self.factors['ODF_leg_matrix']).shape
        prices = list(self.factors['prices'])
        if len(prices) != odf and np.any(prices <= 0):
            raise ValueError("Length of prices must equal number of origin-destination fare classes, and all values must be positive.")
    
    def check_capacity(self) -> None:
        odf,legs = np.array(self.factors['ODF_leg_matrix']).shape
        if (len(self.factors['capacity']) != legs) and np.any(np.array(self.factors['capacity']) <= 0):
            raise ValueError("Length of capacity must equal number of legs, and all values must be positive.")

    def check_booking_limits(self) -> None:
        odf,legs = np.array(self.factors['ODF_leg_matrix']).shape
        booking_limits = list(self.factors['booking limits'])
        if len(booking_limits) != (odf) and np.all(np.subtract(booking_limits.sum(axis=1),np.array(self.factors['capacity']))>0):
            raise ValueError("The shape of the booking limits must have columns equal to the number of fare classes and rows equal to the number of flights. " \
            "The booking limits for all fare classes across a flight must not exceed the capacity of that flight ")
    
    def check_alpha(self) -> None:
        odf,legs = np.array(self.factors['ODF_leg_matrix']).shape
        if (len(self.factors['alpha']) != odf) and np.any(np.array(self.factors['alpha']) <= 0):
            raise ValueError("Length of alpha must equal number of fare classes times number of flights, and all values must be positive.")
        
    def check_beta(self) -> None:
        odf,legs = np.array(self.factors['ODF_leg_matrix']).shape
        if (len(self.factors['beta']) != odf) and np.any(np.array(self.factors['beta']) <= 0):
            raise ValueError("Length of beta must equal number of fare classes times number of flights, and all values must be positive.")
        
    def check_gamma_shape(self) -> None:
        odf,legs = np.array(self.factors['ODF_leg_matrix']).shape
        if (len(self.factors['gamma_shape']) != odf) and np.any(np.array(self.factors['gamma_shape']) <= 0):
            raise ValueError("Length of gamma_shape must equal number of fare classes times number of flights, and all values must be positive.")
        
    def check_gamma_scale(self) -> None:
        odf,legs = np.array(self.factors['ODF_leg_matrix']).shape
        if (len(self.factors['gamma_scale']) != odf) and np.any(np.array(self.factors['gamma_scale']) <= 0):
            raise ValueError("Length of gamma_scale must equal number of fare classes times number of flights, and all values must be positive.")
        
    def check_time_steps(self) -> None:
        if not isinstance(self.factors['time steps'], int) or self.factors['time steps'] <= 0:
            raise ValueError("time steps must be a positive integer.")
        
    def check_tau(self) -> None:
        if (len(self.factors['tau']) != self.factors['time steps']) or np.any(np.array(self.factors['tau']) <= 0):
            raise ValueError("Length of tau must equal number of time steps, and all values must be positive.")



    def __init__(self, fixed_factors: dict | None = None) -> None:
        super().__init__(fixed_factors)
    

    def beta_distribution(self, t: int, tau: int, alpha: float, beta: float) -> float:
        """Compute the cumulative distribution function (CDF) of the Beta distribution.

        Args:
            t (int): Current time step.
            tau (int): Total time steps.
            alpha (float): Alpha parameter of the Beta distribution.
            beta (float): Beta parameter of the Beta distribution.
        Returns:
            float: CDF value at time t.
        """
        t = t + 1 # Adjust for 1-based indexing
        first_term = (1/tau)*(t/tau)**(alpha-1) * (1 - t/tau)**(beta-1)
        second_term = gamma(alpha + beta) / (gamma(alpha) * gamma(beta))
        return first_term * second_term


    
    def gamma_variate(self, rand: MRG32k3a, shape: float, scale: float) -> float:
        """Generate a random value that obeys the Gamma distribution.

        Args:
            shape (float): Shape parameter of the Gamma distribution.
            scale (float): Scale parameter of the Gamma distribution.
        Returns:
            float: Random variate from the Gamma distribution.
        """    
        # Warning: a few older sources define the gamma distribution in terms
        # of alpha > -1.0
        if shape <= 0.0 or scale <= 0.0:
            raise ValueError('gammavariate: shape and scale must be > 0.0')

        random = rand.random
        if shape > 1.0:

            # Uses R.C.H. Cheng, "The generation of Gamma
            # variables with non-integral shape parameters",
            # Applied Statistics, (1977), 26, No. 1, p71-74

            ainv = np.sqrt(2.0 * shape - 1.0)
            bbb = shape - np.log(4.0)
            ccc = shape + ainv

            while True:
                u1 = random()
                if not 1e-7 < u1 < 0.9999999:
                    continue
                u2 = 1.0 - random()
                v = np.log(u1 / (1.0 - u1)) / ainv
                x = shape * np.exp(v)
                z = u1 * u1 * u2
                r = bbb + ccc * v - x
                if r + (1.0 + np.log(4.5)) - 4.5 * z >= 0.0 or r >= np.log(z):
                    return x * scale

        elif shape == 1.0:
            # expovariate(1/beta)
            return -np.log(1.0 - random()) * scale

        else:
            # alpha is between 0 and 1 (exclusive)
            # Uses ALGORITHM GS of Statistical Computing - Kennedy & Gentle
            while True:
                u = random()
                b = (np.e + shape) / np.e
                p = b * u
                if p <= 1.0:
                    x = p ** (1.0 / shape)
                else:
                    x = -np.log((b - p) / shape)
                u1 = random()
                if p > 1.0:
                    if u1 <= x ** (shape - 1.0):
                        break
                elif u1 <= np.exp(-x):
                    break
            return x * scale  
         
    
    def compute_arrivals(self, rand: MRG32k3a, t: int, remaining_capacity: np.ndarray) -> list[int] :
        """Compute the number of arrivals for a given fare class at time t.

        Args:
            rand (MRG32k3a): Random number generator.
            t (int): Current time step.
            remaining_capacity (np.ndarray): Remaining capacity for each leg of the network. This is a 1-d array of number of rows in booking_limits    
        Returns:
            list[int]: Number of arrivals for each odf class at time t.
        """
        booking_limits = list(self.factors['booking limits'])
        odf_classes = len(booking_limits)
        arrivals = [0] * odf_classes

        tau = self.factors['tau'][t]
        alpha = list(self.factors['alpha'])
        beta = list(self.factors['beta'])
        shape = list(self.factors['gamma_shape'])
        scale = list(self.factors['gamma_scale'])
       
        for x in range(odf_classes) : 
                
            # Compute the expected demand using the Gamma distribution
            if t==0:
                # For the first time step, we sample from the gamma distribution to get the expected demand
                expected_demand = self.gamma_variate(rand, shape[x], scale[x])
            else : #TODO: Work out parameters for gamma variable as they depend on number of booking requests up to time t-1
                # For subsequent time steps, we use the gamma distribution with values from the number of booking requests up to time t-1
                expected_demand = self.gamma_variate()
            
            # Compute the probability of arrival using the Beta CDF
            prob_arrival = self.beta_distribution(t, tau, alpha[x], beta[x])
            
            # Compute the number of arrivals as a Poisson random variable
            arrivals[x] = np.random.poisson(expected_demand * prob_arrival)
        
        return arrivals
    
    #TODO: Finish the function 
    def compute_leg_number(self, odf_class: int) -> list[int] :
        """Compute the legs involved for the given odf class.

        Args:
            odf_class (int): Origin-destination fare class.
        Returns:
            list[int]: List of leg numbers that the odf class travels on.
        """
        adjacency_matrix = np.array(self.factors['ODF_leg_matrix'])
        odf_vector = np.zeros((1,adjacency_matrix.shape[0]))
        odf_vector[:,odf_class] = 1
        leg_numbers = odf_vector @ adjacency_matrix
        leg_number_filtered = np.nonzero(leg_numbers)[1].tolist()
        return leg_number_filtered


    def sell_seats(self, arrivals: list[int], remaining_capacity: np.ndarray) -> tuple[list[int], list[int]]:
        """
            For a single time step, sell as many seats as possible given the arrivals, booking limits, and remaining capacity. 
            Args:
                arrivals (list[int]): Number of arrivals for each odf class at time t. This is a 1-d array of number of rows in booking_limits
                timestep (int): Current time step.
                remaining_capacity (np.ndarray): Remaining capacity for each leg of the network. This is a 1-d array of number of rows in booking_limits 
            Returns:
                tuple[list[int], list[int]]: A tuple containing:
                    - seats_sold (list[int]): Number of seats sold for each odf class at time t.
                    - remaining_capacity (list[int]): Updated remaining capacity for each leg of the network after selling seats.
        """
        booking_limits = list(self.factors['booking limits'])
        odf_classes = len(booking_limits)
        seats_sold = [0] * odf_classes

        # for each odf class, compute the leg number it travels on the network

        for odf in range(odf_classes) :
            arrival_no = arrivals[odf]
            # get the legs this odf class flies in the network
            leg_nos = self.compute_leg_number(odf)
            # get the remaining capacity for each leg this odf class flies on
            remaing_capacity_legs = [remaining_capacity[a] for a in leg_nos]
            for _ in range(arrival_no) : 
                # check if there is available space on one of the legs the odf flies on => go to next odf  
                if np.all(remaing_capacity_legs==0) :
                    break 
                # check if the booking limit for this odf has been reached => go to the next odf
                elif booking_limits[odf] <= seats_sold[odf] : 
                    break
                else :
                    # add one seat to the seats sold for this odf class and
                    seats_sold[odf] += 1
                    # update the remaining capacity for each leg the odf flies on 
                    for leg_no in leg_nos :
                        remaining_capacity[leg_no] -= 1 

        return seats_sold, remaining_capacity
    

    def compute_revenue(self, seats_sold: np.ndarray) -> list[float]:
        """Compute the total revenue for all the seats_sold across fare classes for a single time step

        Args:
            seats_sold (np.ndarray): Number of seats sold for each odf class at each time step. 
                                     This is a 2-d array of shape (time_steps, number of rows in booking_limits)
            
        Returns:
            list[float]: Revenue generated from each odf class indexed at each time step.
        """
        # total_revenue = 0.0
        timesteps, odfs = seats_sold.shape 
        prices = list(self.factors['prices'])
        revenue_over_time_steps = []

        for t in range(timesteps) : 
            seats_sold_per_time_step = seats_sold[t].tolist() 
            revenue_each_odf = sum([seats_sold_per_time_step[i] * prices[i] for i in range(odfs)]) 
            revenue_over_time_steps.append(revenue_each_odf)

        return revenue_over_time_steps

    def replicate(self, rng_list: list[MRG32k3a]) -> tuple[dict, dict]:
        """Simulate a single replication of the airline revenue management model.

        Args:
            decision_vars (tuple): Booking limits for each fare class.
            rng_streams (list[MRG32k3a]): List of random number generator streams.
        Returns:
            tuple[dict, dict]: A tuple containing:
                - responses (dict): Performance measures of interest, including:
                    - "revenue": Total revenue from the booking period.
                - gradients (dict): A dictionary of gradient estimates for
                    each response (None if not available).
            
        """ 

        #! Extract model parameters
        odf_legs_matrix = np.array(self.factors['ODF_leg_matrix'])
        odf_no, legs = odf_legs_matrix.shape
        capacity = self.factors['capacity']
        booking_limits = list(self.factors['booking limits'])
        odf = len(booking_limits)  # number of origin-destination fare classes
        time_steps = self.factors['time steps']
        total_revenue = 0

        remaining_capacity = np.array(capacity)  # Remaining capacity for each leg
        seats_remaining_from_booking_limit = np.array(booking_limits)  # Remaining booking limits for

        #! Designate sources of randomness 
        arrival_rng = rng_list[0]  
        """ 
            For each time step, we compute the number of arrivals for each odf class and then compute how many 
            seats can be sold on the networrk based on the booking limits and remaining capacity.
            We then update the remaining capacity and booking limits accordingly.
        """
        seats_sold_all_time_steps = np.zeros((time_steps, odf), dtype=int)
        for time_step in range(time_steps):
            """ 
                Compute arrivals at the current time step and then sell seats based on
                the booking limits and remaining capacity.
            """
            arrivals = self.compute_arrivals(arrival_rng, time_step, remaining_capacity)

            seats_sold_one_time_step, remaining_capacity = self.sell_seats(arrivals, remaining_capacity)
        
            seats_sold_all_time_steps[time_step, :] = seats_sold_one_time_step

        # Compute total revenue
        list_revenues_each_odf = self.compute_revenue(seats_sold_all_time_steps)

        #get the total revenues 
        total_revenue = sum([np.sum(a) for a in list_revenues_each_odf])

        responses = {
            'revenue': total_revenue,
        }
        gradients = {
            response_key: dict.fromkeys(self.specifications, np.nan)
            for response_key in responses
        }
        
        return responses, gradients

class AirlineRevenueBookingLimitProblem(Problem):
    """Finds the optimal set of booking limits to maximize expected revenue."""

    @classproperty
    @override
    def class_name_abbr(cls) -> str:
        return "AIRLINE-1"
    
    @classproperty
    @override
    def class_name(cls) -> str:
        return "Airline Revenue Management Booking Limit Problem"
    
    @classproperty
    @override
    def n_objectives(cls) -> int:
        return 1  # Tuple of decision variables
    

    @classproperty
    @override
    def n_stochastic_constraints(cls) -> int:
        return 0  # No stochastic constraints
    
    @classproperty
    @override
    def minmax(cls) -> tuple[int]:
        return (1,)  # Maximize revenue
    
    @classproperty
    @override
    def constraint_type(cls) -> ConstraintType:
      return ConstraintType.DETERMINISTIC
    
    @classproperty
    @override
    def variable_type(cls) -> VariableType:
        return VariableType.CONTINUOUS
    
    @classproperty
    @override
    def gradient_available(cls) -> bool:
        return False  # No gradient available
    
    @classproperty
    @override
    def optimal_value(cls) -> float | None:
        return None  # Unknown optimal value
    
    @property
    @override
    def optimal_solution(self) -> tuple:
        return None  # Unknown optimal solution
    
    @property
    @override
    def model_default_factors(self) -> dict:
        return {}
    
    @property
    @override
    def model_fixed_factors(self) -> dict:
        return {}
    
    @model_fixed_factors.setter
    def model_fixed_factors(self, value: dict | None) -> None:
         # TODO: figure out if fixed factors should change
        pass 

    @classproperty
    @override
    def model_decision_factors(cls) -> set[str]:
        return {'booking limits'}

    @classproperty
    @override
    def specifications(cls) -> dict[str, dict]:
        return {
            'initial_solution': {
                'description': 'initial booking limits for each fare class',
                'datatype': tuple,
                'default': (3, 3, 3, 3, 3, 3),
            },
            'budget': {
                'description': 'total capacity available',
                'datatype': int,
                'default': 5000,
            },
        }
    
    @property
    @override
    def check_factor_list(self) -> dict[str, Callable]:    
        return {
            'initial_solution': self.check_initial_solution,
            'budget': self.check_budget
        }

    
    @property
    @override
    def dim(self) -> int:
        return len(self.model.factors['booking limits'])
    
    @property
    @override
    def lower_bounds(self) -> tuple:
        return (0,) * self.dim
    
    @property
    @override
    def upper_bounds(self) -> tuple:
        #Booking limits cannot exceed the capacity of the flight
        return (np.inf,) * self.dim
    
    def __init__(self, name: str = 'AIRLINE-1', fixed_factors: dict | None = None, model_fixed_factors: dict | None = None) -> None:
        super().__init__(
            name=name, 
            fixed_factors=fixed_factors, 
            model_fixed_factors=model_fixed_factors,
            model=AirlineRevenueModel
        )
        
    @override
    def vector_to_factor_dict(self, vector: tuple) -> dict:
        return {'booking limits': vector[:]}

    @override
    def factor_dict_to_vector(self, factor_dict: dict) -> tuple:
        return tuple(factor_dict['booking limits'])

    @override
    def response_dict_to_objectives(self, response_dict: dict) -> tuple:
        return (response_dict['revenue'],)

    @override
    def deterministic_objectives_and_gradients(self, _x: tuple) -> tuple:
        det_objectives = (0,)
        det_objectives_gradients = ((0,),)
        return det_objectives, det_objectives_gradients

    @override
    def check_deterministic_constraints(self, x: tuple) -> bool:
        return all(x[j] > 0 for j in range(self.dim))

    @override
    def get_random_solution(self, rand_sol_rng: MRG32k3a) -> tuple:
        return tuple([rand_sol_rng.uniform(0, 10) for _ in range(self.dim)])

    @override
    def get_random_solution(self, rand_sol_rng: MRG32k3a) -> tuple:
        # x = tuple([rand_sol_rng.uniform(-2, 2) for _ in range(self.dim)])
        return tuple(
            rand_sol_rng.mvnormalvariate(
                mean_vec=[0] * self.dim,
                cov=np.eye(self.dim),
                factorized=False,
            )
        )
    

class AirlineRevenueBidPriceProblem(Problem):
    """Finds the optimal set of fare prices to maximize expected revenue."""

    @classproperty
    @override
    def class_name_abbr(cls) -> str:
        return "AIRLINE-2"
    
    @classproperty
    @override
    def class_name(cls) -> str:
        return "Airline Revenue Management Bid Price Problem"
    
    @classproperty
    @override
    def n_objectives(cls) -> int:
        return 1  # Tuple of decision variables
    

    @classproperty
    @override
    def n_stochastic_constraints(cls) -> int:
        return 0  # No stochastic constraints
    
    @classproperty
    @override
    def minmax(cls) -> tuple[int]:
        return (1,)  # Maximize revenue
    
    @classproperty
    @override
    def constraint_type(cls) -> ConstraintType:
       return ConstraintType.DETERMINISTIC
    
    @classproperty
    @override
    def variable_type(cls) -> VariableType:
        return VariableType.CONTINUOUS
    
    @classproperty
    @override
    def gradient_available(cls) -> bool:
        return False  # No gradient available
    
    @classproperty
    @override
    def optimal_value(cls) -> float | None:
        return None  # Unknown optimal value
    
    @property
    @override
    def optimal_solution(self) -> tuple:
        return None  # Unknown optimal solution
    
    @property
    @override
    def model_default_factors(self) -> dict:
        return {}
    
    @property
    @override
    def model_fixed_factors(self) -> dict:
        return {}
    
    @model_fixed_factors.setter
    def model_fixed_factors(self, value: dict | None) -> None:
         # TODO: figure out if fixed factors should change
        pass 

    @classproperty
    @override
    def model_decision_factors(cls) -> set[str]:
        return {'prices'}

    @classproperty
    @override
    def specifications(cls) -> dict[str, dict]:
        return {
            'initial_solution': {
                'description': 'initial prices for each fare class',
                'datatype': tuple,
                'default': (300, 100, 150, 50, 100, 25)
            },
            'budget': {
                'description': 'total capacity available',
                'datatype': int,
                'default': 10,
            },
        }
    
    @property
    @override
    def check_factor_list(self) -> dict[str, Callable]:    
        return {
            'initial_solution': self.check_initial_solution,
            'budget': self.check_budget
        }
    
    @property
    @override
    def dim(self) -> int:
        return len(self.model.factors['prices'])
    
    @property
    @override
    def lower_bounds(self) -> tuple:
        return (0,) * self.dim
    
    @property
    @override
    def upper_bounds(self) -> tuple:
        #TODO: Needs to be changed - has to do with adjacency matrix of flight?
        #Booking limits cannot exceed the capacity of the flight
        return (np.inf,) * self.dim
    
    def __init__(self, name: str = 'AIRLINE-2', fixed_factors: dict | None = None, model_fixed_factors: dict | None = None) -> None:
        super().__init__(
            name=name, 
            fixed_factors=fixed_factors, 
            model_fixed_factors=model_fixed_factors,
            model=AirlineRevenueModel
        )
        
    @override
    def vector_to_factor_dict(self, vector: tuple) -> dict:
        return {'prices': vector[:]}

    @override
    def factor_dict_to_vector(self, factor_dict: dict) -> tuple:
        return tuple(factor_dict['prices'])

    @override
    def response_dict_to_objectives(self, response_dict: dict) -> tuple:
        return (response_dict['revenue'],)

    @override
    def deterministic_objectives_and_gradients(self, _x: tuple) -> tuple:
        det_objectives = (0,)
        det_objectives_gradients = ((0,),)
        return det_objectives, det_objectives_gradients

    @override
    def check_deterministic_constraints(self, x: tuple) -> bool:
        return all(x[j] > 0 for j in range(self.dim))

    @override
    def get_random_solution(self, rand_sol_rng: MRG32k3a) -> tuple:
        return tuple([rand_sol_rng.uniform(0, 10) for _ in range(self.dim)])

    @override
    def get_random_solution(self, rand_sol_rng: MRG32k3a) -> tuple:
        # x = tuple([rand_sol_rng.uniform(-2, 2) for _ in range(self.dim)])
        return tuple(
            rand_sol_rng.mvnormalvariate(
                mean_vec=[0] * self.dim,
                cov=np.eye(self.dim),
                factorized=False,
            )
        )