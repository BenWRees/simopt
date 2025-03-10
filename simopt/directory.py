#!/usr/bin/env python
"""Provide dictionary directories listing solvers, problems, and models."""

from __future__ import annotations

# Import the base classes
from simopt.base import Model, Problem, Solver

# Import Models
from simopt.models.amusementpark import AmusementPark, AmusementParkMinDepart
from simopt.models.chessmm import ChessAvgDifference, ChessMatchmaking
from simopt.models.cntnv import CntNV, CntNVMaxProfit
from simopt.models.contam import (
    Contamination,
    ContaminationTotalCostCont,
    ContaminationTotalCostDisc,
)
from simopt.models.dualsourcing import DualSourcing, DualSourcingMinCost
from simopt.models.dynamnews import DynamNews, DynamNewsMaxProfit
from simopt.models.example import ExampleModel, ExampleProblem
from simopt.models.facilitysizing import (
    FacilitySize,
    FacilitySizingMaxService,
    FacilitySizingTotalCost,
)
from simopt.models.fixedsan import FixedSAN, FixedSANLongestPath
from simopt.models.hotel import Hotel, HotelRevenue
from simopt.models.ironore import IronOre, IronOreMaxRev, IronOreMaxRevCnt
from simopt.models.mm1queue import MM1MinMeanSojournTime, MM1Queue
from simopt.models.network import Network, NetworkMinTotalCost
from simopt.models.paramesti import ParamEstiMaxLogLik, ParameterEstimation
from simopt.models.rmitd import RMITD, RMITDMaxRevenue
from simopt.models.san import SAN, SANLongestPath
from simopt.models.sscont import SSCont, SSContMinCost
from simopt.models.tableallocation import TableAllocation, TableAllocationMaxRev
from simopt.models.simple_function import SimpleFunctionModel, SimpleFunctionProblem
from simopt.models.Gross_model import GrossModel, GrossProblem

# Import Solvers
from simopt.solvers.adam import ADAM
from simopt.solvers.aloe import ALOE
from simopt.solvers.astrodf import ASTRODF
from simopt.solvers.neldmd import NelderMead
from simopt.solvers.randomsearch import RandomSearch
from simopt.solvers.spsa import SPSA
from simopt.solvers.strong import STRONG
from simopt.solvers.TrustRegion.Models import TrustRegion, OMoRF
from simopt.solvers.SGD import SGD
from simopt.solvers.kiefer_wolfowitz import KieferWolfowitz
from simopt.solvers.robbins_monro import RobbinsMonro
from simopt.solvers.mirror_descent import Mirror_Descent
from simopt.solvers.ASTROMoRF import ASTROMoRF

# directory dictionaries
solver_directory: dict[str, type[Solver]] = {
    "ASTRODF": ASTRODF,
    "RNDSRCH": RandomSearch,
    "NELDMD": NelderMead,
    "STRONG": STRONG,
    "SPSA": SPSA,
    "ADAM": ADAM,
    "ALOE": ALOE,
    "TRUSTREGION": TrustRegion,
    "SGD": SGD,
    "KIEFERWOLFOWITZ": KieferWolfowitz,
    "ROBBINSMONRO": RobbinsMonro, 
    "MIRRORDESCENT": Mirror_Descent,
    "OMoRF": OMoRF,
    "ASTROMoRF": ASTROMoRF
}

solver_unabbreviated_directory: dict[str, type[Solver]] = {
    "ASTRO-DF (SBCN)": ASTRODF,
    "Random Search (SSMN)": RandomSearch,
    "Nelder-Mead (SBCN)": NelderMead,
    "STRONG (SBCN)": STRONG,
    "SPSA (SBCN)": SPSA,
    "ADAM (SBCN)": ADAM,
    "ALOE (SBCN)": ALOE,
    "Trust Region Solver": TrustRegion,
    "Stochastic Gradient Descent": SGD,
    "Kiefer-Wolfowitz Algorithm": KieferWolfowitz, 
    "Robbins-Monro Root-Finding": RobbinsMonro,
    "Stochastic Mirror Descent": Mirror_Descent,
    "Optimisation by Moving Ridge Functions": OMoRF,
    "Adaptive Sampling by Moving Ridge Functions": ASTROMoRF
}

problem_directory: dict[str, type[Problem]] = {
    "EXAMPLE-1": ExampleProblem,
    "CNTNEWS-1": CntNVMaxProfit,
    "MM1-1": MM1MinMeanSojournTime,
    "FACSIZE-1": FacilitySizingTotalCost,
    "FACSIZE-2": FacilitySizingMaxService,
    "RMITD-1": RMITDMaxRevenue,
    "SSCONT-1": SSContMinCost,
    "IRONORE-1": IronOreMaxRev,
    "IRONORECONT-1": IronOreMaxRevCnt,
    "DYNAMNEWS-1": DynamNewsMaxProfit,
    "DUALSOURCING-1": DualSourcingMinCost,
    "CONTAM-1": ContaminationTotalCostDisc,
    "CONTAM-2": ContaminationTotalCostCont,
    "CHESS-1": ChessAvgDifference,
    "SAN-1": SANLongestPath,
    "HOTEL-1": HotelRevenue,
    "TABLEALLOCATION-1": TableAllocationMaxRev,
    "PARAMESTI-1": ParamEstiMaxLogLik,
    "FIXEDSAN-1": FixedSANLongestPath,
    "NETWORK-1": NetworkMinTotalCost,
    "AMUSEMENTPARK-1": AmusementParkMinDepart,
    "SIMPLEFUNC-1": SimpleFunctionProblem,
    "GROSS-1": GrossProblem
}

problem_unabbreviated_directory: dict[str, type[Problem]] = {
    "Min Deterministic Function + Noise (SUCG)": ExampleProblem,
    "Max Profit for Continuous Newsvendor (SBCG)": CntNVMaxProfit,
    "Min Mean Sojourn Time for MM1 Queue (SBCG)": MM1MinMeanSojournTime,
    "Min Total Cost for Facility Sizing (SSCG)": FacilitySizingTotalCost,
    "Max Service for Facility Sizing (SDCN)": FacilitySizingMaxService,
    "Max Revenue for Revenue Management Temporal Demand (SDDN)": RMITDMaxRevenue,
    "Min Total Cost for (s, S) Inventory (SBCN)": SSContMinCost,
    "Max Revenue for Iron Ore (SBDN)": IronOreMaxRev,
    "Max Revenue for Continuous Iron Ore (SBCN)": IronOreMaxRevCnt,
    "Max Profit for Dynamic Newsvendor (SBDN)": DynamNewsMaxProfit,
    "Min Cost for Dual Sourcing (SBDN)": DualSourcingMinCost,
    "Min Total Cost for Discrete Contamination (SSDN)": ContaminationTotalCostDisc,
    "Min Total Cost for Continuous Contamination (SSCN)": ContaminationTotalCostCont,
    "Min Avg Difference for Chess Matchmaking (SSCN)": ChessAvgDifference,
    "Min Mean Longest Path for Stochastic Activity Network (SBCG)": SANLongestPath,
    "Max Revenue for Hotel Booking (SBDN)": HotelRevenue,
    "Max Revenue for Restaurant Table Allocation (SDDN)": TableAllocationMaxRev,
    "Max Log Likelihood for Gamma Parameter Estimation (SBCN)": ParamEstiMaxLogLik,
    "Min Mean Longest Path for Fixed Stochastic Activity Network (SBCG)": FixedSANLongestPath,
    "Min Total Cost for Communication Networks System (SDCN)": NetworkMinTotalCost,
    "Min Total Departed Visitors for Amusement Park (SDDN)": AmusementParkMinDepart,
    "Minimise a Deterministic Function with Additive Noise": SimpleFunctionProblem,
    "Minimise a Smooth Continuous Function as seen in Gross": GrossProblem
}
model_directory: dict[str, type[Model]] = {
    "EXAMPLE": ExampleModel,
    "CNTNEWS": CntNV,
    "MM1": MM1Queue,
    "FACSIZE": FacilitySize,
    "RMITD": RMITD,
    "SSCONT": SSCont,
    "IRONORE": IronOre,
    "DYNAMNEWS": DynamNews,
    "DUALSOURCING": DualSourcing,
    "CONTAM": Contamination,
    "CHESS": ChessMatchmaking,
    "SAN": SAN,
    "HOTEL": Hotel,
    "TABLEALLOCATION": TableAllocation,
    "PARAMESTI": ParameterEstimation,
    "FIXEDSAN": FixedSAN,
    "NETWORK": Network,
    "AMUSEMENTPARK": AmusementPark,
    "SIMEPLFUNC": SimpleFunctionModel,
    "GROSS": GrossModel,
}
model_problem_unabbreviated_directory: dict[str, str] = {
    "Min Deterministic Function + Noise (SUCG)": "EXAMPLE",
    "Max Profit for Continuous Newsvendor (SBCG)": "CNTNEWS",
    "Min Mean Sojourn Time for MM1 Queue (SBCG)": "MM1",
    "Min Total Cost for Facility Sizing (SSCG)": "FACSIZE",
    "Max Service for Facility Sizing (SDCN)": "FACSIZE",
    "Max Revenue for Revenue Management Temporal Demand (SDDN)": "RMITD",
    "Min Total Cost for (s, S) Inventory (SBCN)": "SSCONT",
    "Max Revenue for Iron Ore (SBDN)": "IRONORE",
    "Max Revenue for Continuous Iron Ore (SBCN)": "IRONORE",
    "Max Profit for Dynamic Newsvendor (SBDN)": "DYNAMNEWS",
    "Min Cost for Dual Sourcing (SBDN)": "DUALSOURCING",
    "Min Total Cost for Discrete Contamination (SSDN)": "CONTAM",
    "Min Total Cost for Continuous Contamination (SSCN)": "CONTAM",
    "Min Avg Difference for Chess Matchmaking (SSCN)": "CHESS",
    "Min Mean Longest Path for Stochastic Activity Network (SBCG)": "SAN",
    "Max Revenue for Hotel Booking (SBDN)": "HOTEL",
    "Max Revenue for Restaurant Table Allocation (SDDN)": "TABLEALLOCATION",
    "Max Log Likelihood for Gamma Parameter Estimation (SBCN)": "PARAMESTI",
    "Min Mean Longest Path for Fixed Stochastic Activity Network (SBCG)": "FIXEDSAN",
    "Min Total Cost for Communication Networks System (SDCN)": "NETWORK",
    "Min Total Departed Visitors for Amusement Park (SDDN)": "AMUSEMENTPARK",
    "Minimise a Deterministic Function with Additive Noise": "SIMPLEFUNC",
    "Minimise a Smooth Continuous Function as seen in Gross": "GROSS"
}
model_problem_class_directory: dict[str, type[Model]] = {
    "Min Deterministic Function + Noise (SUCG)": ExampleModel,
    "Max Profit for Continuous Newsvendor (SBCG)": CntNV,
    "Min Mean Sojourn Time for MM1 Queue (SBCG)": MM1Queue,
    "Min Total Cost for Facility Sizing (SSCG)": FacilitySize,
    "Max Service for Facility Sizing (SDCN)": FacilitySize,
    "Max Revenue for Revenue Management Temporal Demand (SDDN)": RMITD,
    "Min Total Cost for (s, S) Inventory (SBCN)": SSCont,
    "Max Revenue for Iron Ore (SBDN)": IronOre,
    "Max Revenue for Continuous Iron Ore (SBCN)": IronOre,
    "Max Profit for Dynamic Newsvendor (SBDN)": DynamNews,
    "Min Cost for Dual Sourcing (SBDN)": DualSourcing,
    "Min Total Cost for Discrete Contamination (SSDN)": Contamination,
    "Min Total Cost for Continuous Contamination (SSCN)": Contamination,
    "Min Avg Difference for Chess Matchmaking (SSCN)": ChessMatchmaking,
    "Min Mean Longest Path for Stochastic Activity Network (SBCG)": SAN,
    "Max Revenue for Hotel Booking (SBDN)": Hotel,
    "Max Revenue for Restaurant Table Allocation (SDDN)": TableAllocation,
    "Max Log Likelihood for Gamma Parameter Estimation (SBCN)": ParameterEstimation,
    "Min Mean Longest Path for Fixed Stochastic Activity Network (SBCG)": FixedSAN,
    "Min Total Cost for Communication Networks System (SDCN)": Network,
    "Min Total Departed Visitors for Amusement Park (SDDN)": AmusementPark,
    "Minimise a Deterministic Function with Additive Noise": SimpleFunctionModel,
    "Minimise a Smooth Continuous Function as seen in Gross": GrossModel
}
model_unabbreviated_directory: dict[str, type[Model]] = {
    "Deterministic Function + Noise": ExampleModel,
    "Continuous Newsvendor": CntNV,
    "MM1 Queue": MM1Queue,
    "Facility Sizing": FacilitySize,
    "Revenue Management Temporal Demand": RMITD,
    "(s, S) Inventory": SSCont,
    "Iron Ore": IronOre,
    "Dynamic Newsvendor": DynamNews,
    "Dual Sourcing": DualSourcing,
    "Contamination": Contamination,
    "Chess Matchmaking": ChessMatchmaking,
    "Stochastic Activity Network": SAN,
    "Hotel Booking": Hotel,
    "Restaurant Table Allocation": TableAllocation,
    "Gamma Parameter Estimation": ParameterEstimation,
    "Fixed Stochastic Activity Network": FixedSAN,
    "Communication Networks System": Network,
    "Amusement Park (SDDN)": AmusementPark,
    "Deterministic Function with Additive Noise": SimpleFunctionModel,
    "Minimise a Smooth Continuous Function as seen in Gross": GrossModel,
}
