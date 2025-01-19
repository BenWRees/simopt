"""
    The following experiment looks at how Models under different Polynomial Bases fit to Different Functions
"""
import matplotlib.pyplot as plt
import numpy as np 

from simopt.solvers.active_subspaces.basis import *
from simopt.solvers.TrustRegion.Models import RandomModel
from simopt.base import Problem

class PlotPolyModel :
    def __init__(self, basis: Basis, problem: Problem, model: RandomModel):
        self._basis = basis
        self._problem = Problem 
        self._model = model

    @property
    def X(self) -> np.ndarray : 
        return self._X

    @property.setter
    def X(self, X:np.ndarray) : 
        self._X = X 

    @property
    def fX(self) -> np.ndarray : 
        return self._fX

    @property.setter
    def fX(self, fX:np.ndarray) : 
        self._fX = fX 

    @property
    def basis(self) -> Basis :
        return self._basis
    
    @property
    def model(self) -> RandomModel :
        return self._model

    #fit the model 
    def construct_model(self, X:np.ndarray, fX: np.ndarray) ->  None: 
        
        #now calculate the fvals
        self._calculate_fvals() 
    

    def _calculate_fvals(self) -> None : 
        fvals = []
        for x in self._X : 
            fx = self._model.local_model_evaluate(x.flatten())
            fvals.append(fx)

        self.fX(np.array(fvals))

    def _model_plot(self) -> plt.axes:
        """plot the local model

        Returns:
            plt.axes: the plot object
        """
        pass

    def _problem_plot(self) -> plt.axes :
        """Plots the response surface of the simulation model within the range of X
        Returns:
            plt.axes: the plot object
        """
        pass


    def show_plots(self) -> None : 
        modelPlot = self._model_plot()
        rsPlot = self._problem_plot() 
        plt.show()


def main() -> None : 
    obj = PlotPolyModel()

    problem = Problem('')

    X = np.random.rand(10,problem.dim)
    fX = problem.model.replicate()
    obj.construct_model(X,fX)
    obj.show_plots()