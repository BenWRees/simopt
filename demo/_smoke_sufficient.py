import sys
sys.path.append('.')
import numpy as np
import simopt.solvers.astromorf as ast
func = ast.ASTROMORF.evaluate_and_score_candidate_dimensions

class DummyProblem:
    def __init__(self, dim, budget=1000.0):
        self.dim = dim
        self.factors = {"budget": float(budget)}

class Dummy:
    pass

s = Dummy()
s.problem = DummyProblem(dim=50, budget=500.0)
s.max_d = max(1, s.problem.dim - 1)
s.previous_model_information = []
s.gradient_eigenvalues = [np.concatenate([np.linspace(5.0,1.0,3), np.ones(47)*0.1])]
s.last_validation_by_d = {d: 0.02 + 0.01*(1.0 - (np.sum(s.gradient_eigenvalues[0][:d])/np.sum(s.gradient_eigenvalues[0]))) for d in range(1, s.problem.dim)}

s.cost_penalty = 0.05
s.cost_power = 2.0
s.budget_alpha = 0.5
s.baseline_budget = 1000.0
s.prefer_small_sufficient = True
s.sufficient_score_tol = 0.02

scores = func(s)
max_score = max(m['score'] for m in scores.values())
print('max_score=', max_score)
print('s._last_sufficient_candidates=', getattr(s, '_last_sufficient_candidates', None))
print('smallest sufficient =', min(getattr(s, '_last_sufficient_candidates', [None])))
