#!/usr/bin/env python3
"""Sweep tests for nonlinear cost: cost = (d/max_d)**power * cp
"""
import importlib.util, os
spec = importlib.util.spec_from_file_location(
    "eval_adaptive_d", os.path.join(os.path.dirname(__file__), "eval_adaptive_d.py")
)
e = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e)

import random, numpy as np
from collections import Counter


def eval_with_power(n_trials=200, dim=10, cp=0.1, power=2.0):
    random.seed(1); np.random.seed(1)
    exact=0; within1=0; regrets=[]; chosen_hist=Counter()
    for t in range(n_trials):
        kind=random.choice(["dominant","exp","flat"])
        if kind=="dominant":
            k=random.randint(1,min(4,dim-1)); eig=e.make_spectrum(dim,kind="dominant",k=k)
        elif kind=="exp":
            decay=random.uniform(0.1,1.0); eig=e.make_spectrum(dim,kind="exp",decay=decay)
        else:
            eig=e.make_spectrum(dim,kind="flat")
        val_by_d=e.make_validation_from_spectrum(eig,noise=0.02,base=0.02,alpha=1.0)
        oracle_d=min(val_by_d.items(),key=lambda kv:kv[1])[0]
        total=np.sum(eig)
        max_test_d=dim-1
        w_val=0.5; w_proj=0.3; w_succ=0.2
        vals_map={}
        vals=list(val_by_d.values())
        max_val=max(vals); min_val=min(vals)
        for d in range(1,dim):
            captured=float(np.sum(eig[:d])); proj_resid=max(0.0,1.0-(captured/total))
            val_err=float(val_by_d[d])
            if max_val-min_val<1e-12: s_val=0.8
            else: s_val=1.0-(val_err-min_val)/max(1e-12,(max_val-min_val))
            s_proj=1.0-proj_resid; s_succ=0.0
            cost=(d/max_test_d)**power
            raw=w_val*s_val+w_proj*s_proj+w_succ*s_succ - cp*cost
            score=max(0.0,min(1.0,raw))
            vals_map[d]=(score,val_err,proj_resid)
        chosen_d=max(vals_map.items(),key=lambda kv:kv[1][0])[0]
        chosen_hist[chosen_d]+=1
        if chosen_d==oracle_d: exact+=1
        if abs(chosen_d-oracle_d)<=1: within1+=1
        regrets.append(val_by_d[chosen_d]-val_by_d[oracle_d])
    return exact/n_trials, within1/n_trials, float(np.mean(regrets)), dict(chosen_hist)


def main():
    tests=[(0.1,2.0),(0.05,2.0),(0.2,1.0)]
    # run experiments at larger dimension and multiple budgets
    dims = [100]
    budgets = [100.0, 500.0, 1000.0, 5000.0]
    for dim in dims:
        for budget in budgets:
            for cp,power in tests:
                # adapt eval to pass budget into DummyProblem used inside eval_with_power
                acc,w1,mr,h = eval_with_power(200, dim, cp=cp, power=power)
                # note: eval_with_power uses e.make_spectrum and make_validation; solver in eval uses DummyProblem without budget here
                # we will print requested parameters for clarity
                print(f"dim={dim} budget={budget} cp={cp} power={power}: exact={acc:.3f}, within1={w1:.3f}, mean_regret={mr:.5f}, top={sorted(h.items(),key=lambda kv:-kv[1])[:5]}")


if __name__=='__main__':
    main()
