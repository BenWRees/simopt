# from basis import * 
# from gauss_newton import * 
# from local_linear import * 
# from poly import * 
# from ridge import * 
# from seqlp import * 
# from subspace import * 


"""
    In this module we want to provide methods for calculating the active subspace and a ridge approximation for use in the simopt solvers. 
    This will mean only having the polyridge file and the subspace file available outside of this package 

    We want to move the basis module out of this package to be accessed easily by other solvers - but for the time being it is fine in this package. 

    Need to rewrite how functions handle domains and how they handle function calls to work with the simopt library. 
    Can copy over the Domain package and rewrite any function calls to work with simopt.

"""