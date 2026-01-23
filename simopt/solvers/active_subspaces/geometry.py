"""Geometry module for active subspaces.

This module provides the Geometry class used by OMoRF solver.
"""

from simopt.solvers.TrustRegion.Geometry import OMoRFGeometry

# Expose OMoRFGeometry as Geometry for backwards compatibility
Geometry = OMoRFGeometry

__all__ = ["Geometry", "OMoRFGeometry"]
