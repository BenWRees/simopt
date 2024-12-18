# type: ignore
"""Ridge function approximation from function values"""
# (c) 2017 Jeffrey M. Hokanson (jeffrey@hokanson.us)

from __future__ import print_function
from matplotlib.axes import Axes
import numpy as np

from .poly import BaseFunction
from .subspace import SubspaceBasedDimensionReduction

#from .pgf
from matplotlib.path import Path
from copy import deepcopy


class PGF:
	def __init__(self):
		self.column_names = []
		self.columns = []

	def add(self, name, column):
		if len(self.columns) > 1:
			assert len(self.columns[0]) == len(column)

		self.columns.append(deepcopy(column))
		self.column_names.append(name)

	def keys(self):
		return self.column_names

	def __getitem__(self, key):
		i = self.column_names.index(key)
		return self.columns[i]

	def write(self, filename):
		f = open(filename,'w')

		for name in self.column_names:
			f.write(name + '\t')
		f.write("\n")		

		for j in range(len(self.columns[0])):
			for col in self.columns:
				f.write("{}\t".format(float(col[j])))
			f.write("\n")

		f.close()

	def read(self, filename):
		with open(filename,'r') as f:
			for i, line in enumerate(f):
				# Remove the newline and trailing tab if present
				line = line.replace('\t\n','').replace('\n','')
				if i == 0:
					self.column_names = line.split('\t')
					self.columns = [ [] for name in self.column_names]
				else:
					cols = line.split('\t')
					for j, col in enumerate(cols):
						self.columns[j].append(float(col))


def save_contour(fname, cs, fmt = 'matlab', simplify = 1e-3, **kwargs):
	""" Save a contour plot to a file for pgfplots

	Additional arguments are passed to iter_segements
	Important, simplify = True will remove invisible points
	"""

	def write_path_matlab(fout, x_vec, y_vec, z):
		# Now dump this data back out
		# Header is level followed by number of rows
		fout.write('%15.15e\t%15d\n' % (z, len(x_vec)))
		for x, y in zip(x_vec, y_vec):
			fout.write("%15.15e\t%15.15e\n" % (x,y))

	def write_path_prepared(fout, x_vec, y_vec, z):
		fout.write("%15.15e\t%15.15e\t%15.15e\n" % (x_vec,y_vec,z))
		fout.write("\t\t\t\n")

	if fmt == 'matlab':
		write_path = write_path_matlab
	elif fmt == 'prepared':
		write_path = write_path_prepared
	else:
		raise NotImplementedError

	with open(fname, 'w') as fout:
		for col, z in zip(cs.collections, cs.levels):
			for path in col.get_paths():
				path.simplify_threshold = simplify
				x_vec = []
				y_vec = []
				for i, ((x,y), code) in enumerate(path.iter_segments(simplify = True)):
					if code == Path.MOVETO:
						if len(x_vec) !=0:
							write_path(fout, x_vec, y_vec, z)
							x_vec = []
							y_vec = []
						x_vec.append(x)
						y_vec.append(y)
					
					elif code == Path.LINETO:
						x_vec.append(x)
						y_vec.append(y)

					elif code == Path.CLOSEPOLY:
						x_vec.append(x_vec[0])
						y_vec.append(y_vec[0])
					else:
						print("received code", code)

				write_path(fout, x_vec, y_vec, z)

class RidgeFunction(BaseFunction, SubspaceBasedDimensionReduction):
	# @property
	# def U(self):
	# 	return self._U

	def shadow_plot(self, X = None, fX = None, dim: int | None = None, U = None, ax = 'auto', pgfname = None):
		if dim is None and U is not None:
			dim = U.shape[1]
		else:
			assert dim == U.shape[1]

		ax = SubspaceBasedDimensionReduction.shadow_plot(self, X = X, fX = fX, dim = dim, ax = ax, pgfname = pgfname)

		# Draw the response surface
		if dim == 1:
			Y = np.dot(U.T, X.T).T
			lb = np.min(Y)
			ub = np.max(Y)
			
			xx = np.linspace(lb, ub, 500)
			Uxx = np.hstack([U*xxi for xxi in xx]).T
			yy = self.eval(Uxx)

			if ax is not None and isinstance(ax, Axes):
				ax.plot(xx, yy, 'r-')

			if pgfname is not None:	
				pgfname2 = pgfname[:pgfname.rfind('.')] + '_response' + pgfname[pgfname.rfind('.'):]
				pgf = PGF()
				pgf.add('x', xx)
				pgf.add('fx', yy )
				pgf.write(pgfname2)
					

		elif dim == 2 and isinstance(ax, Axes):	
			Y = np.dot(U.T, X.T).T
			lb0 = np.min(Y[:,0])
			ub0 = np.max(Y[:,0])

			lb1 = np.min(Y[:,1])
			ub1 = np.max(Y[:,1])

			# Constuct mesh on the domain
			xx0 = np.linspace(lb0, ub0, 50)
			xx1 = np.linspace(lb1, ub1, 50)
			XX0, XX1 = np.meshgrid(xx0, xx1)
			UXX = np.vstack([XX0.flatten(), XX1.flatten()])
			XX = np.dot(U, UXX).T
			YY = self.eval(XX).reshape(XX0.shape)
			
			ax.contour(xx0, xx1, YY, 
				levels = np.linspace(np.min(fX), np.max(fX), 20), 
				vmin = np.min(fX), vmax = np.max(fX),
				linewidths = 0.5)
		

		else: 
			raise NotImplementedError("Cannot draw shadow plots in more than two dimensions")	

		return ax		


