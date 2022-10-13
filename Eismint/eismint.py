# 
# Alternating direction implicit scheme for Eismint
# - Adopted from Eismint.f fortran file given by Guðfinna Th. Aðalgeirsdóttir.
# - Current version:
# 
# 
# Governing equation is is the continuity equation:
#
# 	ðh/ðt = b^dot (x,y,t) - Del(q) 
#
# 	h = surface elevation
#	ðh/ðt = change in surface elevation
#	b^dot = mass balance rate
# 	q = mass flux
#	Del(q) = gradient of mass flux ( direction greatest mass flux )
#
#


#
# Importing packages
#
from scipy.linalg import solve_banded #Solving triangular matrices
#from scipy.sparse import spdiags # sparse diagonals
import scipy.sparse as sp
import scipy.sparse.linalg  as la

import numpy as np
import time
import matplotlib.pyplot as plt
import sys

#
# Parameter and constant inputs
#

n_time_steps=20000

dt = 5

Lx = 50
Ly = 50

dx = 50
dy = 50

A = 1e-16
g = 9.8
rho = 8927.1000 / g
n_Glen = 3
m_Weertman=3.0

gamma = 2*A*pow(rho*g,n_Glen)/(n_Glen+2)


#
# Data initialization
#

surf   = np.ones((Lx, Ly))*10 # Surface elevation
surf[1,1] = 15 
bed = np.zeros((Lx, Ly)) # Bedrock elevation

surf_bal = np.ones((Lx, Ly))*0 # Surface balance array
bed_bal = np.zeros((Lx, Ly)) # Bed balance array

Dx = np.zeros((Lx, Ly)) # Diffusion values for x-dir
Dy = np.zeros((Lx, Ly)) # Diffusion values for x-dir


# 
# Compute the timestep
# 
#	- Please note that solving the tridiagonal matrix is done with the scipy package.
#
# Discretization
#	see article: https://tc.copernicus.org/articles/7/229/2013/tc-7-229-2013.pdf
#
#


#
# Calulate directional diffusional values without upstream correction
#
def calculate_D(Dx, Dy, surf, bed, gamma, n_Glen, dx, dy):
	#print("Calculating D")
	n_half = (n_Glen-1)/2
	n_add = n_Glen+2

	idx = 1 / dx
	idy = 1 / dy
	idt = 1 / dt

	for j in range(1, np.shape(Dx)[1] -1 ):
		for i in range(1, np.shape(Dx)[0]-1 ):
			# generating x-direction components
			dhdxx=(surf[i+1,j]-surf[i,j])*idx
			dhdyx=(surf[i,j+1]-surf[i,j-1]+surf[i+1,j+1]-surf[i+1,j-1])*0.25*idy
			alfax=(dhdxx*dhdxx+dhdyx*dhdyx) # DelS^i 
			thickx=(surf[i+1,j]-bed[i+1,j]+surf[i,j]-bed[i,j])*0.5

			Dx[i,j]=gamma*pow(thickx,n_add)*pow(alfax, n_half) # Directly from article

			# generating y-direction components
			dhdxy=(surf[i+1,j]-surf[i-1,j]+surf[i+1,j+1]-surf[i-1,j-1])*0.25*idy
			dhdyy=(surf[i,j+1]-surf[i,j])*idx
			alfay=(dhdyy*dhdyy+dhdxy*dhdxy) # DelS^i 
			thicky=(surf[i,j+1]-bed[i,j+1]+surf[i,j]-bed[i,j])*0.5

			Dy[i,j]=gamma*pow(thicky,n_add)*pow(alfay, n_half)

	return Dx,Dy


#
# Run a single step 
#

def do_step(Dx, Dy, surf, bed, gamma, n_Glen, dx, dy, dt, surf_bal, bed_bal):
	# Setting constants
	s_old = surf.copy()
	idxx = 1 / (dx * dx)
	idyy = 1 / (dy * dy)
	idt = 1 / dt
	
	
	
	# Generating diffusion values
	Dx,Dy = calculate_D(Dx,Dy, surf, bed, gamma, n_Glen, dx, dy)
	# Solving system for x-direction
	for j in range(1, np.shape(surf)[1]-1 ):
		LHS = sp.lil_matrix(surf.shape)
		RHS = sp.lil_matrix((surf.shape[0], 1))

		for i in range(1, np.shape(surf)[0]-1 ):
			LHS[i, i-1] = -Dx[i-1, j] * idxx
			LHS[i, i] =  (Dx[i-1,j] + Dx[i,j]) * idxx + idt
			LHS[i, i+1] = -Dx[i,j] * idxx

			ypart = \
				(Dy[i, j] * s_old[i,j+1] \
				- (Dy[i, j] + Dy[i, j-1]) * s_old[i,j] \
				+  Dy[i, j-1] * s_old[i,j-1] )*idyy
			RHS[i] = surf_bal[i,j]-bed_bal[i,j]+s_old[i,j]*idt+ypart
		
		LHS[0,0] = 1 # end values
		LHS[-1,-1] = 1 # end values
		#print( LHS.toarray() )
		x = la.spsolve(LHS.tocsr(), RHS.tocsr()) # solving each column with sparse matrices
		surf[2:(-1), j] = x[2:(-1)]
		#print( x )
	
	#breakpoint() # for debugging
		
	# Generating diffusion values after updated halfstep
	Dx,Dy = calculate_D(Dx,Dy, surf, bed, gamma, n_Glen, dx, dy)
	# Solving system for y-direction
	for i in range(1, np.shape(surf)[0]-1 ):

		LHS = sp.lil_matrix(surf.shape)
		RHS = sp.lil_matrix((surf.shape[0], 1))

		for j in range(1, np.shape(surf)[1]-1 ):
			LHS[j, j-1] = -Dy[i, j-1] * idyy
			LHS[j, j] =  (Dy[i,j-1] + Dy[i,j]) * idyy + idt
			LHS[j, j+1] = -Dy[i,j] * idyy

			xpart = \
				(Dx[i, j] * s_old[i+1,j] \
				- (Dx[i-1, j] + Dx[i, j]) * s_old[i,j] \
				+  Dx[i-1, j] * s_old[i-1,j] )*idyy
			RHS[j] = surf_bal[i,j]-bed_bal[i,j]+s_old[i,j]*idt+xpart
		
		LHS[0,0] = 1 # end values
		LHS[-1,-1] = 1 # end values
		#print( LHS.toarray() )
		x = la.spsolve(LHS.tocsr(), RHS.tocsr()) # solving each column with sparse matrices
		surf[i, 2:(-1)] = x[2:(-1)]
		#print( x )
	
	residual = np.absolute(surf - s_old).sum()
	
	breakpoint() # for debugging
	
	return surf, residual # return the surface only, everything else doesn't change





#
# Running timesteps
#

for i in range(0,2):

	surf, residual = do_step(Dx,Dy, surf, bed, gamma, n_Glen, dx, dy, dt, surf_bal, bed_bal)
	
	# if the surface is below bedrock then it's equal to bedrock
	surf[surf <= bed] = bed[surf <= bed]
	# plot
	fig, ax = plt.subplots()

	ax.imshow(surf)

	plt.show()





#
# Clean up
# 
