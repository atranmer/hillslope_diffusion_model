#!/usr/bin/env python
# coding: utf-8

# # 1D Diffusion Model

# Develop a hillslope diffusion model.
# Assume constant diffusivity.
# Use regular grid.
# Uses step function as initial profile.
# It has fixed BCs.

# This is the diffusion equation:
# 
# $$ \frac{\partial z}{\partial t} = D\frac{\partial^2 z}{\partial x^2} $$
# 

# Where $z$ is elevation, $t$ is time, $x$ is the spatial dimension, and $D$ is hillslope diffusivity.

# The discretized form of the diffusivity model that we will solve is
# 
# $$ z^{t+1}_x = z^t_x + {D \Delta t \over \Delta x^2} (z^t_{x+1} - 2z^t_x + z^t_{x-1}) $$
# 

# This is the FTCS discretization scheme from Slingerland and Kump (2011).

# We will use two libraries numPy and Matplotlib that are external libraries to Python.

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt 


# Start by setting two fixed model parameters, the diffusivity and the size of the model domain.

# In[ ]:


D = 100
Lx = 300


# Set up the model gridusing NumPy arrays.

# In[ ]:


dx = 0.5
x = np.arange(start=0, stop=Lx, step=dx)
nx = len(x)


# Set the initial conditions for the model. 
# The elevation $z$ is a step function with a high value on the left, a low value on the right, and a cliff at the center of the domain. 

# In[ ]:


z = np.zeros_like(x)
z_hi = 500.0
z_lo = 0.0
z[x <= Lx/2] = z_hi
z[x > Lx/2] = z_lo


# Plot the intial domain of the hillslope.

# In[ ]:


plt.figure()
plt.plot(x,z,"g")
plt.xlabel("x")
plt.ylabel("z, (meters)")
plt.title("Initial hillslope profile")


# Set the number of time steps in the model. 
# Calculate a stable time step using a stability criterion.

# In[ ]:


nt = 5000
dt = 0.5 * dx**2 / D


# Loop over the time steps of the model, solving the diffusion equation using the FTCS scheme descrined above. 
# Use array operation on the variable $z$.

# In[ ]:


for _ in range(0, nt):
	z[1:-1] += D * dt / dx ** 2 * (z[:-2] - 2*z[1:-1] + z[2:])


# Plot the results.

# In[ ]:


plt.figure()
plt.plot(x,z,"b")
plt.xlabel("x")
plt.ylabel("z, (meters)")
plt.title("Final hillslope profile")

