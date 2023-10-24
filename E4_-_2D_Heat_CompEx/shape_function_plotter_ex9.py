import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np


#--------- PLOT SHAPE FUNCTIONS IN LOCAL COORDINATES ---------#

# Local coordinates xi and eta [-1,1]
xi_discrete = np.arange(-1, 1, 0.1)
eta_discrete = np.arange(-1, 1, 0.1)

# Local coordinate mesh
xi_grid, eta_grid = np.meshgrid(xi_discrete, eta_discrete)

L_1_eta = (eta_grid/2)*(eta_grid-1)
L_1_xi =(xi_grid/2)*(xi_grid-1)
L0_eta = (eta_grid+1)*(1-eta_grid)
L0_xi = (xi_grid+1)*(1-xi_grid)
L1_eta = (eta_grid/2)*(eta_grid+1)
L1_xi = (xi_grid/2)*(xi_grid+1)

# Shape function phi_C in local coordinates as numpy matrices of its values at xi and eta positions inside the element
phi_c_grid = L1_eta * L1_xi 
phi_g_grid = L1_eta * L0_xi 
phi_i_grid = L0_eta * L0_xi 
# New figure to plot shape functions on local geometry (xi, eta)
fig_SF_local = plt.figure()

# Plot phi_I
ax = fig_SF_local.add_subplot(1, 1, 1, projection='3d')
surf = ax.plot_surface(xi_grid, eta_grid, phi_i_grid, cmap=cm.viridis, linewidth=0, antialiased=False)
ax.set_xlabel(r'$\xi$')
ax.set_ylabel(r'$\eta$')
ax.set_zlabel(r'$\phi_I$')
ax.grid(False)

# Format and show figure
#plt.subplots_adjust(left=0.0, right=0.9)
cax = plt.axes([0.85, 0.4, 0.03, 0.3])
fig_SF_local.colorbar(surf, cax=cax, boundaries=np.linspace(0.0,1.0,100), ticks=np.linspace(0.0,1.0,6))
plt.show()


#--------- PLOT SHAPE FUNCTIONS IN GLOBAL COORDINATES --------#

# Shape function in local coordinates as numpy matrices of their values at xi and eta positions inside the element
phi_a_grid = L_1_eta * L_1_xi
phi_b_grid = L_1_eta * L1_xi
phi_c_grid = L1_eta * L1_xi
phi_d_grid = L1_eta * L_1_xi
phi_e_grid = L_1_eta * L0_xi
phi_f_grid = L0_eta * L1_xi
phi_g_grid = L1_eta * L0_xi
phi_h_grid = L0_eta * L_1_xi
phi_i_grid = L0_eta * L0_xi

# Calculate x and y grid values (global coordinates) using grids of shape functions (local coordinates) and node coordinates of element
x_grid = phi_a_grid*1+phi_b_grid*7+phi_c_grid*6+phi_d_grid*2+phi_e_grid*4.5+phi_f_grid*7.5+phi_g_grid*4+phi_h_grid*2.5 +phi_i_grid*5
y_grid = phi_a_grid*3+phi_b_grid*1+phi_c_grid*8+phi_d_grid*7+phi_e_grid*2.5+phi_f_grid*5+phi_g_grid*8+phi_h_grid*5+phi_i_grid*5

# New figure to plot shape functions on global geometry (x,y)
fig_SF_global = plt.figure()

# Plot phi_i
ax = fig_SF_global.add_subplot(1, 1, 1, projection='3d')
surf = ax.plot_surface(x_grid, y_grid, phi_i_grid, cmap=cm.viridis, linewidth=0, antialiased=False)
z_min, z_max = np.min(phi_i_grid) - 0.2, np.max(phi_i_grid) + 0.2
cset = ax.contourf(x_grid, y_grid, phi_i_grid, offset=z_min, colors='darkgray', alpha=0.5)
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$')
ax.set_zlabel(r'$\phi_I$')
ax.set_zlim([z_min, z_max])
ax.grid(False)

# Format and show figure
plt.subplots_adjust(left=0.0, right=0.9)
cax = plt.axes([0.85, 0.4, 0.03, 0.3])
fig_SF_global.colorbar(surf, cax=cax, boundaries=np.linspace(0.0,1.0,100), ticks=np.linspace(0.0,1.0,6))
plt.show()