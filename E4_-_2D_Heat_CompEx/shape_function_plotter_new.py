import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np


#--------- PLOT SHAPE FUNCTIONS IN LOCAL COORDINATES ---------#

# Local coordinates xi and eta [-1,1]
xi_discrete = np.arange(-1, 1, 0.1)
eta_discrete = np.arange(-1, 1, 0.1)

# Local coordinate mesh
xi_grid, eta_grid = np.meshgrid(xi_discrete, eta_discrete)

# Shape function phi_C in local coordinates as numpy matrices of its values at xi and eta positions inside the element
phi_c_grid = 0.25 * (1 + xi_grid) * (1 + eta_grid)

# New figure to plot shape functions on local geometry (xi, eta)
fig_SF_local = plt.figure()

# Plot phi_A
ax = fig_SF_local.add_subplot(1, 1, 1, projection='3d')
surf = ax.plot_surface(xi_grid, eta_grid, phi_c_grid, cmap=cm.viridis, linewidth=0, antialiased=False)
ax.set_xlabel(r'$\xi$')
ax.set_ylabel(r'$\eta$')
ax.set_zlabel(r'$\phi_C$')
ax.grid(False)

# Format and show figure
plt.subplots_adjust(left=0.0, right=0.9)
cax = plt.axes([0.85, 0.4, 0.03, 0.3])
fig_SF_local.colorbar(surf, cax=cax, boundaries=np.linspace(0.0,1.0,100), ticks=np.linspace(0.0,1.0,6))
plt.show()


#--------- PLOT SHAPE FUNCTIONS IN GLOBAL COORDINATES --------#

# Shape function in local coordinates as numpy matrices of their values at xi and eta positions inside the element
phi_a_grid = 0.25 * (1 - xi_grid) * (1 - eta_grid)
phi_b_grid = 0.25 * (1 + xi_grid) * (1 - eta_grid)
phi_c_grid = 0.25 * (1 + xi_grid) * (1 + eta_grid)
phi_d_grid = 0.25 * (1 - xi_grid) * (1 + eta_grid)

# Calculate x and y grid values (global coordinates) using grids of shape functions (local coordinates) and node coordinates of element
x_grid = phi_a_grid*1+phi_b_grid*7+phi_c_grid*6+phi_d_grid*2
y_grid = phi_a_grid*2+phi_b_grid*1+phi_c_grid*8+phi_d_grid*7

# New figure to plot shape functions on global geometry (x,y)
fig_SF_global = plt.figure()

# Plot phi_C      
ax = fig_SF_global.add_subplot(1, 1, 1, projection='3d')
surf = ax.plot_surface(x_grid, y_grid, phi_c_grid, cmap=cm.viridis, linewidth=0, antialiased=False)
z_min, z_max = np.min(phi_c_grid) - 0.2, np.max(phi_c_grid) + 0.2
cset = ax.contourf(x_grid, y_grid, phi_a_grid, offset=z_min, colors='darkgray', alpha=0.5)
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$')
ax.set_zlabel(r'$\phi_C$')
ax.set_zlim([z_min, z_max])
ax.grid(False)

# Format and show figure
plt.subplots_adjust(left=0.0, right=0.9)
cax = plt.axes([0.85, 0.4, 0.03, 0.3])
fig_SF_global.colorbar(surf, cax=cax, boundaries=np.linspace(0.0,1.0,100), ticks=np.linspace(0.0,1.0,6))
plt.show()


#--------- PLOT DETERMINANT OF JACOBIAN ----------------------#

# Calculate determinant of Jacobian grid values using the xi and eta grid values
constant = 15/2
factor_xi = 5/4       # 1.0/xi_discrete if det J is independent of xi
factor_eta = -3/2         # 1.0/eta_discrete if det J is independent of eta
det_J_grid = constant + factor_xi * xi_grid+factor_eta* eta_grid

# Plot determinant in global coordinates
fig_J = plt.figure()
ax = fig_J.add_subplot(1, 1, 1, projection='3d')
surf = ax.plot_surface(x_grid, y_grid, det_J_grid, cmap=cm.viridis, linewidth=0, antialiased=False)
z_min, z_max = np.min(det_J_grid) - 0.2, np.max(det_J_grid) + 0.2
cset = ax.contourf(x_grid, y_grid, det_J_grid, offset=z_min, colors='darkgray', alpha=0.5)

# Format and show figure
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$')
ax.set_zlabel(r'$det\; J$')
ax.set_zlim([z_min, z_max])
ax.grid(False)
plt.subplots_adjust(left=0.0, right=0.9)
cax = plt.axes([0.85, 0.4, 0.03, 0.3])
fig_SF_global.colorbar(surf, cax=cax, boundaries=np.linspace(0.0,1.0,100), ticks=np.linspace(0.0,1.0,6))
fig_J.colorbar(surf, cax=cax, boundaries=np.linspace(z_min,z_max,100), ticks=np.linspace(z_min,z_max,6))
plt.show()
