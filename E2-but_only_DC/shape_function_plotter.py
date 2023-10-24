import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np


#--------- PLOT SHAPE FUNCTIONS IN LOCAL COORDINATES ---------#

# Local coordinates xi and eta [-1,1]
xi = np.arange(-1, 1, 0.1)
eta = np.arange(-1, 1, 0.1)

# Local coordinate mesh
xi_grid, eta_grid = np.meshgrid(xi, eta)

# Shape function phi_A in local coordinates
phi_a_grid = 0.25 * (1 - xi_grid) * (1 - eta_grid)

# New figure to plot shape functions on local geometry (xi,eta)
fig_SF_local = plt.figure()

# Plot phi_A
ax = fig_SF_local.add_subplot(1, 1, 1, projection='3d')
surf = ax.plot_surface(xi_grid, eta_grid, phi_a_grid, cmap=cm.viridis, linewidth=0, antialiased=False)
ax.set_xlabel(r'$\xi$')
ax.set_ylabel(r'$\eta$')
ax.set_zlabel(r'$\phi_A$')
ax.grid(False)

# Format and show figure
plt.subplots_adjust(left=0.0, right=0.9)
cax = plt.axes([0.85, 0.4, 0.03, 0.3])
fig_SF_local.colorbar(surf, cax=cax, boundaries=np.linspace(0.0,1.0,100), ticks=np.linspace(0.0,1.0,6))
plt.show()


#--------- PLOT SHAPE FUNCTIONS IN GLOBAL COORDINATES --------#

# Grid as numpy matrix of shape functions in local coordinates at discrete positions of xi and eta in the range -1 to 1
# Transposed because x_values need to be horizontal and y values vertical for matplotlib's "plot_surface()"
xi_discrete = np.asarray([np.arange(-1, 1, 0.1)]).T
eta_discrete = np.asarray([np.arange(-1, 1, 0.1)])
phi_a = ( 0.25 * (1 - xi_discrete) @ (1 - eta_discrete) ).T
phi_b =
phi_c =
phi_d =

# Calculate x and y grid values (global coordinates) using grids of shape function values (local coordinates) and node coordinates of the element
x_grid =
y_grid =

# New figure to plot shape functions on global geometry (x,y)
fig_SF_global = plt.figure()

# Plot phi_A
ax = fig_SF_global.add_subplot(1, 1, 1, projection='3d')
surf = ax.plot_surface(x_grid, y_grid, phi_a, cmap=cm.viridis, linewidth=0, antialiased=False)
z_min, z_max = np.min(phi_a) - 0.2, np.max(phi_a) + 0.2
cset = ax.contourf(x_grid, y_grid, phi_a, offset=z_min, colors='darkgray', alpha=0.5)
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$')
ax.set_zlabel(r'$\phi_A$')
ax.set_zlim([z_min, z_max])
ax.grid(False)

# Format and show figure
plt.subplots_adjust(left=0.0, right=0.9)
cax = plt.axes([0.85, 0.4, 0.03, 0.3])
fig_SF_global.colorbar(surf, cax=cax, boundaries=np.linspace(0.0,1.0,100), ticks=np.linspace(0.0,1.0,6))
plt.show()


#--------- PLOT DETERMINANT OF JACOBIAN ----------------------#

# Calculate determinant of Jacobian as grid
constant =
factor_xi =                 # 1.0/xi_discrete if det J is independent of xi
factor_eta =                # 1.0/eta_discrete if det J is independent of eta
det_J_grid = ( (factor_xi * xi_discrete) @ (constant + factor_eta * eta_discrete) ).T

# Plot determinant in global coordinates
fig_J = plt.figure()
ax = fig_J.add_subplot(1, 1, 1, projection='3d')
surf = ax.plot_surface(x_grid, y_grid, det_J_grid, cmap=cm.viridis, linewidth=0, antialiased=False)
z_min, z_max = np.min(det_J_grid) - 0.2, np.max(det_J_grid) + 0.2
cset = ax.contourf(x_grid, y_grid, det_J_grid, offset=z_min, colors='darkgray', alpha=0.5)

# Format and show figure
ax.set_xlabel(r'$\xi$')
ax.set_ylabel(r'$\eta$')
ax.set_zlabel(r'$det\; J$')
ax.set_zlim([z_min, z_max])
ax.grid(False)
plt.subplots_adjust(left=0.0, right=0.9)
cax = plt.axes([0.85, 0.4, 0.03, 0.3])
fig_SF_global.colorbar(surf, cax=cax, boundaries=np.linspace(0.0,1.0,100), ticks=np.linspace(0.0,1.0,6))
fig_J.colorbar(surf, cax=cax, boundaries=np.linspace(z_min,z_max,100), ticks=np.linspace(z_min,z_max,6))
plt.show()
