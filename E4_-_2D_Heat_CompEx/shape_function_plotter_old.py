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
phi_c_grid = 0.25 * (1 + xi_grid) * (1 + eta_grid)

# New figure to plot shape functions on local geometry (xi,eta)
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
''
# Grid as numpy matrix of shape functions in local coordinates at discrete positions of xi and eta in the range -1 to 1
# Transposed because x_values need to be horizontal and y values vertical for matplotlib's "plot_surface()"
xi_discrete = np.asarray([np.arange(-1, 1, 0.1)]).T
eta_discrete = np.asarray([np.arange(-1, 1, 0.1)])
phi_a = ( 0.25 * (1 - xi_discrete) @ (1 - eta_discrete) ).T
phi_b = ( 0.25 * (1 + xi_discrete) @ (1 - eta_discrete) ).T
phi_c = ( 0.25 * (1 + xi_discrete) @ (1 + eta_discrete) ).T
phi_d = ( 0.25 * (1 - xi_discrete) @ (1 + eta_discrete) ).T

'''
x=0.5
y=0.5
a = 0.25 * (1 - x) * (1 - y)
b = 0.25 * (1 + x) * (1 - y) 
c = 0.25 * (1 + x) * (1 + y)
d = 0.25 * (1 - x) * (1 + y)

x_global = 1*a+7*b+6*c+2*d
y_global = 2*a+1*b+8*c+7*d

print("\t", x_global, "\t", y_global)

'''

# Calculate x and y grid values (global coordinates) using grids of shape function values (local coordinates) and node coordinates of the element
x_grid = phi_a*1+phi_b*7+phi_c*6+phi_d*2
y_grid = phi_a*2+phi_b*1+phi_c*8+phi_d*7


# New figure to plot shape functions on global geometry (x,y)
fig_SF_global = plt.figure()

# Plot phi_A
ax = fig_SF_global.add_subplot(1, 1, 1, projection='3d')
surf = ax.plot_surface(x_grid, y_grid, phi_c, cmap=cm.viridis, linewidth=0, antialiased=False)
z_min, z_max = np.min(phi_a) - 0.2, np.max(phi_a) + 0.2
cset = ax.contourf(x_grid, y_grid, phi_a, offset=z_min, colors='darkgray', alpha=0.5)
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

# Calculate determinant of Jacobian as grid
constant = 15/2
factor_xi = 5/4       # 1.0/xi_discrete if det J is independent of xi
factor_eta = -3/2         # 1.0/eta_discrete if det J is independent of eta
det_J_grid = ( (factor_xi * xi_discrete) * (constant + factor_eta * eta_discrete) ).T

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