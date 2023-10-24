#----------ISD | TU Braunschweig----------#
#-----------Beethovenstraße 51------------#
#-----------38106 Braunschweig------------#
#-----------------------------------------#

import numpy as np
import matplotlib.pyplot as plt

from preprocessing.input_data import process_input_data
from FE_functions import build_system_matrix, build_load_vector, apply_neumann_boundary_conditions, apply_dirichlet_boundary_conditions, calculate_resultants
from postprocessing.plot_primary_unknowns import plot_primary_unknowns
from postprocessing.plot_resultants import plot_resultants
from postprocessing.export_values_to_csv import export_primary_unknowns_at_nodes, export_resultants_at_nodes_of_elements


#------------------------------------------------------------#
#------                  PREPROCESSING                 ------#
#------------------------------------------------------------#
# The necessary input for the analysis is loaded in this part.
# Therefore, we need to read the input data files and process them in order to generate lists and dictionaries that are easy to work with.
print("\n\n----- PREPROCESSING ---------\n")

# Read and process data files
data = process_input_data()

#print(data["settings"]["element_type"])

# Extract and calculate relevant variables from input data
n_dofs_per_node = len(data["settings"]["element_type"].dof_names)       # number of nodal unknowns (dof) per node (e.g. displacements or temperature)
n_nodes_in_system = len(data["node_coordinates"])                       # total number of nodes in the system
n_dofs_in_system = n_dofs_per_node * n_nodes_in_system                  # total number of unknowns, which is the total number of degrees of freedom (dof)


#------------------------------------------------------------#
#------                  SOLUTION                      ------#
#------------------------------------------------------------#
# This part of the analysis is dedicated to the setup and the solution of the linear equation system [K]{v} = {f}.
# [K] is the system matrix, {f} is the load vector, {v} is the vector of the nodal unknowns.
# [K], {v} and {f} are structured using ascending node numbers.
# --> The first entries are related to all degrees of freedom (dof) of node number 0, then all dof of node number 1 and so on.
# As a last step, the solved nodal unknowns are used to calculate resultants.
print("\n\n----- SOLUTION --------------\n")

# Build the system matrix K_system by inserting all element matrices at the correct position using the connectivity matrix

K_system = build_system_matrix(data["elements"], data["connectivities"], n_dofs_in_system, n_dofs_per_node)
print("\nSystem matrix before applying boundary conditions:\n" + str(K_system) + "\n")


# Build the system load vector f_system by inserting all element load vectors at the correct position using the connectivity matrix
#f_system = build_load_vector(data["elements"], data["connectivities"], n_dofs_in_system, n_dofs_per_node, data["element_loads"])

#print("\nLoad vector before applying boundary conditions:\n" + str(f_system) + "\n")

# Add all Neumann boundary conditions to the load vector (e.g force at a node)
#f_system = apply_neumann_boundary_conditions(f_system, data["neumann_boundary_conditions"], n_dofs_per_node)

#print(f"\nLoad vector before applying Dirichlet boundary conditions:\n" + str(f_system) + "\n")
f_system = np.zeros(15)
# Apply nodal Dirichlet boundary conditions by modifying system matrix and load vector (e.g. support at a node)
K_system, f_system = apply_dirichlet_boundary_conditions(K_system, f_system, data["dirichlet_boundary_conditions"], n_dofs_per_node)

# Print system matrix and load vector
print("\nSystem matrix:\n" + str(K_system))
print("\nLoad vector:\n" + str(f_system))

# Solve the system of equations for the vector of nodal unknowns (e.g. displacements at the nodes)
# Matrix multiplication (numpy's @ operator) of the inverse of the system matrix and the system load vector: {v} = [K]^(-1) {f}
print("\nSolve system of equations ... " , end='')
v_system = np.linalg.inv(K_system) @ f_system
print("DONE")
print(v_system)
'''
# Subroutine analysis for calculating resultants
resultants_per_element = calculate_resultants(v_system, data["elements"], data["settings"]["element_type"], data["connectivities"])


#------------------------------------------------------------#
#------                  POSTPROCESSING                ------#
#------------------------------------------------------------#
# In the last part of the analysis the nodal unknowns and resultants are then visualized (e.g. plot over the element mesh)
# or exported to be visualized with postprocessing software (e.g. ParaView).
print("\n\n----- POSTPROCESSING --------\n")

# Print solution for unknowns and plot them
print("Nodal unknowns:\n" + str(v_system))
plot_primary_unknowns(v_system, data["node_coordinates"], data["settings"]["element_type"], data["connectivities"])

# Print and plot secondary unknowns (resultants)
for ele_no, resultants in enumerate(resultants_per_element):
    print("\nResultants of element No. " + str(ele_no) + ":")
    for res_name in resultants:
        print(str(res_name) + "^T = " + str(resultants[res_name].T[0]))
plot_resultants(resultants_per_element, data["node_coordinates"], data["settings"]["element_type"], data["connectivities"])

# Blocks to keep showing all open windows with figures until user closes them
plt.show()

print("\n\n----- FINISHED ANALYSIS -----\n")
'''