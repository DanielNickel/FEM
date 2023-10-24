#----------ISD | TU Braunschweig----------#
#-----------Beethovenstraße 51------------#
#-----------38106 Braunschweig------------#
#-----------------------------------------#

import numpy as np


def build_system_matrix(elements, connectivities, n_dofs_in_system, n_dofs_per_node):
    """
    Builds the system matrix K_system by inserting all element matrices at the correct position using the connectivity matrix.
    Iterates over all elements and calls their "build_element_stiffness_matrix" method.

    Args:
        elements (list of instances):               all elements of the system
        connectivities (list of tuple of int):      global node numbers corresponding to the nodes of each element
        n_dofs_in_system (int):                     number of unknowns/ degrees of freedom in the system
        n_dofs_per_node (int):                      number of degrees of freedom of a node

    Returns:
        K_system (np.array):                        global system matrix
    """
    print("[FE_FUNCTIONS] Building system matrix ... " , end='')

    # Initialize the system matrix
    K_system = np.zeros( (n_dofs_in_system, n_dofs_in_system) )

    # Loop over all elements
    for ele in elements:

        # Get element stiffness matrix from respective element
        K_element = ele.build_element_stiffness_matrix()
        
        # Get global node numbers of the element using the element's number
        node_numbers = connectivities[ele.ele_no]

        # Loop over the element's nodes to add correct part of element matrix ("ele") to correct part of system matrix ("sys")
        # python's slice() function is used to get smaller matrices w.r.t nodes from matrix K_element (e.g 2-node-element: k_AA, k_AB, k_BA, k_BB of sizes [n_dof_per_node x n_dof_per_node])
        # and insert into correct matrix part of K_system
        # (e.g. slice(0,2) takes the first and the second row (i-direction) or column (j-direction) of a matrix respectively)
        for i_ele, i_sys in enumerate(node_numbers):
            for j_ele, j_sys in enumerate(node_numbers):
                i_slice_ele = slice( i_ele * n_dofs_per_node, (i_ele+1) * n_dofs_per_node )
                j_slice_ele = slice( j_ele * n_dofs_per_node, (j_ele+1) * n_dofs_per_node )

                i_slice_sys = slice( i_sys * n_dofs_per_node, (i_sys+1) * n_dofs_per_node )
                j_slice_sys = slice( j_sys * n_dofs_per_node, (j_sys+1) * n_dofs_per_node )

                K_system[i_slice_sys, j_slice_sys] += K_element[i_slice_ele, j_slice_ele]
    
    print("DONE")
    return K_system


def build_load_vector(elements, connectivities, n_dofs_in_system, n_dofs_per_node, element_loads):
    """
    Builds the system load vector f_system by inserting all element load vectors at the correct position using the connectivity matrix.
    Iterates over all elements and calls their "build_element_load_vector" method.

    Args:
        elements (list of instances):                   all elements of the system
        connectivities (list of tuple of int):          global node numbers corresponding to the nodes of each element
        n_dofs_in_system (int):                         number of unknowns/ degrees of freedom in the system
        n_dofs_per_node (int):                          number of degrees of freedom of a node
        element_loads (list of dict):                   all element loads as dict containing respective element number, number of degree of freedom the load refers to and load value
                                                            e.g. [{"ele_no":0, "dof_no":0, "value":10}, {"ele_no":1, "dof_no":0, "value":10}]

    Returns:
        f_system (np.array):                            global load vector of the system
    """
    print("[FE_FUNCTIONS] Building load vector ... " , end='')

    # Initialize the system load vector
    f_system = np.zeros( (n_dofs_in_system, 1) )

    # Loop over all elements
    for ele in elements:
        # Loop over element_loads to find the respective element's loads
        load_dof_numbers = []
        load_values = []
        for ele_load in element_loads:
            if ele_load["ele_no"] == ele.ele_no:
                load_dof_numbers.append(ele_load["dof_no"])
                load_values.append(ele_load["value"])
                break

        # Get element load vector from respective element handing over the number of degree of freedom the load refers to and the load's value
        f_element = ele.build_element_load_vector(load_dof_numbers, load_values)

        # Get global node numbers of the element using the element's number
        node_numbers = connectivities[ele.ele_no]

        # Loop over the element's nodes to add correct part of element vector ("ele") to correct part of system load vector ("sys")
        # python's slice() function is used to get smaller vectors w.r.t nodes from vector f_element (e.g 2-node-element: f_A, f_B of sizes [n_dof_per_node x 1])
        # and insert into correct vector part of f_system
        # (e.g. slice(0,2) takes the first and the second row (i-direction) of a vector)
        for i_ele, i_sys in enumerate(node_numbers):
            i_slice_ele = slice( i_ele * n_dofs_per_node, (i_ele+1) * n_dofs_per_node )

            i_slice_sys = slice( i_sys * n_dofs_per_node, (i_sys+1) * n_dofs_per_node )

            f_system[i_slice_sys] += f_element[i_slice_ele]

    print("DONE")
    return f_system


def apply_neumann_boundary_conditions(f_system, neumann_boundary_conditions, n_dof_per_node):
    """
    Add all nodal loads to the load vector f_global.
    (Example for Neumann BC: force at a node)

    Args:
        f_system (np.array):                            global load vector of the system
        neumann_boundary_conditions (list of dict):     all Neumann boundary conditions as dict containing respective node number, degree of freedom and value
                                                            e.g. [{"node_no":2, "dof_no":0, "value":10}]
        n_dofs_per_node (int):                          number of degrees of freedom of a node

    Returns:
        f_system (np.array):                            global load vector of the system
    """
    print("[FE_FUNCTIONS] Applying Neumann boundary conditions ... " , end='')

    # Loop over all Neumann boundary conditions
    # add value of each boundary condition at correct position in the system load vector ("sys")
    for n_bc in neumann_boundary_conditions:
        i_position_sys = n_bc["node_no"] * n_dof_per_node + n_bc["dof_no"]

        f_system[i_position_sys] += n_bc["value"]

    print("DONE")
    return f_system


def apply_dirichlet_boundary_conditions(K_system, f_system, dirichlet_boundary_conditions, n_dofs_per_node):
    """
    Apply nodal Dirichlet boundary conditions by modifying system matrix and load vector.
    (Example for Dirichlet BC: support at a node)

    Args:
        K_system (np.array):                            global system matrix
        f_system (np.array):                            global load vector of the system
        dirichlet_boundary_conditions (list of dict):   all Dirichlet boundary conditions as dict containing respective node number, degree of freedom and value
                                                            e.g. [{"node_no":0, "dof_no":0, "value":0.5}]
        n_dofs_per_node (int):                          number of degrees of freedom of a node

    Returns:
        K_system (np.array):                            global system matrix
        f_system (np.array):                            global load vector of the system
    """
    print("[FE_FUNCTIONS] Applying Dirichlet boundary conditions ... " , end='')

    # Loop over all Dirichlet boundary conditions
    for d_bc in dirichlet_boundary_conditions:
        # i_position_sys is the position of the specified degree of freedom in the system
        # the position in K_system in j-direction is the same as in i-direction because it is one specific degree of freedom where the boundary condition is applied
        # to get the entire row or colum of K_system in which the degree of freedom lies, we need to extract that row or colum from the matrix
        # e.g. K[:, slice()] gets all rows (i-direction) of the column (j-direction) specified using slice()
        i_position_sys = d_bc["node_no"] * n_dofs_per_node + d_bc["dof_no"]
        j_position_sys = i_position_sys

        i_slice_sys = slice(i_position_sys, i_position_sys+1)
        j_slice_sys = i_slice_sys
        
        # Multiply respective column of the system matrix by the boundary condition value and substract it from the load vector
        # here ":" gives all rows of the specified column
        f_system -= K_system[:, j_slice_sys] * d_bc["value"] #Geändert?
        
        # Set respective row of the system matrix to zero
        # here ":" gives all columns of the specified row
        K_system[i_slice_sys, :] *= 0

        # Set respective column of the system matrix to zero
        K_system[:, j_slice_sys] *= 0

        # Set specified dof to the value of the boundary condition
        K_system[i_position_sys][j_position_sys] = 1
        f_system[i_position_sys] = d_bc["value"]
        
    print("DONE")
    return K_system, f_system


def calculate_resultants(v_system, elements, element_type, connectivities):
    """
    Calculates the resultants as a dict for each element by iterating over all elements and calling their "calculate_element_resultants" method.

    Args:
        v_system (np.array):                    nodal unknowns (dofs) of the system for which the system was solved for
        elements (list of instances):           all elements of the system
        element_type (type):                    type of the class of the system's elements
                                                    provides access to element class variables
        connectivities (list of tuple of int):  global node numbers corresponding to the nodes of each element

    Returns:
        resultants_per_element (list of dict):  dictionary linking name of resultants (dict key)
                                                    to the resultants' values at the element's nodes as vertical numpy array (dict value)
                                                    for each element
    """
    print("[CALCULATE_RESULTANTS] Calculating resultants for each element ... " , end='')
    n_dofs_per_node = len(element_type.dof_names)

    # Initialize list of dict of resultants for each element
    resultants_per_element = []

    for ele in elements:
        # Get global node numbers of the element using the element's number
        node_numbers = connectivities[ele.ele_no]

        # Initialize vector of nodal unknowns for the nodes of the element
        v_element = np.zeros( (element_type.n_dofs_per_element, 1) )

        # Loop over the element's nodes to add correct part of the system's dofs ("sys") to correct part of the element's dofs ("ele")
        # python's slice() function is used to get smaller vectors w.r.t nodes from vector v_system
        # and insert into correct vector part of v_element (e.g 2-node-element: u_A, u_B of sizes [n_dof_per_node x 1])
        # (e.g. slice(0,2) takes the first and the second row (i-direction) of a vector)
        for i_ele, i_sys in enumerate(node_numbers):
            i_slice_ele = slice( i_ele * n_dofs_per_node, (i_ele+1) * n_dofs_per_node )

            i_slice_sys = slice( i_sys * n_dofs_per_node, (i_sys+1) * n_dofs_per_node )

            v_element[i_slice_ele] += v_system[i_slice_sys]

        resultants_per_element.append( ele.calculate_element_resultants(v_element) )

    print("DONE")
    return resultants_per_element
