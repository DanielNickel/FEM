#----------ISD | TU Braunschweig----------#
#-----------Beethovenstraße 51------------#
#-----------38106 Braunschweig------------#
#-----------------------------------------#

import matplotlib.pyplot as plt
import numpy as np


def plot_deformed_shape(v_system, node_coordinates, element_type, connectivities, show_figure=True, scaling_factor=1.0):
    """
    This function plots the deformed shape of a mesh based on the node coordinates and displacement of the nodes.
    Therefore, the class variable "dof_names" of the element type is searched for displacements in x- and y-direction,
    which are then extracted from "v_system".
    A transformation matrix depending on the element type is used in order to get the correct sequence of an element's nodes for plotting.

    Args:
        v_system (np.array):                        nodal unknowns of the system for which the system was solved for
        node_coordinates (list of tuple of float):  coordinates of the nodes in the mesh
        element_type (type):                        type of the class of the elements in the mesh
        connectivities (list of tuple of int):      connectivity matrix of the mesh relating global node numbers to the nodes of each element
        show_figure (bool, optional):               determines whether to show the figure - defaults to True
        scaling_factor (float, optional):           scaling factor for the deformation of the system - defaults to 1.0, which equals no scaling

    Returns:
        fig (matplotlib.figure):                    figure object containing the plotted deformed shape

    Raises:
        AttributeError:                             error if 'dof_names' is not found as class variable of the given element_type
        NameError:                                  error if the transformation matrix corresponding to the element name is not found
    """
    ## CALCULATE NODE POSITIONS AFTER DEFORMATION

    # Get names of degrees of freedom per node of the element class
    try:
        dof_names = element_type.dof_names
    except AttributeError:
        raise AttributeError("\n\nError:  [PLOT_DEFORMED_SHAPE] 'dof_names' was not found in the given element_type.\n"
                             + "\tMake sure to not pass element_type as a string.\n\n")

    # Format v_system according to the number of dofs as [[u0, u1, u2, ...], [w0, w1, w2, ...], ...]
    v_system_per_dof_no = [[] for _ in range(len(dof_names))]
    for i_dof, dof in enumerate(v_system):
        v_system_per_dof_no[i_dof % len(dof_names)].append(dof)

    # Extract deformations in x direction from v_system_per_dof_no corresponding to the dof_name
    if "u" in dof_names:
        deformation_x = np.array(v_system_per_dof_no[dof_names.index("u")])
    elif "u_x" in dof_names:
        deformation_x = np.array(v_system_per_dof_no[dof_names.index("u_x")])
    else:
        deformation_x = np.zeros(len(node_coordinates))

    # Extract deformations in y directions from v_system_per_dof_no corresponding to the dof_name
    if "v" in dof_names:
        deformation_y = np.array(v_system_per_dof_no[dof_names.index("v")])
    elif "w" in dof_names:
        deformation_y = np.array(v_system_per_dof_no[dof_names.index("w")])
    elif "u_y" in dof_names:
        deformation_y = np.array(v_system_per_dof_no[dof_names.index("u_y")])
    else:
        deformation_y = np.zeros(len(node_coordinates))

    # Scale deformation
    deformation_x *= scaling_factor
    deformation_y *= scaling_factor

    # Convert the list of tuples to a matrix containing all x coordinates in the first column and all y coordinates in the second
    coordinates = np.array(node_coordinates)

    # Add deformations to the node coordinates
    x_coordinates_deformed = [x_coord + def_x for x_coord, def_x in zip(coordinates[:,0], deformation_x)]
    y_coordinates_deformed = [y_coord + def_y for y_coord, def_y in zip(coordinates[:,1], deformation_y)]

    ## TRANSFORM SEQUENCE OF CONNECTIVITIES INTO CORRECT PLOTTING SEQUENCE

    # Convert connectivities from list of tuples to a matrix (1 row per element)
    connectivity_matrix = np.array(connectivities)

    # Get the transformation matrix to get the required node sequence from the function for the corresponding element name
    transf_mat = get_node_sequence_transformation_matrix(element_type.__name__)

    # Transform connectivities of each element into the sequence needed for plotting using numpy matrix multiplication
    nodes_seq_per_element = []
    for ele_connect in connectivity_matrix:
        nodes_seq_per_element.append( (ele_connect @ transf_mat) )

    ## PLOTTING

    fig, ax = plt.subplots()

    # Plot the undeformed mesh
    for ele_connect in connectivity_matrix:
        x = [coordinates[n, 0] for n in ele_connect]
        y = [coordinates[n, 1] for n in ele_connect]
        plt.plot(x, y, 'b--')  # Use blue dashed line for undeformed shape

    # Plot the deformed mesh
    for nodes_seq in nodes_seq_per_element:
        x = [x_coordinates_deformed[n] for n in nodes_seq]
        y = [y_coordinates_deformed[n] for n in nodes_seq]
        plt.plot(x, y, 'k-')

    # Plot the function graph
    x_function = np.linspace(np.min(coordinates[:, 0]), np.max(coordinates[:, 0]), 100)
    y_function = -(10 / (2 * 120000 * (0.15 * 5**3 / 12))) * (-1/3 * x_function**3 + 15 * x_function**2) * scaling_factor + 2.5
    plt.plot(x_function, y_function, 'r-', label='Analytical solution')

    # Format plot
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.set_xlim((np.min(x_coordinates_deformed)-2, np.max(x_coordinates_deformed)+2))
    ax.set_ylim((np.min(y_coordinates_deformed)-2, np.max(y_coordinates_deformed)+2))
    ax.axis('equal')
    scaling_text = "" if scaling_factor == 1.0 else "x " + str(scaling_factor)
    plt.text(0.9, 0.95, scaling_text, transform=plt.gca().transAxes)
    plt.legend()


    n_nodes_x=0
    for row in coordinates:
        if row[0]==0.0:
            n_nodes_x+=1
    n_nodes_y=int(len(coordinates)/n_nodes_x)
    middle_nodes=[]
    for i in range (0,n_nodes_y):
        middle_nodes.append(int(n_nodes_x/2)+i*n_nodes_x)
    middle_nodes_coords=coordinates[middle_nodes]
    deformation_x.ravel()
    #x_coordinates_error = [x_coord + def_x for x_coord, def_x in zip(middle_nodes_coords[:,0], deformation_x[middle_nodes])]
    y_coordinates_error = [y_coord + def_y for y_coord, def_y in zip(middle_nodes_coords[:,1], deformation_y[middle_nodes])]
    y_coordinates_error = [float(item[0]) for item in y_coordinates_error]
    x_deflection=np.linspace(0,15,n_nodes_y)
    deflection=-(10 / (2 * 120000 * (0.15 * 5**3 / 12))) * (-1/3 * x_deflection**3 + 15 * x_deflection**2) * scaling_factor + 2.5
    print(deflection)
    error = np.abs(np.array(deflection) - np.array(y_coordinates_error))
    
    
    fig1,ax1 = plt.subplots()
    plt.plot(x_deflection, error, 'k-')
    #plt.plot(x_deflection, deflection, 'g-')
    ax1.set_xlabel(r'$x$')
    ax1.set_ylabel(r'$absolute error$')
    ax1.set_xlim((np.min(x_coordinates_deformed)-2, np.max(x_coordinates_deformed)+2))
    ax1.set_ylim((0, np.max(y_coordinates_error)))
    #ax1.axis('equal')
    plt.grid(True)
    scaling_text = "" if scaling_factor == 1.0 else "x " + str(scaling_factor)
    plt.text(0.9, 0.95, scaling_text, transform=plt.gca().transAxes)

    return fig


def get_node_sequence_transformation_matrix(element_name):
    """
    This function returns a transformation matrix based on the given element name to get the nodes in a sequence required for plotting.
    eg. in case of Heat2D4N, connectivity [A,B,C,D] is transformed into the format [A,B,C,D,A], as required sequentially by plt.plot() for creating a polygonal shape.
    Similarly, for Bar1D3N, connectivity [A,B,C] is transformed into the format [A,C,B], so that the nodes are linearly connected in the correct sequence.

    Args:
        element_name (str):         name of the element type for which the transformation matrix is requested

    Returns:
        np.array:                   a transformation matrix to get correct node sequence corresponding to the given element name

    Raises:
        NameError:                  error if the given element name is not found
    """

    if element_name == "Bar1D2N":
        return np.array([[1,0],
                         [0,1]])
    elif element_name == "Frame1D3N_TEST":
        return np.array([[1,0,0],
                         [0,0,1],
                         [0,1,0]])
    elif element_name == "Heat2D4N" or element_name == "PlaneStress2D4N" or element_name == "PlaneStress2D4N_SRI" or element_name == "Element2D4N_TEST":
        return np.array([[1,0,0,0,1],
                         [0,1,0,0,0],
                         [0,0,1,0,0],
                         [0,0,0,1,0]])
    else:
        raise NameError("[PLOT_DEFORMED_SHAPE] The given element type \"" + element_name + "\" was not found.")
