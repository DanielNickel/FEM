#----------ISD | TU Braunschweig----------#
#-----------Beethovenstraße 51------------#
#-----------38106 Braunschweig------------#
#-----------------------------------------#

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

from postprocessing.plotting_utilities import set_color_map, set_color_bar_ticks


def plot_primary_unknowns(v_system, node_coordinates, element_type, connectivities, show_figure=True, plot_element_bounds=True, color_limits=False):
    """
    Checks the number of dimensions, order of shape functions, number of nodes and other attributes of the element type
    in order to call a suitable plotting function.

    Args:
        v_system (np.array):                        nodal unknowns (dofs) of the system for which the system was solved for
        node_coordinates (list of tuple of float):  coordinates of all nodes
        element_type (type):                        type of the class of the system's elements
                                                        provides access to element class variables
        connectivities (list of tuple of int):      global node numbers corresponding to the nodes of each element
        show_figure (bool, optional):               if True then all figures are shown when directly after having been created - defaults to True

    Raises:
        Exception:                                  an Exception is raised if no suitable plotting function for the given element type can be found

    Returns:
        figures (list of matplotlib.figure):        list containing all figures that have been created using matplotlib
    """
    # Get names of degrees of freedom per node of the element class
    dof_names = element_type.dof_names

    # Plot 1D elements
    if element_type.n_dimensions == 1:
        print("\nWarning: [PLOT PRIMARY UNKNOWNS] Plotting linearly along x-axis, although given nodes might have non-zero y-coordinates.")
        if element_type.p_order_of_shape_functions > 1:
            print("\nWarning: [PLOT PRIMARY UNKNOWNS] Suitable plotting of primary unknowns for the given element type might not be implemented.\n"
              + "\t Plotting linearly along x-axis, although given element's shape functions are of higher order.")
        figures = plot_dofs_along_x_linear(v_system, node_coordinates, dof_names, show_figure, plot_element_bounds)

    # Plot quadrilateral elements
    elif element_type.n_dimensions == 2 and element_type.n_nodes_per_element == 4:
        if element_type.p_order_of_shape_functions > 1:
            print("\nWarning: [PLOT PRIMARY UNKNOWNS] Suitable plotting of primary unknowns for the given element type might not be implemented.\n"
              + "\t Using Gourand shading for smooth (linear) interpolation inside the elements, although given element's shape functions are of higher order.")
        else:
            print("\nWarning: [PLOT PRIMARY UNKNOWNS] The Gourand shading for smooth (linear) interpolation inside the elements might not match the element's shape functions exactly.")
        figures = plot_dofs_in_quads_linear(v_system, node_coordinates, connectivities, dof_names, show_figure, plot_element_bounds, color_limits)

     # Plot triangular elements
    elif element_type.n_dimensions == 2 and element_type.n_nodes_per_element == 3:
        if element_type.p_order_of_shape_functions > 1:
            print("\nWarning: [PLOT PRIMARY UNKNOWNS] Suitable plotting of primary unknowns for the given element type might not be implemented.\n"
              + "\t Using Gourand shading for smooth (linear) interpolation inside the elements, although given element's shape functions are of higher order.")
        else:
            print("\nWarning: [PLOT PRIMARY UNKNOWNS] The Gourand shading for smooth (linear) interpolation inside the elements might not match the element's shape functions exactly.")
        figures = plot_dofs_in_tris_linear(v_system, node_coordinates, connectivities, dof_names, show_figure, plot_element_bounds, color_limits)

    # Raise Exception because no matching plotting routine was found
    else:
        raise Exception("\n\t[PLOT PRIMARY UNKNOWNS] Suitable plotting of primary unknowns for the given element type might not be implemented.\n"
                        + "\tAvailable plotting routines are: Plot linearly along x-axis for 1D elements. Plot quadrilateral elements linearly. Plot triangular elements linearly.")


    return figures


def plot_dofs_along_x_linear(v_system, node_coordinates, dof_names, show_figure, plot_element_bounds):
    """
    Creates a matplotlib figure for each kind of degree of freedom separately.
    The degree of freedom is plotted for all nodes linearly along the x-axis using the nodal unknowns of the system and the
    x values of the node coordinates.
    This function is suitable for 1D elements which are aligned with the x-axis and have constant or linear shape functions.

    Args:
        v_system (np.array):                        nodal unknowns (dofs) of the system for which the system was solved for
        node_coordinates (list of tuple of float):  coordinates of all nodes
        dof_names (int):                            names of degrees of freedom per node of the element class
        show_figure (bool):                         if True then all figures are shown when directly after having been created

    Returns:
        figures (list of matplotlib.figure):        list containing all figures that have been created using matplotlib
    """
    # Initialize list of figures that are created
    figures = []

    # number of degrees of freedom per node
    n_dofs_per_node = len(dof_names)

    for dof_no, dof_name in enumerate(dof_names):
        # Get x values from node coordinates
        x_values = [node_x for node_x, node_y in node_coordinates]

        # Get values for degree of freedom solved for by slicing it from the vector of unknowns in the system
        # here, the slice function takes the respective dof number from v_system in steps of number of dofs per node
        slice_dof = slice(dof_no, len(v_system), n_dofs_per_node)
        dof_values = v_system[slice_dof]
        # Reshape dof values to list
        dof_values = dof_values.T[0].tolist()

        # Create plot
        fig,ax = plt.subplots()

        # Plot dof along x
        ax.plot(x_values, dof_values, color='blue')

        # Add dashed vertical lines at element transitions
        if plot_element_bounds:
            for x_border, d_border in zip(x_values, dof_values):
                ax.plot([x_border, x_border], [0, d_border], linestyle='--', color='blue')

        # Format plot
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(dof_name)

        # Show plot
        fig.tight_layout()
        if show_figure: plt.show(block=False)
        figures.append(fig)

    return figures


def plot_dofs_in_quads_linear(v_system, node_coordinates, connectivities, dof_names, show_figure, plot_element_bounds, color_limits):
    """
    Creates a matplotlib figure for each kind of degree of freedom separately.
    The degree of freedom is plotted onto the undeformed 2D geometry of the system applying Gouraud shading for smooth (linear)
    interpolation inside a quadrilateral element and using the nodal unknowns of the system, the node coordinates and the connectivity matrix.
    This function is suitable for 2D elements which have four nodes (quadrilaterals) and constant or linear shape functions.

    Args:
        v_system (np.array):                        nodal unknowns (dofs) of the system for which the system was solved for
        node_coordinates (list of tuple of float):  coordinates of all nodes
        connectivities (list of tuple of int):      global node numbers corresponding to the nodes of each element
        dof_names (int):                            names of degrees of freedom per node of the element class
        show_figure (bool):                         if True then all figures are shown when directly after having been created

    Returns:
        figures (list of matplotlib.figure):        list containing all figures that have been created using matplotlib
    """
    # Line width of the grid line. Set to zero to plot dof without outlining the mesh.
    plot_grid_lw = 0.5

    # Initialize list of figures that are created
    figures = []

    # number of degrees of freedom per node
    n_dofs_per_node = len(dof_names)

    for dof_no, dof_name in enumerate(dof_names):
        # Create plot
        fig,ax = plt.subplots()

        x_per_element = []
        y_per_element = []
        xy_contour_per_element = []
        v_per_element = []

        # Get x, y coordinate and data information for each element separately
        for node_no_a, node_no_b, node_no_c, node_no_d in connectivities:
            # Get node coordinates for element
            coord_a = node_coordinates[node_no_a]
            coord_b = node_coordinates[node_no_b]
            coord_c = node_coordinates[node_no_c]
            coord_d = node_coordinates[node_no_d]
            x_per_element.append( np.array([[coord_a[0],coord_b[0]],
                                            [coord_d[0],coord_c[0]]]) )
            y_per_element.append( np.array([[coord_a[1],coord_b[1]],
                                            [coord_d[1],coord_c[1]]]) )
            xy_contour_per_element.append( [coord_a, coord_b, coord_c, coord_d, coord_a] )

            # Get dof data for element
            v_a = v_system[ node_no_a * n_dofs_per_node + dof_no ][0]
            v_b = v_system[ node_no_b * n_dofs_per_node + dof_no ][0]
            v_c = v_system[ node_no_c * n_dofs_per_node + dof_no ][0]
            v_d = v_system[ node_no_d * n_dofs_per_node + dof_no ][0]
            v_per_element.append( np.array([[v_a,v_b],
                                            [v_d,v_c]]) )

        # Set color map and its limits according to min and max values of the dof's values or user-defined
        c_limit_min = color_limits[0] if color_limits else np.min(v_per_element)
        c_limit_max = color_limits[1] if color_limits else np.max(v_per_element)
        c_map, c_limits = set_color_map(min=c_limit_min, max=c_limit_max)

        # Plot all elements
        for x, y, v in zip(x_per_element, y_per_element, v_per_element):
            # Gouraud shading for smooth (linear) interpolation inside the element
            plt.pcolormesh(x, y, v, shading='gouraud', cmap=c_map, clim=c_limits)

        # Plot all element boundaries
        if plot_element_bounds:
            for xy in xy_contour_per_element:
                plt.plot([x for x,y in xy], [y for x,y in xy], 'k-', linewidth=plot_grid_lw)

        # Format plot
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$y$')
        ax.set_xlim((np.min(x_per_element)-0.2, np.max(x_per_element)+0.2))
        ax.set_ylim((np.min(y_per_element)-0.2, np.max(y_per_element)+0.2))
        ax.axis('equal')
        cbar = plt.colorbar()
        cbar.set_ticks(set_color_bar_ticks(cbar.get_ticks(), c_limits))
        cbar.ax.set_ylabel(dof_name)

        fig.tight_layout()
        if show_figure: plt.show(block=False)
        figures.append(fig)

    return figures


def plot_dofs_in_tris_linear(v_system, node_coordinates, connectivities, dof_names, show_figure, plot_element_bounds, color_limits):
    """
    Creates a matplotlib figure for each kind of degree of freedom separately.
    The degree of freedom is plotted onto the undeformed 2D geometry of the system applying Gouraud shading for smooth (linear)
    interpolation inside a triangular element and using the nodal unknowns of the system, the node coordinates and the connectivity matrix.
    This function is suitable for 2D elements which have three nodes (triangles) and constant or linear shape functions.

    Args:
        v_system (np.array):                        nodal unknowns (dofs) of the system for which the system was solved for
        node_coordinates (list of tuple of float):  coordinates of all nodes
        connectivities (list of tuple of int):      global node numbers corresponding to the nodes of each element
        dof_names (int):                            names of degrees of freedom per node of the element class
        show_figure (bool):                         if True then all figures are shown when directly after having been created

    Returns:
        figures (list of matplotlib.figure):        list containing all figures that have been created using matplotlib
    """
    # Line width of the grid line. Set to zero to plot dof without outlining the mesh.
    plot_grid_lw = 0.5

    # Initialize list of figures that are created
    figures = []

    # number of degrees of freedom per node
    n_dofs_per_node = len(dof_names)

    for dof_no, dof_name in enumerate(dof_names):
        # Create plot
        fig,ax = plt.subplots()

        x_per_element = []
        y_per_element = []
        v_per_element = []

        # Get x, y coordinate and data information for each element separately
        for node_no_a, node_no_b, node_no_c in connectivities:
            # Get node coordinates for element
            coord_a = node_coordinates[node_no_a]
            coord_b = node_coordinates[node_no_b]
            coord_c = node_coordinates[node_no_c]
            x_per_element.append( np.array([coord_a[0] ,coord_b[0], coord_c[0]]) )
            y_per_element.append( np.array([coord_a[1] ,coord_b[1], coord_c[1]]) )

            # Get dof data for element
            v_a = v_system[ node_no_a * n_dofs_per_node + dof_no ]
            v_b = v_system[ node_no_b * n_dofs_per_node + dof_no ]
            v_c = v_system[ node_no_c * n_dofs_per_node + dof_no ]
            v_per_element.append( np.array([v_a, v_b, v_c]) )

        # Set color map and its limits according to min and max values of the dof's values or user-defined
        c_limit_min = color_limits[0] if color_limits else np.min(v_per_element)
        c_limit_max = color_limits[1] if color_limits else np.max(v_per_element)
        c_map, c_limits = set_color_map(min=c_limit_min, max=c_limit_max)

        # Plot all elements
        for x, y, v in zip(x_per_element, y_per_element, v_per_element):
            tris = mtri.Triangulation(x, y, [[0,1,2]])
            # Use tricontourf or to plot without interpolation
            # ax.tricontourf(tris, v, cmap=c_map, clim=c_limits)
            plt.tripcolor(tris, v, shading='gouraud', cmap=c_map, clim=c_limits)
            if plot_element_bounds:
                plt.triplot(tris, 'k-', linewidth=plot_grid_lw)

        # Format plot
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$y$')
        ax.set_xlim((np.min(x_per_element)-0.2, np.max(x_per_element)+0.2))
        ax.set_ylim((np.min(y_per_element)-0.2, np.max(y_per_element)+0.2))
        ax.axis('equal')
        cbar = plt.colorbar()
        cbar.set_ticks(set_color_bar_ticks(cbar.get_ticks(), c_limits))
        cbar.ax.set_ylabel(dof_name)

        fig.tight_layout()
        if show_figure: plt.show(block=False)
        figures.append(fig)

    return figures
