#----------ISD | TU Braunschweig----------#
#-----------Beethovenstraße 51------------#
#-----------38106 Braunschweig------------#
#-----------------------------------------#

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

from postprocessing.plotting_utilities import set_color_map, set_color_bar_ticks


def plot_resultants(resultants_per_element, node_coordinates, element_type, connectivities, show_figure=True, plot_element_bounds=True, color_limits=False):
    """
    Checks the number of dimensions, order of shape functions, number of nodes and other attributes of the element type
    in order to call a suitable plotting function.

    Args:
        resultants_per_element (list of dict):      dictionary linking name of resultants (dict key)
                                                        to the resultants' values at the element's nodes as vertical numpy array (dict value)
                                                        for each element
        node_coordinates (list of tuple of float):  coordinates of all nodes
        element_type (type):                        type of the class of the system's elements
                                                        provides access to element class variables
        connectivities (list of tuple of int):      global node numbers corresponding to the nodes of each element
        show_figure (bool, optional):               if True then all figures are shown when directly after having been created - defaults to True
        plot_element_bounds (bool, optional):       if True then all element boundaries are plotted - defaults to True
        color_limits (tuple, optional):             if given then those limits are used for colored plots - defaults to False,
                                                        e.g. (-0.4, 15.6)

    Raises:
        Exception:                                  an Exception is raised if no suitable plotting function for the given element type can be found

    Returns:
        figures (list of matplotlib.figure):        list containing all figures that have been created using matplotlib
    """
    # Get number of different resultants of the element class
    n_resultants = len(element_type.p_order_of_resultants)

    # Plot 1D elements
    if element_type.n_dimensions == 1:
        print("\nWarning: [PLOT RESULTANTS] Plotting linearly along x-axis, although given nodes might have non-zero y-coordinates.")
        if not all(p < 2 for p in element_type.p_order_of_resultants):
            print("\nWarning: [PLOT RESULTANTS] Suitable plotting of resultants for the given element type might not be implemented.\n"
              + "\t Plotting linearly along x-axis, although given element's resultants are functions of higher order.")
        figures = plot_resultants_along_x_linear(resultants_per_element, node_coordinates, connectivities, n_resultants, show_figure, plot_element_bounds)

    # Plot quadrilateral elements
    elif element_type.n_dimensions == 2 and element_type.n_nodes_per_element == 4:
        if not all(p < 2 for p in element_type.p_order_of_resultants):
            print("\nWarning: [PLOT RESULTANTS] Suitable plotting of resultants for the given element type might not be implemented.\n"
              + "\t Using Gourand shading for smooth (linear) interpolation inside the elements, although given element's resultants are functions of higher order.")
        else:
            print("\nWarning: [PLOT RESULTANTS] The Gourand shading for smooth (linear) interpolation inside the elements might not match the element's shape functions exactly.")
        figures = plot_resultants_in_quads_linear(resultants_per_element, node_coordinates, connectivities, n_resultants, show_figure, plot_element_bounds, color_limits)

    # Plot triangular elements
    elif element_type.n_dimensions == 2 and element_type.n_nodes_per_element == 3:
        if not all(p < 2 for p in element_type.p_order_of_resultants):
            print("\nWarning: [PLOT RESULTANTS] Suitable plotting of resultants for the given element type might not be implemented.\n"
              + "\t Using Gourand shading for smooth (linear) interpolation inside the elements, although given element's resultants are functions of higher order.")
        else:
            print("\nWarning: [PLOT RESULTANTS] The Gourand shading for smooth (linear) interpolation inside the elements might not match the element's shape functions exactly.")
        figures = plot_resultants_in_tris_linear(resultants_per_element, node_coordinates, connectivities, n_resultants, show_figure, plot_element_bounds, color_limits)

    # Raise Exception because no matching plotting routine was found
    else:
        raise Exception("\n\t[PLOT RESULTANTS] Suitable plotting of resultants for the given element type might not be implemented.\n"
                        + "\tAvailable plotting routines are: Plot linearly along x-axis for 1D elements. Plot quadrilateral elements linearly. Plot triangular elements linearly.")

    return figures


def plot_resultants_along_x_linear(resultants_per_element, node_coordinates, connectivities, n_resultants, show_figure, plot_element_bounds):
    """
    Creates a matplotlib figure for each kind of resultant separately.
    The resultant is plotted for all nodes of each element linearly along the x-axis using the resultants per element, the
    x values of the node coordinates and an element's connectivities.
    This function is suitable for 1D elements which are aligned with the x-axis and for which linear interpolation of the resultants is sufficient.

    Args:
        resultants_per_element (list of dict):      dictionary linking name of resultants (dict key)
                                                        to the resultants' values at the element's nodes as vertical numpy array (dict value)
                                                        for each element
        node_coordinates (list of tuple of float):  coordinates of all nodes
        connectivities (list of tuple of int):      global node numbers corresponding to the nodes of each element
        n_resultants (int):                         number of different resultants of the element class
        show_figure (bool):                         if True then all figures are shown when directly after having been created

    Returns:
        figures (list of matplotlib.figure):        list containing all figures that have been created using matplotlib
    """
    # Initialize list of figures that are created
    figures = []

    for res_no in range(n_resultants):
        # Create plot
        fig,ax = plt.subplots()

        resultant_name = list(resultants_per_element[0].keys())[res_no]

        x_values = []
        res_values = []

        for ele_no, resultants in enumerate(resultants_per_element):
            # Get global node numbers of the element using the element's number
            node_numbers = connectivities[ele_no]
            x_ele = []
            for node_no in node_numbers:
                x_node = node_coordinates[node_no][0]
                x_ele.append(x_node)
                x_values.append(x_node)
            # Reshape results array to list
            res_ele = resultants[resultant_name].T[0].tolist()
            res_values += res_ele

            # Plot resultant values along x for element
            ax.plot(x_ele, res_ele, color='red')

        # Format plot
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(resultant_name)

        # Add dashed vertical lines at nodes (equals element transitions)
        if plot_element_bounds:
            for x_border, d_border in zip(x_values, res_values):
                    ax.plot([x_border, x_border], [0, d_border], linestyle='--', color='red')

        # Show plot
        fig.tight_layout()
        if show_figure: plt.show(block=False)
        figures.append(fig)

    return figures


def plot_resultants_in_quads_linear(resultants_per_element, node_coordinates, connectivities, n_resultants, show_figure, plot_element_bounds, color_limits):
    """
    Creates a matplotlib figure for each kind of resultant separately.
    The resultant is plotted onto the undeformed 2D geometry of the system applying Gouraud shading for smooth (linear)
    interpolation inside a quadrilateral element and using the resultants per element, the node coordinates and the connectivity matrix.
    This function is suitable for 2D elements which have four nodes (quadrilaterals) and for which linear interpolation of the resultants is sufficient.

    Args:
        resultants_per_element (list of dict):      dictionary linking name of resultants (dict key)
                                                        to the resultants' values at the element's nodes as vertical numpy array (dict value)
                                                        for each element
        node_coordinates (list of tuple of float):  coordinates of all nodes
        connectivities (list of tuple of int):      global node numbers corresponding to the nodes of each element
        n_resultants (int):                         number of different resultants of the element class
        show_figure (bool):                         if True then all figures are shown when directly after having been created

    Returns:
        figures (list of matplotlib.figure):        list containing all figures that have been created using matplotlib
    """
    # Line width of the grid line. Set to zero to plot dof without outlining the mesh.
    plot_grid_lw = 0.5

    # Initialize list of figures that are created
    figures = []

    for res_no in range(n_resultants):
        # Create plot
        fig,ax = plt.subplots()

        resultant_name = list(resultants_per_element[0].keys())[res_no]

        x_per_element = []
        y_per_element = []
        xy_contour_per_element = []
        res_per_element = []

        # Get x, y coordinate and data information for each element separately
        for ele_no, resultants in enumerate(resultants_per_element):
            # Get global node numbers of the element using the element's number
            node_no_a, node_no_b, node_no_c, node_no_d = connectivities[ele_no]

            # Get node coordinates for element
            coord_a = node_coordinates[node_no_a]
            coord_b = node_coordinates[node_no_b]
            coord_c = node_coordinates[node_no_c]
            coord_d = node_coordinates[node_no_d]
            x_per_element.append( np.array([[coord_a[0] ,coord_b[0]], [coord_d[0], coord_c[0]]]) )
            y_per_element.append( np.array([[coord_a[1] ,coord_b[1]], [coord_d[1], coord_c[1]]]) )
            xy_contour_per_element.append( [coord_a, coord_b, coord_c, coord_d, coord_a] )

            # Get resultant data for element
            res_ele = resultants[resultant_name].T[0].tolist()
            res_per_element.append( np.array([[res_ele[0], res_ele[1]], [res_ele[3], res_ele[2]]]) )

        # Set color map and its limits according to min and max values of the resultants' values or user-defined
        c_limit_min = color_limits[0] if color_limits else np.min(res_per_element)
        c_limit_max = color_limits[1] if color_limits else np.max(res_per_element)
        c_map, c_limits = set_color_map(min=c_limit_min, max=c_limit_max)

        # Plot all elements
        for x, y, res in zip(x_per_element, y_per_element, res_per_element):
            # Gouraud shading for smooth (linear) interpolation inside the element
            plt.pcolormesh(x, y, res, shading='gouraud', cmap=c_map, clim=c_limits)

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
        cbar.ax.set_ylabel(resultant_name)

        fig.tight_layout()
        if show_figure: plt.show(block=False)
        figures.append(fig)

    return figures


def plot_resultants_in_tris_linear(resultants_per_element, node_coordinates, connectivities, n_resultants, show_figure, plot_element_bounds, color_limits):
    """
    Creates a matplotlib figure for each kind of resultant separately.
    The resultant is plotted onto the undeformed 2D geometry of the system applying Gouraud shading for smooth (linear)
    interpolation inside a triangular element and using the resultants per element, the node coordinates and the connectivity matrix.
    This function is suitable for 2D elements which have three nodes (triangles) and for which linear interpolation of the resultants is sufficient.

    Args:
        resultants_per_element (list of dict):      dictionary linking name of resultants (dict key)
                                                        to the resultants' values at the element's nodes as vertical numpy array (dict value)
                                                        for each element
        node_coordinates (list of tuple of float):  coordinates of all nodes
        connectivities (list of tuple of int):      global node numbers corresponding to the nodes of each element
        n_resultants (int):                         number of different resultants of the element class
        show_figure (bool):                         if True then all figures are shown when directly after having been created

    Returns:
        figures (list of matplotlib.figure):        list containing all figures that have been created using matplotlib
    """
    # Line width of the grid line. Set to zero to plot dof without outlining the mesh.
    plot_grid_lw = 0.5

    # Initialize list of figures that are created
    figures = []

    for res_no in range(n_resultants):
        # Create plot
        fig,ax = plt.subplots()

        resultant_name = list(resultants_per_element[0].keys())[res_no]

        x_per_element = []
        y_per_element = []
        res_per_element = []

        # Get x, y coordinate and data information for each element separately
        for ele_no, resultants in enumerate(resultants_per_element):
            # Get global node numbers of the element using the element's number
            node_no_a, node_no_b, node_no_c = connectivities[ele_no]

            # Get node coordinates for element
            coord_a = node_coordinates[node_no_a]
            coord_b = node_coordinates[node_no_b]
            coord_c = node_coordinates[node_no_c]
            x_per_element.append( np.array([coord_a[0] ,coord_b[0], coord_c[0]]) )
            y_per_element.append( np.array([coord_a[1] ,coord_b[1], coord_c[1]]) )

            # Get resultant data for element
            res_ele = resultants[resultant_name].T[0].tolist()
            res_per_element.append( np.array([res_ele[0], res_ele[1], res_ele[2]]) )

        # Set color map and its limits according to min and max values of the resultants' values or user-defined
        c_limit_min = color_limits[0] if color_limits else np.min(res_per_element)
        c_limit_max = color_limits[1] if color_limits else np.max(res_per_element)
        c_map, c_limits = set_color_map(min=c_limit_min, max=c_limit_max)

        # Plot all elements
        for x, y, res in zip(x_per_element, y_per_element, res_per_element):
            tris = mtri.Triangulation(x, y, [[0,1,2]])
            # Use tricontourf or to plot without interpolation
            # ax.tricontourf(tris, res, cmap=c_map, clim=c_limits)
            plt.tripcolor(tris, res, shading='gouraud', cmap=c_map, clim=c_limits)
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
        cbar.ax.set_ylabel(resultant_name)

        fig.tight_layout()
        if show_figure: plt.show(block=False)
        figures.append(fig)

    return figures
