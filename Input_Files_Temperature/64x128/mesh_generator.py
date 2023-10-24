#----------ISD | TU Braunschweig----------#
#-----------Beethovenstraße 51------------#
#-----------38106 Braunschweig------------#
#-----------------------------------------#

import sys
import numpy as np
import pandas as pd


def generate_1d_mesh_lin(coord_corner_points, n_elements):
    """
    Function generating finite element mesh coordinate and connectivity data
    for a one dimensional domain using 2 node linear elements.
    The data is exported to "node_coordinates.csv" and "elements.csv".

    Parameters:
        coord_corner_points (tuple of float): coordinates of corner points ( x_min, x_max )
        n_elements (int): number of elements in x-direction
    """
    # Check input
    if len(coord_corner_points) != 2:
        raise Exception("[generate_1d_mesh_lin] Length of coord_corner_points should be 2!")
    if not isinstance(n_elements, int):
        raise Exception("[generate_1d_mesh_lin] n_elements should be 'int'!")

     # Read out input data
    x_min = coord_corner_points[0]
    x_max = coord_corner_points[1]
    n_nodes = n_elements + 1

    # Create nodal coordinates
    node_numbers = []
    node_coord_x = []
    node_coord_y = []
    for i, x in enumerate(np.linspace(x_min, x_max, n_nodes)):
        node_numbers.append(i)
        node_coord_x.append(x)
        node_coord_y.append(0)

    # Create connectivities
    ele_numbers = []
    material_id = np.zeros(n_elements)
    node_no_a = []
    node_no_b = []
    for i in range(n_elements):
        ele_numbers.append(i)
        node_no_a.append(i)
        node_no_b.append(i+1)

    # Export nodal coordinates and connectivities to csv files
    df_nodes = pd.DataFrame(data={"node_no": node_numbers, "x": node_coord_x, "y": node_coord_y})
    df_nodes.to_csv("./node_coordinates.csv", sep=',', index=False)
    df_connect = pd.DataFrame(data={"ele_no": ele_numbers,"material_id":material_id, "node_A": node_no_a, "node_B": node_no_b})
    df_connect.to_csv("./elements.csv", sep=',', index=False)


def generate_2d_mesh_rect_lin(coord_corner_points, n_elements):
    """
    Function generating finite element mesh coordinate and connectivity data
    for a one dimensional domain using 2 node linear elements.
    The data is exported to "node_coordinates.csv" and "elements.csv".

    Parameters:
        coord_corner_points (tuple of float): coordinates of corner points ( x_min, x_max, y_min, y_max )
        n_elements (tuple of int): number of elements in x- and y-direction ( n_elements_x, n_elements_y )
    """
    # Check input
    if len(coord_corner_points) != 4:
        raise Exception("[generate_2d_mesh_rect_lin] Length of coord_corner_points should be 4!")
    if len(n_elements) != 2:
        raise Exception("[generate_2d_mesh_rect_lin] Length of n_elements should be 2!")
    if not isinstance(n_elements[0], int):
        raise Exception("[generate_2d_mesh_rect_lin] n_elements for x-direction should be 'int'!")
    if not isinstance(n_elements[1], int):
        raise Exception("[generate_2d_mesh_rect_lin] n_elements for y-direction should be 'int'!")

    # Read out input data
    x_min = coord_corner_points[0]
    x_max = coord_corner_points[1]
    y_min = coord_corner_points[2]
    y_max = coord_corner_points[3]
    n_ele_x = n_elements[0]
    n_ele_y = n_elements[1]
    n_nodes_x = n_ele_x + 1
    n_nodes_y = n_ele_y + 1

    # Create nodal coordinates
    node_numbers = []
    node_coord_x = []
    node_coord_y = []
    coord_x = np.linspace(x_min, x_max, n_nodes_x)
    coord_y = np.linspace(y_min, y_max, n_nodes_y)
    node_no = 0
    for x in coord_x:
        for y in coord_y:
            node_coord_x.append(x)
            node_coord_y.append(y)
            node_numbers.append(node_no)
            node_no +=1

    # Create connectivities
    ele_numbers = []
    material_id = np.zeros(n_ele_x * n_ele_y)
    node_no_a = []
    node_no_b = []
    node_no_c = []
    node_no_d = []
    ele_no = 0
    for i in range(n_ele_x):
        for j in range(n_ele_y):
            no_a = i * n_nodes_y + j
            node_no_a.append(no_a)
            node_no_b.append(no_a + n_nodes_y)
            node_no_c.append(no_a + n_nodes_y + 1)
            node_no_d.append(no_a + 1)
            ele_numbers.append(ele_no)
            ele_no += 1

    # Export nodal coordinates and connectivities to csv files
    df_nodes = pd.DataFrame(data={"node_no": node_numbers, "x": node_coord_x, "y": node_coord_y})
    df_nodes.to_csv("./node_coordinates.csv", sep=',', index=False)
    df_connect = pd.DataFrame(data={"ele_no": ele_numbers,"material_id":material_id, "node_A": node_no_a, "node_B": node_no_b, "node_C": node_no_c, "node_D": node_no_d})
    df_connect.to_csv("./elements.csv", sep=',', index=False)


if __name__ == "__main__":
    args = sys.argv
    
    try:

        # Generate linear 1D mesh
        if len(args) == 3:
            x_min = float(args[1].split('-')[0])
            x_max = float(args[1].split('-')[1])
            n_ele = int(args[2])
            generate_1d_mesh_lin( (x_min, x_max), n_ele )

        # Generate linear 2D mesh
        elif len(args) == 5:
            x_min = float(args[1].split('-')[0])
            x_max = float(args[1].split('-')[1])
            y_min = float(args[2].split('-')[0])
            y_max = float(args[2].split('-')[1])
            n_ele_x = int(args[3])
            n_ele_y = int(args[4])
            generate_2d_mesh_rect_lin( (x_min, x_max, y_min, y_max), (n_ele_x, n_ele_y) )

        else:
            raise Exception
        
    except (Exception):
        print(("""\n\nERROR: [MESH GENERATOR] Unknown arguments passed! """ +
               """Make sure 'x_min and x_max' and 'y_min and y_max' are separated by a dash '-'.\n\n""" +
               """\tCORRECT USAGE:\n""" +
               """\t1D mesh: python mesh_generator.py <x_min>-<x_max> <number of elements>\n""" +
               """\t2D mesh: python mesh_generator.py <x_min>-<x_max> <y_min>-<y_max> <number of elements in x direction> <number of elements in y direction>\n"""))
