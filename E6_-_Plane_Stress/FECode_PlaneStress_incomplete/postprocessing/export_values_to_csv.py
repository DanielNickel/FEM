#----------ISD | TU Braunschweig----------#
#-----------Beethovenstraße 51------------#
#-----------38106 Braunschweig------------#
#-----------------------------------------#

import pandas as pd
import numpy as np

import os, sys
path_file_dir = os.path.dirname(os.path.abspath(__file__))


def export_primary_unknowns_at_nodes(v_system, node_coordinates, element_type, file_name, node_numbers=False):
    """
    Export primary unknowns at nodes to a CSV file with the given file name in the following format:
    <"node_no", "x", "y", "dof1", "dof2", ... ">
    A list of node numbers can be given to only export values at the given nodes.
    Otherwise the values for the unknowns are exported for all nodes of the system.

    Args:
        v_system (np.array):                    nodal unknowns (dofs) of the system for which the system was solved for
        node_coordinates (list):                list of tuples representing the coordinates of nodes (x, y)
        element_type (type):                    type of the class of the system's elements
                                                    provides access to element class variables
        file_name (str):                        name of the output CSV file
        node_numbers (list, optional):          list of node numbers to include in the export
                                                    defaults to False, in which case all node numbers are included

    Raises:
        KeyError:                               error if a node number provided in `node_numbers` does not exist
    """

    # Get number of dofs per element along with their names
    try:
        dof_names = element_type.dof_names
    # TODO: remove when PlaneStress2D4N is implemented
    except:
        dof_names = ["u_x", "u_y"]
    n_dofs_per_node = len(dof_names)

    # Convert v_system into a list of number of dofs where each list entry contains a list of the dof's value at each node
    nodal_values_per_dof = v_system.reshape(-1, n_dofs_per_node)

    # Convert node_coordinates list of tuples to a 2D array
    node_coords = np.array(node_coordinates)

    # Create DataFrame of dofs (variable number of headers)
    df_nodal_values_per_dof = pd.DataFrame(nodal_values_per_dof, columns=dof_names)

    # Create DataFrame of node_no, x and y coordinates (fixed number of headers)
    df_nodes = pd.DataFrame({"node_no":[node_no for node_no in range(len(node_coordinates))], "x":node_coords[:,0], "y":node_coords[:,1]})

    # Concatenate all DataFrame objects and format node numbers as integer
    df_concat = pd.concat([df_nodes, df_nodal_values_per_dof], axis=1)
    df_concat["node_no"] = df_concat["node_no"].values.astype(int)

    # If node numbers are given, keep only the values corresponding to the required nodes in given order
    if node_numbers:
        try:
            df_concat = df_concat.loc[node_numbers]
        except KeyError as key_error:
            print("\n\nError: [EXPORT_VALUES_TO_CSV] Node number " + str(key_error.args[0].split()[0][1]) + " was not found. Check if the given node number exists!\n\n")
            sys.exit(1)  # Stops further execution due to the above error.

    # Convert the DataFrame to CSV with the file_name provided by the user
    df_concat.to_csv(path_file_dir +"/"+str(file_name)+".csv", sep=',', index=False)


def export_resultants_at_nodes_of_elements(resultants_per_element, node_coordinates, connectivities, file_name, element_numbers, node_numbers):
    """
    Export ele_resultants at nodes of elements to a CSV file with the given file name in the following format:
    <"element_no", "node_no", "x", "y", "resultant1", "resultant2", ... >

    Args:
        resultants_per_element (list):          list of dictionaries containing ele_resultants per element
        node_coordinates (list of tuples):      list of tuples representing the coordinates of nodes
        connectivities (list of tuple of int):  global node numbers corresponding to the nodes of each element
        file_name (str):                        name of the output CSV file
        element_numbers (list):                 list of element numbers to include in the exported data
        node_numbers (list):                    list of node numbers to include in the exported data

    Raises:
        Exception:                              error if an invalid element number is provided
    """

    # Get resultant names
    resultant_names = list(resultants_per_element[0].keys())

    # Convert node_coordinates list of tuples to a 2D array
    node_coords = np.array(node_coordinates).T

    # Initialize lists to record element and node numbers
    list_ele_nums = []
    list_node_nums = []
    x_coords = []
    y_coords = []

    # Initialize dictionary with a key:list for each resultant to record the values of all resultants per element and node
    dict_resultants = {}
    for res_name in resultant_names:
        dict_resultants[res_name] = []

    # Loop over all given element numbers
    for ele_no in element_numbers:

        # Check if element exists or not
        if ele_no > len(connectivities)-1 or ele_no < 0:
            raise Exception("\n\nError: [EXPORT_VALUES_TO_CSV] Element number " + str(ele_no) + " does not exist!\n\n")

        # Get resultants and node numbers of the element
        ele_resultants = resultants_per_element[ele_no]
        ele_connectivity = list(connectivities[ele_no])

        # Loop over all given node numbers
        for node_no in node_numbers:
            # Check if node number is one of the nodes of the element
            try:
                node_index = ele_connectivity.index(node_no)
            # Node number is not a part of the nodes of the element
            except ValueError:
                pass

            # Node is a part of the nodes of the element, so append one entry to each list for the node
            else:
                # Append element number, node number and node coordinates
                list_ele_nums.append(ele_no)
                list_node_nums.append(node_no)
                x_coords.append( node_coords[0][node_no] )
                y_coords.append( node_coords[1][node_no] )
                # Append the values of all resultants for the node of the element
                for res_name in resultant_names:
                    dict_resultants[res_name].append( ele_resultants[res_name][node_index][0] )

    # Create DataFrames for ele_resultants and other parameters separately
    df_resultants = pd.DataFrame(dict_resultants)
    df_ele_no_and_nodes = pd.DataFrame({"ele_no": list_ele_nums, "node_no": list_node_nums, "x":x_coords, "y":y_coords})

    # Concatenate both the DataFrames
    df_concat = pd.concat([df_ele_no_and_nodes, df_resultants], axis=1)

    # Check if all requested node_numbers are present in the DataFrame
    missing_nodes = set(node_numbers) - set(df_concat['node_no'])
    if missing_nodes:
        print("\nWarning: [EXPORT_VALUES_TO_CSV] Given the provided list of element numbers, the following node numbers were missing: " + str(missing_nodes))

    # Convert the DataFrame to CSV with the file_name provided by the user
    df_concat.to_csv(path_file_dir +"/"+str(file_name)+".csv", sep=',', index=False)
