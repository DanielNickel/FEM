#----------ISD | TU Braunschweig----------#
#-----------Beethovenstraße 51------------#
#-----------38106 Braunschweig------------#
#-----------------------------------------#

import pandas as pd
import numpy as np

import os

#from elements.bar_1d2n import Bar1D2N
#from elements.heat_2d4n import Heat2D4N
from elements.plane_stress_2d4n import PlaneStress2D4N


#NOTE All CSV files should be kept inside a folder called "input_data" inside the folder "preprocessing"
#NOTE node numbers, element numbers, dof numbers of one node all start from zero

def process_input_data():
    """
    Reads .csv input files using pandas and processes them.

    Returns:
        data (dict):    key-value pairs of preprocessed input data:
                            "settings":dict using string keys,
                            "node_coordinates":list of node coordinates tuples,
                            "connectivities":list of tuples,
                            "elements":list of instance,
                            "element_loads":list of dict using string keys ("ele_no", "dof_no", "value")
                            "neumann_boundary_conditions":list of dict using string keys ("ele_no", "dof_no", "value"),
                            "dirichlet_boundary_conditions":list of dict using string keys ("ele_no", "dof_no", "value").
    """
    print("[INPUT_DATA] Reading and processing .csv-files ... ", end='')

    # Set path to input files
    path_file_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "input_data")

    # Dictionary of element types, that are available
    ele_types = {'PlaneStress2D4N':PlaneStress2D4N}


    #------------------------------------------------------------#
    #------             fe_settings.csv                    ------#
    #------------------------------------------------------------#
    # Read fe_settings.csv to a pandas DataFrame object
    df_settings = pd.read_csv(os.path.join(path_file_dir, "fe_settings.csv"), header=0)

    # Convert pandas DataFrame to dictionary
    settings = df_settings.to_dict(orient='records')[0]

    # Check whether element type exists
    if settings['element_type'] in ele_types:
        ele_type = ele_types[settings['element_type']]
        settings['element_type'] = ele_type
    else:
        raise ValueError ("[INPUT_DATA] Given element type '" + str(settings['element_type']) + "' does not exist.")


    #------------------------------------------------------------#
    #------             node_coordinates.csv               ------#
    #------------------------------------------------------------#
    # Read node_coordinates.csv to a pandas DataFrame object
    df_coords = pd.read_csv(os.path.join(path_file_dir, "node_coordinates.csv"), header=0)

    # Check if any element number is missing (in ascending order) or whether element numbering does not start at zero
    for flag in df_coords['node_no'] == range(len(df_coords)):
        if flag == False:
            raise ValueError("[INPUT_DATA] Node numbering does not start with zero OR there is a node missing. Please check node_coordinates.csv")
    df_coords['x'] = pd.to_numeric(df_coords['x'], errors='coerce')
    if df_coords['x'].isnull().sum() > 0 :
        raise TypeError("[INPUT_DATA] No valid x value given for " + str(df_coords['x'].isnull().sum()) + " node(s). Please check node_coordinates.csv")
    df_coords['y'] = pd.to_numeric(df_coords['y'], errors='coerce')
    if df_coords['y'].isnull().sum() > 0:
        raise TypeError("[INPUT_DATA] No valid y value given for " + str(df_coords['y'].isnull().sum()) + " node(s). Please check node_coordinates.csv")

    # Convert DataFrame to list of tuples to create list of node coordinates (dropping "node_no" column).
    # Each tuple contains the coordinates of one node and the list index corresponds to node numbering.
    df_coords.pop(df_coords.columns[0])
    node_coordinates = [tuple(coords) for coords in df_coords.values.tolist()]


    #------------------------------------------------------------#
    #------             elements.csv                       ------#
    #------------------------------------------------------------#
    # Read elements.csv to a pandas DataFrame object in order to get connectivities and a list of material ids
    df_elements = pd.read_csv(os.path.join(path_file_dir, "elements.csv"), header=0)
                    # CSV to DataFrame
    # Check if any element number is missing (in ascending order) or whether element numbering does not start at zero
    for flag in df_elements['ele_no'] == range(len(df_elements)):
       if flag == False:
            raise ValueError("[INPUT_DATA] Element numbering does not start with zero OR there is an element missing. Please check elements.csv")

    # Create a list of one material ID per element. The list index corresponds to element numbering
    material_id_per_element = df_elements['material_id'].tolist()

    # Convert DataFrame to list of tuples to create list of connectivities (dropping "ele_no" and "material_id" column).
    # Each tuple contains the connectivities of one element and the list index corresponds to element numbering.
    df_elements.pop(df_elements.columns[0])
    df_elements = df_elements.drop('material_id', axis=1)
    connectivities = [tuple(connect) for connect in df_elements.values.tolist()]


    #------------------------------------------------------------#
    #------             material_parameters.csv            ------#
    #------------------------------------------------------------#
    # Read material_parameters.csv to a pandas DataFrame object
    df_material = pd.read_csv(os.path.join(path_file_dir, "material_parameters.csv"), header=0)

    # Create a dictionary of different materials with unique IDs. The material's ID is the key and its value a dictionary of its parameters.
    df_material = df_material.set_index('material_id')
    material_ids = df_material.to_dict(orient='index')


    #------------------------------------------------------------#
    #------             element_loads.csv                  ------#
    #------------------------------------------------------------#
    # Read element_loads.csv to a pandas DataFrame object
    df_loads = pd.read_csv(os.path.join(path_file_dir, "element_loads.csv"), header=0)

    # Check whether element with given element number exists
    df_loads['ele_no'] = pd.to_numeric(df_loads['ele_no'], errors='coerce')
    if df_loads['ele_no'].isnull().sum() > 0:
        raise TypeError("[INPUT_DATA] No valid 'ele_no' given for " + str(df_loads['ele_no'].isnull().sum()) + " element load(s). Please check element_loads.csv")
    for ele_no in df_loads['ele_no']:
        if ele_no > len(df_elements)-1:
            raise IndexError("[INPUT_DATA] Element No. " + str(ele_no) + " given in element_loads.csv does not exist.")

    # Check whether given number of degree of freedom for given element type exists
    df_loads['dof_no'] = pd.to_numeric(df_loads['dof_no'], errors='coerce')
    if df_loads['dof_no'].isnull().sum() > 0:
        raise TypeError("[INPUT_DATA] No valid 'dof_no' given for " + str(df_loads['dof_no'].isnull().sum()) + " element load(s). Please check element_loads.csv")
    for dof_no in df_loads['dof_no']:
        if dof_no > len(ele_type.dof_names)-1:
            raise IndexError("[INPUT_DATA] Dof No. " + str(dof_no) + " given in element_loads.csv does not exist for given element type.")

    # Check whether "value" is given for all element loads
    df_loads['value'] = pd.to_numeric(df_loads['value'], errors='coerce')
    if df_loads['value'].isnull().sum() > 0:
        raise TypeError("[INPUT_DATA] No valid 'value' given for " + str(df_loads['value'].isnull().sum()) + " element load(s). Please check element_loads.csv")

    # Convert DataFrame to dictionary
    ele_loads = df_loads.to_dict(orient='records')


    #------------------------------------------------------------#
    #------             neumann_boundary_conditions.csv    ------#
    #------------------------------------------------------------#
    # Read neumann_boundary_conditions.csv to a pandas DataFrame object
    df_neumann_bc = pd.read_csv(os.path.join(path_file_dir, "neumann_boundary_conditions.csv"), header=0)

    # Check whether node with given node number exists
    df_neumann_bc['node_no'] = pd.to_numeric(df_neumann_bc['node_no'], errors='coerce')
    if df_neumann_bc['node_no'].isnull().sum() > 0:
        raise TypeError("[INPUT_DATA] No valid 'node_no' given for " + str(df_neumann_bc['node_no'].isnull().sum()) + " Neumann boundary condition(s). Please check neumann_boundary_conditions.csv")
    for node_no in df_neumann_bc['node_no']:
        if node_no > len(node_coordinates)-1:
            raise IndexError("[INPUT_DATA] Node No. " + str(node_no) + " given in neumann_boundary_conditions.csv does not exist.")

    # Check whether given number of degree of freedom for given element type exists
    df_neumann_bc['dof_no'] = pd.to_numeric(df_neumann_bc['dof_no'], errors='coerce')
    if df_neumann_bc['dof_no'].isnull().sum() > 0:
        raise TypeError("[INPUT_DATA] No valid 'dof_no' given for " + str(df_neumann_bc['dof_no'].isnull().sum()) + " Neumann boundary condition(s). Please check neumann_boundary_conditions.csv")
    for dof_no in df_neumann_bc['dof_no']:
        if dof_no > len(ele_type.dof_names)-1:
            raise IndexError("[INPUT_DATA] Dof No. " + str(dof_no) + " given in neumann_boundary_conditions.csv does not exist for given element type.")

    # Check whether "value" is given for all Neumann boundary conditions
    df_neumann_bc['value'] = pd.to_numeric(df_neumann_bc['value'], errors='coerce')
    if df_neumann_bc['value'].isnull().sum() > 0:
        raise TypeError("[INPUT_DATA] No valid 'value' given for " + str(df_neumann_bc['value'].isnull().sum()) + " Neumann boundary condition(s). Please check neumann_boundary_conditions.csv")

    # Convert DataFrame to dictionary
    neumann_boundary_conditions = df_neumann_bc.to_dict(orient='records')


    #------------------------------------------------------------#
    #------             dirichlet_boundary_conditions.csv  ------#
    #------------------------------------------------------------#
    # Read dirichlet_boundary_conditions.csv to a pandas DataFrame object
    df_dirichlet_bc = pd.read_csv(os.path.join(path_file_dir, "dirichlet_boundary_conditions.csv"), header=0)

    # Check whether node with given node number exists
    df_dirichlet_bc['node_no'] = pd.to_numeric(df_dirichlet_bc['node_no'], errors='coerce')
    if df_dirichlet_bc['node_no'].isnull().sum() > 0:
        raise TypeError("[INPUT_DATA] No valid 'node_no' given for " + str(df_dirichlet_bc['node_no'].isnull().sum()) + " Dirichlet boundary condition(s). Please check dirichlet_boundary_conditions.csv")
    for node_no in df_dirichlet_bc['node_no']:
        if node_no > len(node_coordinates)-1:
            raise IndexError("[INPUT_DATA] Node No. " + str(node_no) + " given in dirichlet_boundary_conditions.csv does not exist.")

    # Check whether given number of degree of freedom for given element type exists
    df_dirichlet_bc['dof_no'] = pd.to_numeric(df_dirichlet_bc['dof_no'], errors='coerce')
    if df_dirichlet_bc['dof_no'].isnull().sum() > 0:
        raise TypeError("[INPUT_DATA] No valid 'dof_no' given for " + str(df_dirichlet_bc['dof_no'].isnull().sum()) + " Dirichlet boundary condition(s). Please check dirichlet_boundary_conditions.csv")
    for dof_no in df_dirichlet_bc['dof_no']:
        if dof_no > len(ele_type.dof_names)-1:
            raise IndexError("[INPUT_DATA] Dof No. " + str(dof_no) + " given in dirichlet_boundary_conditions.csv does not exist for given element type.")

    # Check whether "value" is given for all Neumann boundary conditions
    df_dirichlet_bc['value'] = pd.to_numeric(df_dirichlet_bc['value'], errors='coerce')
    if df_dirichlet_bc['value'].isnull().sum() > 0:
        raise TypeError("[INPUT_DATA] No valid 'value' given for " + str(df_dirichlet_bc['value'].isnull().sum()) + " Dirichlet boundary condition(s). Please check dirichlet_boundary_conditions.csv")

    # Convert DataFrame to dictionary
    dirichlet_boundary_conditions = df_dirichlet_bc.to_dict(orient='records')


    #------------------------------------------------------------#
    #------             CREATE ELEMENTS AND DATA DICT      ------#
    #------------------------------------------------------------#
    # Create list of elements depending on input parameters
    elements = []
    for ele_no, connect in enumerate(connectivities):
        coords = []
        # Get node coordinates for all global node numbers "node_no" of the nodes of the element stored in "connect"
        for node_no in connect:
            # if node number can not be converted to integer it is "nan" and will be ignored
            try:
                node_no = int(node_no)
                try:
                    coords.append( node_coordinates[node_no] )
                except:
                    raise IndexError("[INPUT_DATA] Node No. " + str(node_no) + " that was given for element No. " + str(ele_no) + " does not exist in node coordinates. Please check elements.csv.")
            except:
                print("\nWarning: [INPUT_DATA] No valid node number given for connectivity of element No. " + str(ele_no) + ". Please check elements.csv.")
        # Get material parameters for the element from the list of available material IDs
        ele_material_id = material_id_per_element[ele_no]
        try:
            ele_material_params = material_ids[ele_material_id]
        except:
            raise IndexError("[INPUT_DATA] Material ID '" + str(ele_material_id) + "' that was given for element No. " + str(ele_no) + " does not exist in material_parameters.csv.")
        elements.append( ele_type(ele_no, coords, ele_material_params) )

    # Create data dictionary
    data = {"settings":settings, "node_coordinates":node_coordinates, "connectivities":connectivities, "elements":elements,
            "element_loads":ele_loads, "neumann_boundary_conditions":neumann_boundary_conditions, "dirichlet_boundary_conditions":dirichlet_boundary_conditions}
    print("DONE")
    return data