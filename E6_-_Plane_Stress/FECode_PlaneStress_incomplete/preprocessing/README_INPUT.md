# Pre-processing

## File Structure of Input Data

- All csv files should be placed inside a folder called "input_data" inside the folder "preprocessing".
- These are the names of the required csv files:
    1. fe_settings.csv
    2. material_parameters.csv
    3. elements.csv
    4. node_coordinates.csv
    5. element_loads.csv
    6. neumann_boundary_conditions.csv
    7. dirichlet_boundary_conditions.csv

## Explanation of the CSV Input Files

1. fe_settings.csv:

    element_type             | .....
    -------------------------|----------
    Bar1D2N                  | .....

    - This file contains the Finite Element settings required for the analysis.
    - These settings apply to the entire analysis and all Finite elements
    - The attributes are eg. element_type, linearity etc.
    - For now, the setting "element_type" is sufficient.

---
2. material_parameters.csv:

    material_id | e_modulus | area  | ....
    ------------|-----------|-------|----------
    1           | 12000     | 0.001 | ....
    5           | 12000     | 0.002 | ....
    .           |   .       |   .   |
    .           |   .       |   .   |

    - This file contains a list of different materials. Each material has a unique ID.
    - material_id: A material ID for each material which can be used in elements.csv
    - e_modulus: A material's Young's modulus as an example for a material parameter.
    - area: The cross-section area as another example for a quantity assigned to a material ID.
    - Different material properties can be added and must to be specified if required by the element type, which is defined in fe_settings.csv.
    - Ideally, every material ID has unique properties to avoid redundancy.

---
3. elements.csv:

    ele_no | material_id | node_A | node_B | node_C | ....
    -------|-------------|--------|--------|--------|----------
    0      | 1           | 0      | 1      | ....   | ....
    1      | 5           | 1      | 2      | ....   | ....
    .      |    .        |  .     | .      |
    .      |    .        |  .     | .      |

    - This file defines all Finite Elements. It contains the material ID and connectivities of each element.
    - ele_no: Element numbers must start from zero and must be in ascending order without any gaps.
    - material_id: A material must be assigned to each element using a material ID. This ID must be specified in material_parameters.csv
    - node_A/B/C/...: Global node numbers for the local node names A, B, C, .. of each element must be given (connectivity matrix).

---
4. node_coordinates.csv:

    node_no | x  | y  | z
    --------|----|----|-----------
    0       | 0  | 0  | 0
    1       | 1  | 0  | 0
    2       | 2.5| 0  | 0
    .       | .  | .  |
    .       | .  | .  |

    - This file contains the coordinates of all nodes.
    - node_no: Node numbers must start from zero and must be in ascending order without any gaps.
    - x, y, z: x, y and z coordinates of each node.

---
5. element_loads.csv:

    ele_no | dof_no | value
    -------|--------|------------
    0      | 0      | 4
    1      | 0      | 6
    .      | .      | .
    .      | .      | .

    - This file contains all loads which are applied to the Finite Elements.
    - ele_no: Element number of the element on which an elemental load is applied.
    - dof_no: The number of degree of freedom that the load corresponds to.
    - value: The magnitude of the load.
    - The directions of the load that are available and the unit of a load depend on the element type.

---
6. neumann_boundary_conditions.csv:

    node_no | dof_no | value
    --------|--------|------------
    2       | 0      | 10

    - This file contains all Neumann boundary conditions.
    - node_no: The node at which the Neumann boundary condition is applied.
    - dof_no: Number of the degree of freedom at the node onto which the boundary condition is applied.
    - value: The magnitude of the boundary condition.
    - The meaning of the numbers of degree of freedom at a node depend on the element type.

---
7. dirichlet_boundary_conditions.csv:

    node_no | dof_no | value
    --------|--------|------------
    0       | 0      | 0.2

    - This file contains all Dirichlet boundary conditions.
    - node_no: The node at which the Dirichlet boundary condition is applied.
    - dof_no: The degree of freedom at the node onto which the boundary condition is applied.
    - value: The magnitude of the boundary condition.
    - The meaning of the numbers of degree of freedom at a node depend on the element type.

## Automated mesh generation
Separate functions for 1D linear and 2D linear rectangular uniform mesh generation are available. The Python script can be executed in a terminal using following commands:
- Command for 1D mesh: `python mesh_generator.py <x_min>-<x_max> <number of elements>`
- Command for 2D mesh: `python mesh_generator.py <x_min>-<x_max> <y_min>-<y_max> <number of elements in x direction> <number of elements in y direction>`
The script creates the files node_coordinates.csv and elements.csv with material_ID 0 as default.
