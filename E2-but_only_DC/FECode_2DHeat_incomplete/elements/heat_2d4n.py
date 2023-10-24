#----------ISD | TU Braunschweig----------#
#-----------Beethovenstraße 51------------#
#-----------38106 Braunschweig------------#
#-----------------------------------------#

import numpy as np


class Heat2D4N:
    n_dimensions = 2                                    # 2D element type
    p_order_of_shape_functions = 1                      # polynomial order of the shape functions used for the approximation of an element's unknowns
    n_nodes_per_element = 4                             # number of nodes of one element of this type
    dof_names = ["T"]                                   # names of the degrees of freedom at a node for an element of this type (here: temperature T)
    p_order_of_resultants = [0, 0]                      # polynomial order of an element's resultants (here: heat fluxes q_x and q_y are const -> p_order = 0))
    material_param_names = ["lambda_x", "lambda_y"]     # names of the material parameters needed for an element of this type
                                                        # (here: Conductivity lambda for x- and y-direction)
    n_dofs_per_element = n_nodes_per_element * len(dof_names)

    def __init__(self, ele_no, node_coords, material):
        """
        Args:
            ele_no (int):                           number of the element
            node_coords (list of tuple of float):   coordinates of the element's nodes
                                                        e.g. [(-1.0, 1.0), (1.0, 1.0), (1.0, -1.0), (-1.0, -1.0)]
            material (dict of string:float):        key "lambda_x" should give the conductivity in x-direction
                                                        key "lambda_y" should give the conductivity in y-direction
                                                        e.g. {"lambda_x":6, "lambda_y":7}
        """
        self.ele_no = ele_no
        self.node_coords = node_coords
        self.material = material

        # Check number of nodes
        if len(self.node_coords) != self.n_nodes_per_element:
            raise Exception("[Heat2D4N] Element No. " + str(self.ele_no) + ": Number of nodes should be " + str(self.n_nodes_per_element) + " not " + str(len(self.node_coords)) + ".")

        # Check availability of material parameters
        if not "lambda_x" in self.material:
            raise KeyError("[Heat2D4N] Element No. " + str(self.ele_no) + ": Material parameter 'lambda_x' needed but not given.")
        if not "lambda_y" in self.material:
            raise KeyError("[Heat2D4N] Element No. " + str(self.ele_no) + ": Material parameter 'lambda_y' needed but not given.")
        try:
            self.material["lambda_x"] = float(self.material["lambda_x"])
        except:
            raise TypeError("[Heat2D4N] Element No. " + str(self.ele_no) + ": Value for material parameter 'lambda_x' needed but '" + str(self.material["lambda_x"]) + "' is given. Please check material_parameters.csv.")
        if np.isnan(self.material["lambda_x"]):
            raise TypeError("[Heat2D4N] Element No. " + str(self.ele_no) + ": Value for parameter 'lambda_x' needed but not given. Please check material_parameters.csv.")
        try:
            self.material["lambda_y"] = float(self.material["lambda_y"])
        except:
            raise TypeError("[Heat2D4N] Element No. " + str(self.ele_no) + ": Value for material parameter 'lambda_y' needed but '" + str(self.material["lambda_y"]) + "' is given. Please check material_parameters.csv.")
        if np.isnan(self.material["lambda_y"]):
            raise TypeError("[Heat2D4N] Element No. " + str(self.ele_no) + ": Value for parameter 'lambda_y' needed but not given. Please check material_parameters.csv.")

    def build_element_stiffness_matrix(self):
        """
        This function builds the element stiffness matrix using a local coordinate system (xi, eta),
        the isoparametric element concept and Gaussian quadrature for numerical integration.

        Returns:
            K_element (np.array):                   element stiffness matrix of size [n_dof_per_element x n_dof_per_element]
        """
        # Format node coordinates as numpy matrix
        x_tilde = np.array(self.node_coords)

        # Get material parameters
        lambda_x = self.material["lambda_x"]
        lambda_y = self.material["lambda_y"]

        E_element = np.array([[lambda_x, 0],
                              [0, lambda_y]])

        # Initialize element stiffness matrix
        K_element = np.zeros( (self.n_dofs_per_element, self.n_dofs_per_element) )

        # Gauss points
        gauss_points = [-1.0 / np.sqrt(3), 1.0 / np.sqrt(3)]

        # Gauss weights
        weights = [1.0 , 1.0]

        # Calculate element matrix as a sum of the integrant evaluated at the Gauss points
        for xi, w_xi in zip(gauss_points, weights):
            for eta, w_eta in zip(gauss_points, weights):

                # Define shape function matrix
                Omega = np.array([0.25*(1-xi)*(1-eta), 0.25*(1+xi)*(1-eta), 0.25*(1+xi)*(1+eta), 0.25*(1-xi)*(1+eta)])

                # Define matrix of derivatives of the shape functions w.r.t the local coordinate system
                Omega_dXI = np.array([[0.25*(1-eta), -0.25*(1-eta), -0.25*(1+eta), 0.25*(1+eta)],[0.25*(1-xi), 0.25*(1+xi), -0.25*(1+xi), -0.25*(1-xi)]])

                # Calculate the Jacobian matrix, its inverse and determinant (Section 5.1.2 in handout)
                J =np.dot(Omega_dXI, x_tilde)
                detJ =np.linalg.det(J)
                invJ =(1/detJ)*np.array([[J[1,1],-J[0,1]],[-J[1,0],J[0,0]]])

                # Calculate the H matrix by transforming the shape function derivatives from local to global coordinate system (Section 5.2.2 in handout)
                # H = J^(-1) * D * Omega
                H = np.dot(invJ, Omega_dXI)
                
                # Calculate the element stiffness matrix K_element for the Gauss point (Section 6.1 & 7.1 in handout)
                # K_element = numerical integration in local coordinate system of H^T * E * H


                K_element +=np.dot(np.dot(np.transpose(H),E_element),H)*detJ*w_xi*w_eta
        return K_element

    def build_element_load_vector(self, dof_numbers, load_values):
        """
        This function builds the element load vector for a constant heat source using a local coordinate system (xi, eta)
        and Gaussian quadrature for numerical integration.

        Args:
            dof_numbers (list of int):              number of the degree of freedom the load refers to, should be 0
            load_values (list of float):            values of the load

        Returns:
            f_element (np.array):                   element load vector of [size n_dof_per_element x 1]
        """
        # Format node coordinates as numpy matrix
        x_tilde = np.array(self.node_coords)

        # Initialize load vector
        f_element = np.zeros( (self.n_dofs_per_element, 1) )

        # Gauss points
        gauss_points = [-1.0 / np.sqrt(3), 1.0 / np.sqrt(3)]

        # Gauss weights
        weights = [1.0 , 1.0]

        # Calculate load vector as a sum of the integrant evaluated at the Gauss points
        for dof_no, value in zip(dof_numbers, load_values):

            # Loop through all Gauss points (n_GP = n_GP in xi-direction * n_GP in eta-direction)
            for xi, w_xi in zip(gauss_points, weights):
                for eta, w_eta in zip(gauss_points, weights):

                    # Define shape function matrix
                    Omega =0

                    # Define matrix of derivatives of the shape functions w.r.t the local coordinate system
                    Omega_dXI =0

                    # Calculate the Jacobian matrix and its determinant (Section 5.1.2 in handout)
                    J =0
                    detJ =0

                    # Calculate the element load vector f_element for the Gauss point (Section 6.2 & 7.2 in handout)
                    f_element +=0

        return f_element

    def calculate_element_resultants(self, v_element):
        """
        This function calculates the element's resultants for each node of the element.
        For this element the resultants are heat flux in x- and y-direction, which are calculated for each node.

        Args:
            v_element (np.array):                   the nodal unknowns of the element (temperature at the nodes) after which was solved
                                                        here: [T_A, T_B, T_C, T_D]^T

        Returns:
            q_x, q_y (dict):                        vectors of all the values of all resultants for every node of the element, ordered by direction
                                                        here: {"q_x":[q_xA, q_xB, q_xC, q_xD]^T, "q_y":[q_yA, q_yB, q_yC, q_yD]^T}
        """
        # Node coordinates, global and local
        x_tilde = np.array(self.node_coords)
        xi_per_node = [-1, 1, 1, -1]
        eta_per_node = [-1, -1, 1, 1]

        # Get material parameters and insert into elasticity matrix
        lambda_x = self.material["lambda_x"]
        lambda_y = self.material["lambda_y"]

        E_element = np.array([[lambda_x, 0],
                              [0, lambda_y]])

        # Initialize the element's heat flux vectors as one heat flux value per node
        q_x = np.zeros((self.n_nodes_per_element, 1))
        q_y = np.zeros((self.n_nodes_per_element, 1))

        # Loop through the nodes of the element to build the element heat flux vectors (Section 9 in handout)
        for it_node in range(self.n_nodes_per_element):

            # Get xi and eta for that node
            xi = xi_per_node[it_node]
            eta = eta_per_node[it_node]

            # Define shape function matrix for the node
            Omega = 0

            # Define matrix of derivatives of the shape functions at the node
            Omega_dXI =0

            # Calculate the Jacobian matrix and its inverse (Section 5.1.2 in handout)
            J =0
            invJ =0

            # Calculate the H matrix by transforming the shape function derivatives from local to global coordinate system (Section 5.2.2 in handout)
            # H = J^(-1) * D * Omega
            H =0

            # Calculate the vector of heat fluxes at the node
            sigma = E_element @ H @ v_element

            # Add values of the node to the element's heat flux vectors
            q_x[it_node] =0
            q_y[it_node] =0

        return {"q_x":q_x, "q_y":q_y}
