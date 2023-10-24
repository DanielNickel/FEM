#----------ISD | TU Braunschweig----------#
#-----------Beethovenstraße 51------------#
#-----------38106 Braunschweig------------#
#-----------------------------------------#

import numpy as np


class PlaneStress2D4N:
    n_dimensions = 2                                                    # 2D element type
    p_order_of_shape_functions = 1                                      # polynomial order of the shape functions used for the approximation of an element's unknowns
    n_nodes_per_element = 4                                             # number of nodes of one element of this type
    dof_names = ["u", "v"]                                              # names of the degrees of freedom at a node for an element of this type (here: displacements u in x-direction and v in y-direction)
    p_order_of_resultants = [1, 1, 1]                                   # polynomial order of an element's resultants (here: stresses fluxes sigma_xx and sigma_yy and sigma_xy are linear -> p_order = 1))
    material_param_names = ["e_modulus", "poisson_ratio", "thickness"]  # names of the material parameters needed for an element of this type
                                                                        # (here: Young's modulus E, Poisson's ratio "nu" and the Slab's thickness t)
    n_dofs_per_element = n_nodes_per_element * len(dof_names)

    def __init__(self, ele_no, node_coords, material):
        """
        Args:
            ele_no (int):                           number of the element
            node_coords (list of tuple of float):   coordinates of the element's nodes
                                                        e.g. [(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0)]
            material (dict of string:float):        key "e_modulus" should give young's modulus of the element's material [N/m^2]
                                                    key "poisson_ratio" should give the Poisson ratio of the elements' material [-]
                                                    key "thickness" should give the Slab's thickness t in [m]
                                                        e.g. {"e_modulus":10000.0, "poisson_ratio":0.3, "thickness":0.1}
        """
        self.ele_no = ele_no
        self.node_coords = node_coords
        self.material = material

        # Check number of nodes
        if len(self.node_coords) != self.n_nodes_per_element:
            raise Exception("[PlaneStress2D4N] Element No. " + str(self.ele_no) + ": Number of nodes should be " + str(self.n_nodes_per_element) + " not " + str(len(self.node_coords)) + ".")

        # Check availability of material parameters
        if not "e_modulus" in self.material:
            raise KeyError("[PlaneStress2D4N] Element No. " + str(self.ele_no) + ": Material parameter 'e_modulus' needed but not given.")
        if not "poisson_ratio" in self.material:
            raise KeyError("[PlaneStress2D4N] Element No. " + str(self.ele_no) + ": Material parameter 'poisson_ratio' needed but not given.")
        if not "thickness" in self.material:
            raise KeyError("[PlaneStress2D4N] Element No. " + str(self.ele_no) + ": Material parameter 'thickness' needed but not given.")
        try:
            self.material["e_modulus"] = float(self.material["e_modulus"])
        except:
            raise TypeError("[PlaneStress2D4N] Element No. " + str(self.ele_no) + ": Value for material parameter 'e_modulus' needed but '" + str(self.material["e_modulus"]) + "' is given. Please check material_parameters.csv.")
        if np.isnan(self.material["e_modulus"]):
            raise TypeError("[PlaneStress2D4N] Element No. " + str(self.ele_no) + ": Value for parameter 'e_modulus' needed but not given. Please check material_parameters.csv.")
        try:
            self.material["poisson_ratio"] = float(self.material["poisson_ratio"])
        except:
            raise TypeError("[PlaneStress2D4N] Element No. " + str(self.ele_no) + ": Value for material parameter 'poisson_ratio' needed but '" + str(self.material["poisson_ratio"]) + "' is given. Please check material_parameters.csv.")
        if np.isnan(self.material["poisson_ratio"]):
            raise TypeError("[PlaneStress2D4N] Element No. " + str(self.ele_no) + ": Value for parameter 'poisson_ratio' needed but not given. Please check material_parameters.csv.")
        try:
            self.material["thickness"] = float(self.material["thickness"])
        except:
            raise TypeError("[PlaneStress2D4N] Element No. " + str(self.ele_no) + ": Value for material parameter 'thickness' needed but '" + str(self.material["thickness"]) + "' is given. Please check material_parameters.csv.")
        if np.isnan(self.material["thickness"]):
            raise TypeError("[PlaneStress2D4N] Element No. " + str(self.ele_no) + ": Value for parameter 'thickness' needed but not given. Please check material_parameters.csv.")

    def get_eleNo_and_node_coords(self):
        return self.ele_no, self.node_coords

    def build_element_stiffness_matrix(self):
        """
        This function builds the element stiffness matrix using a local coordinate system (xi, eta),
        the iso-parametric element concept and Gaussian quadrature for numerical integration.
        Discretization utilizes bi-linear shape functions.

        Returns:
            K_element (np.array):                   element stiffness matrix of size [n_dof_per_element x n_dof_per_element] [N/m]
        """
        # Get material parameters
        e_modulus = self.material["e_modulus"]
        nu = self.material["poisson_ratio"]
        t = self.material["thickness"]

        E_element = t * e_modulus/(1-nu**2) * np.array([[ 1, nu, 0],
                                                        [nu,  1, 0],
                                                        [ 0,  0, 0.5*(1-nu)]])
        
        E_element_sigma = t * e_modulus/(1-nu**2) * np.array([[ 1, nu, 0],
                                                        [nu,  1, 0],
                                                        [ 0,  0, 0]])
        
        E_element_tau = t * e_modulus/(1-nu**2) * np.array([[ 0, 0, 0],
                                                        [0,  0, 0],
                                                        [ 0,  0, 0.5*(1-nu)]])        
        

        # Initialize element stiffness matrix
        K_element = np.zeros( (self.n_dofs_per_element, self.n_dofs_per_element) )
        K_element_sigma = np.zeros( (self.n_dofs_per_element, self.n_dofs_per_element) )
        K_element_tau = np.zeros( (self.n_dofs_per_element, self.n_dofs_per_element) )

        # Format node coordinates as numpy matrix
        x_tilde = np.array(self.node_coords)

        # Gauss points
        gauss_points = [-1.0 / np.sqrt(3), 1.0 / np.sqrt(3)]
        gauss_point_SRI = 0

        # Gauss weights
        gauss_weights = [1.0, 1.0]
        gauss_weight_SRI = 2

        # Calculate element matrix as a sum of the integrant evaluated at the Gauss points
        for xi, w_xi in zip(gauss_points, gauss_weights):
            for eta, w_eta in zip(gauss_points, gauss_weights):

                # Define derivatives of the shape functions
                Phi_dXI = np.array([[-0.25*(1-eta), 0.25*(1-eta), 0.25*(1+eta), -0.25*(1+eta)],[-0.25*(1-xi), -0.25*(1+xi), 0.25*(1+xi), 0.25*(1-xi)]])
                # Calculate the Jacobian matrix, its inverse and determinant (Section 3.1.2 in handout)
                J =np.dot(Phi_dXI, x_tilde)
                detJ =np.linalg.det(J)
                invJ =(1/detJ)*np.array([[J[1,1],-J[0,1]],[-J[1,0],J[0,0]]])

                # Calculate the H matrix by transforming the shape function derivatives from local to global coordinate system (Section 3.2.2 in handout)
                # H = D * Omega, a transformation matrix is used for easier multiplication with the Jacobian
                transformation_matrix =np.array([[1, 0, 0, 0],[0, 0, 0, 1], [0, 1, 1, 0]])

                J_for_H = np.zeros((4, 4))

                # Fill the diagonal with two times the 2x2 matrix A
                J_for_H[:2, :2] = invJ

                # Copy the values of A to the remaining diagonal elements
                J_for_H[2:, 2:] = invJ

                Omega_dXI = np.array([[-0.25*(1-eta),0,0.25*(1-eta),0,0.25*(1+eta),0,-0.25*(1+eta),0],
                                      [-0.25*(1-xi),0,-0.25*(1+xi),0,0.25*(1+xi),0,0.25*(1-xi),0],
                                      [0,-0.25*(1-eta),0,0.25*(1-eta),0,0.25*(1+eta),0,-0.25*(1+eta)],
                                      [0,-0.25*(1-xi),0,-0.25*(1+xi),0,0.25*(1+xi),0,0.25*(1-xi)]])

                H = np.dot(transformation_matrix, np.dot(J_for_H, Omega_dXI))

                # Calculate the element stiffness matrix K_element for the Gauss point (Section 4.1 & 5.1 in handout)
                # K_element = numerical integration in local coordinate system of H^T * E * H
                K_element += np.dot(np.dot(np.transpose(H),E_element),H)*detJ*w_xi*w_eta
                K_element_sigma += np.dot(np.dot(np.transpose(H),E_element_sigma),H)*detJ*w_xi*w_eta

        #SRI Calculation
        Phi_dXI_SRI =np.array([[0.25*(1-gauss_point_SRI),-0.25*(1-gauss_point_SRI),-0.25*(1+gauss_point_SRI),0.25*(1+gauss_point_SRI)],
                                  [0.25*(1-gauss_point_SRI),0.25*(1+gauss_point_SRI),-0.25*(1+gauss_point_SRI),-0.25*(1-gauss_point_SRI)]])
        J_SRI =np.dot(Phi_dXI_SRI, x_tilde)
        detJ_SRI =np.linalg.det(J_SRI)
        invJ_SRI = (1/detJ_SRI)*np.array([[J_SRI[1,1],-J_SRI[0,1]],[-J_SRI[1,0],J_SRI[0,0]]])
        transformation_matrix_SRI =np.array([[1,0,0,0],
                                                 [0,0,0,1],
                                                 [0,1,1,0]])

        J_for_H_SRI = np.array([[invJ_SRI[0,0],invJ_SRI[0,1],0,0],
                            [invJ_SRI[1,0],invJ_SRI[1,1],0,0],
                            [0,0,invJ_SRI[0,0],invJ_SRI[0,1]],
                            [0,0,invJ_SRI[1,0],invJ_SRI[1,1]]])

        Omega_dXI_SRI = np.array([[-0.25*(1-gauss_point_SRI),0,0.25*(1-gauss_point_SRI),0,0.25*(1+gauss_point_SRI),0,-0.25*(1+gauss_point_SRI),0],
                              [-0.25*(1-gauss_point_SRI),0,-0.25*(1+gauss_point_SRI),0,0.25*(1+gauss_point_SRI),0,0.25*(1-gauss_point_SRI),0],
                              [0,-0.25*(1-gauss_point_SRI),0,0.25*(1-gauss_point_SRI),0,0.25*(1+gauss_point_SRI),0,-0.25*(1+gauss_point_SRI)],
                              [0,-0.25*(1-gauss_point_SRI),0,-0.25*(1+gauss_point_SRI),0,0.25*(1+gauss_point_SRI),0,0.25*(1-gauss_point_SRI)]])

        H_SRI = np.dot(np.dot(transformation_matrix_SRI,J_for_H_SRI),Omega_dXI_SRI)

        K_element_tau=np.dot(np.dot(np.transpose(H_SRI),E_element_tau),H_SRI)*detJ_SRI*gauss_weight_SRI

        print("E_element_tau:")
        print(E_element_tau)

        print("K_element_tau for Element no.", self.ele_no)
        print(K_element_sigma +K_element_tau)
        
            
        #return (K_element_sigma + K_element_tau)
        return K_element

    def build_element_load_vector(self, dof_numbers, load_values):
        """
        This function builds the element load vector for a constant surface load using a local coordinate system (xi, eta)
        and Gaussian quadrature for numerical integration.

        Args:
            dof_numbers (list of int):              number of the degree of freedom the load refers to, should be 0 for "u" (x-direction) and 1 for "v"
            load_values (list of float):            values of the load [N/m^2]

        Returns:
            f_element (np.array):                   element load vector of [size n_dof_per_element x 1] [N]
        """
        # Initialize load vector
        f_element = np.zeros( (self.n_dofs_per_element, 1) )

        # Format node coordinates as numpy matrix
        x_tilde = np.array(self.node_coords)

        # Format and sum up load values for x- and y-direction as column vector
        p = np.zeros( (len(self.dof_names), 1) )
        for dof_no, value in zip(dof_numbers, load_values):
            p[dof_no, 0] += value

        # Gauss points
        gauss_points = [-1.0 / np.sqrt(3), 1.0 / np.sqrt(3)]

        # Gauss weights
        gauss_weights = [1.0 , 1.0]

        # Calculate load vector as a sum of the integrant evaluated at the Gauss points
        # Loop through all Gauss points (n_GP = n_GP in xi-direction * n_GP in eta-direction)
        for xi, w_xi in zip(gauss_points, gauss_weights):
            for eta, w_eta in zip(gauss_points, gauss_weights):

                # Define shape functions
                phi_A = 0.25 * (1 - xi) * (1 - eta)
                phi_B = 0.25 * (1 + xi) * (1 - eta)
                phi_C = 0.25 * (1 + xi) * (1 + eta)
                phi_D = 0.25 * (1 - xi) * (1 + eta)      
                # Define derivatives of the shape functions
                Phi_dXI = np.array([[-0.25*(1-eta), 0.25*(1-eta), 0.25*(1+eta), -0.25*(1+eta)],[-0.25*(1-xi), -0.25*(1+xi), 0.25*(1+xi), 0.25*(1-xi)]])

                # Calculate the Jacobian matrix and its determinant (Section 3.1.2 in handout)
                J =np.dot(Phi_dXI, x_tilde)
                detJ =np.linalg.det(J)

                # Define shape function matrix (Section 3.2.1 in handout)
                Omega = np.array([[phi_A,0,phi_B,0,phi_C,0,phi_D,0],
                                  [0,phi_A,0,phi_B,0,phi_C,0,phi_D]])

                # Calculate the element load vector f_element for the Gauss point (Section 4.2 & 5.2 in handout)
                f_element +=np.dot(np.transpose(Omega),p)*detJ*w_xi*w_eta

        return f_element

    def calculate_element_resultants(self, v_element):
        """
        This function calculates the element's resultants for each node of the element.
        For this element the resultants are heat stresses in x- and y-direction and shear stress, which are calculated for each node.

        Args:
            v_element (np.array):                       the nodal unknowns of the element (displacements at the nodes) after which was solved [m]
                                                            here: [u_A, v_A, u_B, v_B, u_C, v_C, u_D, v_D]^T

        Returns:
            sigma_xx, sigma_yy, sigma_xy (dict):        vectors of all the values of all resultants for every node of the element, ordered by direction [N/m^2]
                                                            here: {"sigma_xx":[sigma_xxA, sigma_xxB, sigma_xxC, sigma_xxD]^T,
                                                                   "sigma_yy":[sigma_yyA, sigma_yyB, sigma_yyC, sigma_yyD]^T
                                                                   "sigma_xy":[sigma_xyA, sigma_xyB, sigma_xyC, sigma_xyD]^T}
        """
        # Node coordinates, global and local
        x_tilde = np.array(self.node_coords)
        xi_per_node = [-1, 1, 1, -1]
        eta_per_node = [-1, -1, 1, 1]

        # Get material parameters and insert into elasticity matrix
        e_modulus = self.material["e_modulus"]
        nu = self.material["poisson_ratio"]
        t = self.material["thickness"]

        E_element = e_modulus/(1-nu**2) * np.array([[ 1, nu, 0],
                                                    [nu,  1, 0],
                                                    [ 0,  0, 0.5*(1-nu)]])

        # Initialize the element's stress vectors as one value per node
        sigma_xx = np.zeros( (self.n_nodes_per_element, 1) )
        sigma_yy = np.zeros( (self.n_nodes_per_element, 1) )
        sigma_xy = np.zeros( (self.n_nodes_per_element, 1) )

        # Loop through the nodes of the element to build the element stress vectors (Section 7 in handout)
        for it_node in range(self.n_nodes_per_element):

            # Get xi and eta for that node
            xi = xi_per_node[it_node]
            eta = eta_per_node[it_node]

            # Define derivatives of the shape functions
            Phi_dXI = np.array([[-0.25*(1-eta), 0.25*(1-eta), 0.25*(1+eta), -0.25*(1+eta)],[-0.25*(1-xi), -0.25*(1+xi), 0.25*(1+xi), 0.25*(1-xi)]])

            # Calculate the Jacobian matrix and its inverse (Section 3.1.2 in handout)
            J =np.dot(Phi_dXI, x_tilde)
            detJ =np.linalg.det(J)
            invJ =(1/detJ)*np.array([[J[1,1],-J[0,1]],[-J[1,0],J[0,0]]])

            # Calculate the H matrix by transforming the shape function derivatives from local to global coordinate system (Section 3.2.2 in handout)
            # H = D * Omega, a transformation matrix is used for easier multiplication with the Jacobian
            transformation_matrix =np.array([[1, 0, 0, 0],[0, 0, 0, 1], [0, 1, 1, 0]])

            J_for_H = np.zeros((4, 4))

            # Fill the diagonal with two times the 2x2 matrix A
            J_for_H[:2, :2] = invJ

            # Copy the values of A to the remaining diagonal elements
            J_for_H[2:, 2:] = invJ

            Omega_dXI = np.array([[-0.25*(1-eta),0,0.25*(1-eta),0,0.25*(1+eta),0,-0.25*(1+eta),0],
                                      [-0.25*(1-xi),0,-0.25*(1+xi),0,0.25*(1+xi),0,0.25*(1-xi),0],
                                      [0,-0.25*(1-eta),0,0.25*(1-eta),0,0.25*(1+eta),0,-0.25*(1+eta)],
                                      [0,-0.25*(1-xi),0,-0.25*(1+xi),0,0.25*(1+xi),0,0.25*(1-xi)]])

            H = np.dot(np.dot(transformation_matrix,J_for_H), Omega_dXI)

            # Calculate the stress matrix, evaluated at the node
            S = np.dot(E_element, H)

            # Calculate the vector of stresses for the node
            sigma_at_node = np.dot(S, v_element)

            # Add values of the node to the element's heat flux vectors
            sigma_xx[it_node] = sigma_at_node[0]
            sigma_yy[it_node] = sigma_at_node[1]
            sigma_xy[it_node] = sigma_at_node[2]

        return {"sigma_xx":sigma_xx, "sigma_yy":sigma_yy, "sigma_xy":sigma_xy}
    
    def calculate_element_resultants_SRI(self, v_element):
        """
        This function calculates the element's resultants for each node of the element.
        For this element the resultants are heat stresses in x- and y-direction and shear stress, which are calculated for each node.

        Args:
            v_element (np.array):                       the nodal unknowns of the element (displacements at the nodes) after which was solved [m]
                                                            here: [u_A, v_A, u_B, v_B, u_C, v_C, u_D, v_D]^T

        Returns:
            sigma_xx, sigma_yy, sigma_xy (dict):        vectors of all the values of all resultants for every node of the element, ordered by direction [N/m^2]
                                                            here: {"sigma_xx":[sigma_xxA, sigma_xxB, sigma_xxC, sigma_xxD]^T,
                                                                   "sigma_yy":[sigma_yyA, sigma_yyB, sigma_yyC, sigma_yyD]^T
                                                                   "sigma_xy":[sigma_xyA, sigma_xyB, sigma_xyC, sigma_xyD]^T}
        """
        # Node coordinates, global and local
        x_tilde = np.array(self.node_coords)
        xi_per_node = [-1, 1, 1, -1]
        eta_per_node = [-1, -1, 1, 1]

        # Get material parameters and insert into elasticity matrix
        e_modulus = self.material["e_modulus"]
        nu = self.material["poisson_ratio"]
        t = self.material["thickness"]

        E_element = e_modulus/(1-nu**2) * np.array([[ 1, nu, 0],
                                                    [nu,  1, 0],
                                                    [ 0,  0, 0.5*(1-nu)]])

        # Initialize the element's stress vectors as one value per node
        sigma_xx = np.zeros( (self.n_nodes_per_element, 1) )
        sigma_yy = np.zeros( (self.n_nodes_per_element, 1) )
        sigma_xy = np.zeros( (self.n_nodes_per_element, 1) )

        # Loop through the nodes of the element to build the element stress vectors (Section 7 in handout)
        for it_node in range(self.n_nodes_per_element):

            # Get xi and eta for that node
            xi = xi_per_node[it_node]
            eta = eta_per_node[it_node]

            # Define derivatives of the shape functions
            Phi_dXI = np.array([[-0.25*(1-eta), 0.25*(1-eta), 0.25*(1+eta), -0.25*(1+eta)],
                                [-0.25*(1-xi), -0.25*(1+xi), 0.25*(1+xi), 0.25*(1-xi)]])

            # Calculate the Jacobian matrix and its inverse (Section 3.1.2 in handout)
            J =np.dot(Phi_dXI, x_tilde)
            detJ =np.linalg.det(J)
            invJ =(1/detJ)*np.array([[J[1,1],-J[0,1]],[-J[1,0],J[0,0]]])

            # Calculate the H matrix by transforming the shape function derivatives from local to global coordinate system (Section 3.2.2 in handout)
            # H = D * Omega, a transformation matrix is used for easier multiplication with the Jacobian
            transformation_matrix =np.array([[1, 0, 0, 0],[0, 0, 0, 1], [0, 1, 1, 0]])

            J_for_H = np.zeros((4, 4))

            # Fill the diagonal with two times the 2x2 matrix A
            J_for_H[:2, :2] = invJ

            # Copy the values of A to the remaining diagonal elements
            J_for_H[2:, 2:] = invJ

            Omega_dXI = np.array([[-0.25*(1-eta),0,0.25*(1-eta),0,0.25*(1+eta),0,-0.25*(1+eta),0],
                                      [-0.25*(1-xi),0,-0.25*(1+xi),0,0.25*(1+xi),0,0.25*(1-xi),0],
                                      [0,-0.25*(1-eta),0,0.25*(1-eta),0,0.25*(1+eta),0,-0.25*(1+eta)],
                                      [0,-0.25*(1-xi),0,-0.25*(1+xi),0,0.25*(1+xi),0,0.25*(1-xi)]])

            H = np.dot(np.dot(transformation_matrix,J_for_H), Omega_dXI)

            # Calculate the stress matrix, evaluated at the node
            S = np.dot(E_element, H)

            # Calculate the vector of stresses for the node
            sigma_at_node = np.dot(S, v_element)

            # Add values of the node to the element's heat flux vectors
            sigma_xx[it_node] = sigma_at_node[0]
            sigma_yy[it_node] = sigma_at_node[1]
            #sigma_xy[it_node] = sigma_at_node[2]

    # Get xi and eta for that node
        xi = 0
        eta = 0

        # Define derivatives of the shape functions
        Phi_dXI = np.array([[-0.25*(1-eta), 0.25*(1-eta), 0.25*(1+eta), -0.25*(1+eta)],[-0.25*(1-xi), -0.25*(1+xi), 0.25*(1+xi), 0.25*(1-xi)]])

        # Calculate the Jacobian matrix and its inverse (Section 3.1.2 in handout)
        J =np.dot(Phi_dXI, x_tilde)
        detJ =np.linalg.det(J)
        invJ =(1/detJ)*np.array([[J[1,1],-J[0,1]],[-J[1,0],J[0,0]]])

        # Calculate the H matrix by transforming the shape function derivatives from local to global coordinate system (Section 3.2.2 in handout)
        # H = D * Omega, a transformation matrix is used for easier multiplication with the Jacobian
        transformation_matrix =np.array([[1, 0, 0, 0],[0, 0, 0, 1], [0, 1, 1, 0]])

        J_for_H = np.zeros((4, 4))

        # Fill the diagonal with two times the 2x2 matrix A
        J_for_H[:2, :2] = invJ

        # Copy the values of A to the remaining diagonal elements
        J_for_H[2:, 2:] = invJ

        Omega_dXI = np.array([[-0.25*(1-eta),0,0.25*(1-eta),0,0.25*(1+eta),0,-0.25*(1+eta),0],
                                    [-0.25*(1-xi),0,-0.25*(1+xi),0,0.25*(1+xi),0,0.25*(1-xi),0],
                                    [0,-0.25*(1-eta),0,0.25*(1-eta),0,0.25*(1+eta),0,-0.25*(1+eta)],
                                    [0,-0.25*(1-xi),0,-0.25*(1+xi),0,0.25*(1+xi),0,0.25*(1-xi)]])

        H = np.dot(np.dot(transformation_matrix,J_for_H), Omega_dXI)

        # Calculate the stress matrix, evaluated at the node
        S = np.dot(E_element, H)

        # Calculate the vector of stresses for the node
        sigma_at_node = np.dot(S, v_element)
    
        sigma_xy = np.full((self.n_nodes_per_element,1), sigma_at_node[2]) 

        return {"sigma_xx":sigma_xx, "sigma_yy":sigma_yy, "sigma_xy":sigma_xy}
