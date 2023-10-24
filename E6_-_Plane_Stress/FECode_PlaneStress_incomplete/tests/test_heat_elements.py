#----------ISD | TU Braunschweig----------#
#-----------Beethovenstraße 51------------#
#-----------38106 Braunschweig------------#
#-----------------------------------------#

import unittest
import numpy as np

import sys, os
path_file_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(path_file_dir, '..'))

from elements.heat_2d4n import Heat2D4N


class TestHeatElements(unittest.TestCase):

    def test_heat2d4n_element_scaled(self):
        # Check variables of element type
        self.assertEqual(Heat2D4N.n_dimensions, 2)
        self.assertEqual(Heat2D4N.p_order_of_shape_functions, 1)
        self.assertEqual(Heat2D4N.n_nodes_per_element, 4)
        self.assertEqual(Heat2D4N.dof_names, ["T"])
        self.assertEqual(Heat2D4N.p_order_of_resultants, [1, 1])
        self.assertEqual(Heat2D4N.material_param_names, ["lambda_x", "lambda_y"])

        # Pre-processing
        element1 = Heat2D4N(0, [[1,2], [4,2], [4,3.5], [1,3.5]], {"lambda_x":6, "lambda_y":6})

        # Assemble system
        K_element = element1.build_element_stiffness_matrix()
        f_element = element1.build_element_load_vector([0], [4])

        # Check system matrix and load vector
        self.assertListEqual(np.around(K_element,10).tolist(), [[5,1,-2.5,-3.5], [1,5,-3.5,-2.5], [-2.5,-3.5,5,1], [-3.5,-2.5,1,5]])
        self.assertListEqual(np.around(f_element,10).tolist(), [[4.5], [4.5], [4.5], [4.5]])

        # Calculate and check resultants
        res_element = element1.calculate_element_resultants(np.array([[1],[3],[10],[6]]))
        self.assertListEqual(np.around(res_element["q_x"],10).tolist(), [[-4],[-4],[-8],[-8]])
        self.assertListEqual(np.around(res_element["q_y"],10).tolist(), [[-20],[-28],[-28],[-20]])

    def test_heat2d4n_element_skewed(self):
        # Pre-processing
        element1 = Heat2D4N(0, [[2,7], [1,3], [8,1], [6,8]], {"lambda_x":6, "lambda_y":6})

        # Assemble system
        K_element = element1.build_element_stiffness_matrix()
        f_element = element1.build_element_load_vector([0], [4])

        # Check system matrix and load vector
        self.assertListEqual(np.around(K_element,8).tolist(), [[4.87272727, -1.58181818, -1.70909091, -1.58181818],[-1.58181818, 4.65454545, -0.52727273, -2.54545455],[-1.70909091, -0.52727273, 2.76363636, -0.52727273],[-1.58181818, -2.54545455, -0.52727273, 4.65454545]])
        self.assertListEqual(np.around(f_element,10).tolist(), [[25], [30], [35], [30]])

        # Calculate and check resultants (nodal unknowns do not represent reality, purely exemplary)
        res_element = element1.calculate_element_resultants(np.array([[1.5e17],[1.5e17],[1.5e17],[1.5e17]]))
        self.assertListEqual(np.around(res_element["q_x"],10).tolist(), [[0],[32],[-16],[-32]])
        self.assertListEqual(np.around(res_element["q_y"],10).tolist(), [[0],[32],[0],[-32]])


if __name__ == '__main__':
    unittest.main()
