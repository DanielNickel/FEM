#----------ISD | TU Braunschweig----------#
#-----------Beethovenstraße 51------------#
#-----------38106 Braunschweig------------#
#-----------------------------------------#

import unittest
import numpy as np

import sys, os
path_file_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(path_file_dir, '..'))

from elements.plane_stress_2d4n import PlaneStress2D4N


class TestPlaneStressElements(unittest.TestCase):

    def test_plane_stress_2d4n_element_scaled(self):
        # Check variables of element type
        self.assertEqual(PlaneStress2D4N.n_dimensions, 2)
        self.assertEqual(PlaneStress2D4N.p_order_of_shape_functions, 1)
        self.assertEqual(PlaneStress2D4N.n_nodes_per_element, 4)
        self.assertEqual(PlaneStress2D4N.dof_names, ["u","v"])
        self.assertEqual(PlaneStress2D4N.p_order_of_resultants, [1, 1, 1])
        self.assertEqual(PlaneStress2D4N.material_param_names, ["e_modulus", "poisson_ratio", "thickness"])

        # Pre-processing
        element1 = PlaneStress2D4N(0, [(0,-1.5), (3,-1.5), (3,0), (0,0)], {"e_modulus":35000, "poisson_ratio":0.2, "thickness": 1.0})

        # Assemble system
        K_element = element1.build_element_stiffness_matrix()
        f_element = element1.build_element_load_vector([0, 1], [4, 8])

        # Check system matrix and load vector
        self.assertListEqual(np.around(K_element,8).tolist(), [[15798.61111111, 5468.75, -1215.27777778, -1822.91666667, -7899.30555556, -5468.75, -6684.02777778, 1822.91666667],
                                                               [5468.75, 26736.11111111, 1822.91666667, 9722.22222222, -5468.75, -13368.05555556, -1822.91666667, -23090.27777778],
                                                               [-1215.27777778, 1822.91666667, 15798.61111111, -5468.75, -6684.02777778, -1822.91666667, -7899.30555556, 5468.75],
                                                               [-1822.91666667, 9722.22222222, -5468.75, 26736.11111111, 1822.91666667, -23090.27777778, 5468.75, -13368.05555556],
                                                               [-7899.30555556, -5468.75, -6684.02777778, 1822.91666667, 15798.61111111, 5468.75, -1215.27777778, -1822.91666667],
                                                               [-5468.75, -13368.05555556, -1822.91666667, -23090.27777778, 5468.75, 26736.11111111, 1822.91666667, 9722.22222222],
                                                               [-6684.02777778, -1822.91666667, -7899.30555556, 5468.75, -1215.27777778, 1822.91666667, 15798.61111111, -5468.75],
                                                               [1822.91666667, -23090.27777778, 5468.75, -13368.05555556, -1822.91666667, 9722.22222222, -5468.75, 26736.11111111]])
        self.assertListEqual(np.around(f_element,10).tolist(), [[4.5], [9], [4.5], [9], [4.5], [9], [4.5], [9]])

        # Calculate and check resultants
        res_element = element1.calculate_element_resultants(np.array([[1],[3],[10],[6],[1],[3],[10],[6]]))
        self.assertListEqual(np.around(res_element["sigma_xx"],8).tolist(), [[123958.33333333],[94791.66666667],[-123958.33333333],[-94791.66666667]])
        self.assertListEqual(np.around(res_element["sigma_yy"],8).tolist(), [[94791.66666667],[-51041.66666667],[-94791.66666667],[51041.66666667]])
        self.assertListEqual(np.around(res_element["sigma_xy"],8).tolist(), [[102083.33333333],[-72916.66666667],[-102083.33333333],[72916.66666667]])

    def test_plane_stress_2d4n_element_skewed(self):
        # Pre-processing
        element1 = PlaneStress2D4N(0, [[1,3], [8,1], [6,8], [2,7]], {"e_modulus":1e5, "poisson_ratio":0.3, "thickness":0.1})

        # Assemble system
        K_element = element1.build_element_stiffness_matrix()
        f_element = element1.build_element_load_vector([0, 1], [4, 8])

        # Check system matrix and load vector
        self.assertListEqual(np.around(K_element,8).tolist(), [[ 5754.24575425,  2229.43722944, -2437.56243756,  -180.65268065, -3146.85314685, -2056.27705628,  -169.83016983,     7.49250749],
                                                               [ 2229.43722944,  5754.24575425,    94.07259407,  1133.86613387, -2056.27705628, -3146.85314685,  -267.23276723, -3741.25874126],
                                                               [-2437.56243756,    94.07259407,  3416.58341658, -1466.45021645,  1133.86613387,  -180.65268065, -2112.88711289,  1553.03030303],
                                                               [ -180.65268065,  1133.86613387, -1466.45021645,  3416.58341658,    94.07259407, -2437.56243756,  1553.03030303, -2112.88711289],
                                                               [-3146.85314685, -2056.27705628,  1133.86613387,    94.07259407,  5754.24575425,  2229.43722944, -3741.25874126,  -267.23276723],
                                                               [-2056.27705628, -3146.85314685,  -180.65268065, -2437.56243756,  2229.43722944,  5754.24575425,     7.49250749,  -169.83016983],
                                                               [ -169.83016983,  -267.23276723, -2112.88711289,  1553.03030303, -3741.25874126,     7.49250749,  6023.97602398, -1293.29004329],
                                                               [    7.49250749, -3741.25874126,  1553.03030303, -2112.88711289,  -267.23276723,  -169.83016983, -1293.29004329,  6023.97602398]])

        self.assertListEqual(np.around(f_element,10).tolist(), [[30.], [60.], [35.], [70.], [30.], [60.], [25.], [50.]])

        # Calculate and check resultants
        res_element = element1.calculate_element_resultants(np.array([[0],[0],[0],[0],[0.02],[0.01],[0.05],[0.03]]))
        self.assertListEqual(np.around(res_element["sigma_xx"],8).tolist(), [[ 597.06959707],[ 148.96214896],[-842.49084249],[-937.72893773]])
        self.assertListEqual(np.around(res_element["sigma_yy"],8).tolist(), [[ 879.12087912],[ 200.24420024],[-252.74725275],[ 652.01465201]])
        self.assertListEqual(np.around(res_element["sigma_xy"],8).tolist(), [[ 525.64102564],[ 136.75213675],[-166.66666667],[ 307.69230769]])


if __name__ == '__main__':
    unittest.main()
