from sympy import Symbol, latex, Matrix, simplify

xi = Symbol('xi', positive=True)
eta = Symbol('eta', positive=True)
lambda_x = Symbol('lambda_x', positive=True)
lambda_y = Symbol('lambda_y', positive=True)


# d_phi_A_xi = Symbol(r'{\phi_{A,\xi}}', positive=True)
# d_phi_B_xi = Symbol(r'{\phi_{B,\xi}}', positive=True)
# d_phi_C_xi = Symbol(r'{\phi_{C,\xi}}', positive=True)
# d_phi_D_xi = Symbol(r'{\phi_{D,\xi}}', positive=True)
# d_phi_A_eta = Symbol(r'{\phi_{A,\eta}}', positive=True)
# d_phi_B_eta = Symbol(r'{\phi_{B,\eta}}', positive=True)
# d_phi_C_eta = Symbol(r'{\phi_{C,\eta}}', positive=True)
# d_phi_D_eta = Symbol(r'{\phi_{D,\eta}}', positive=True)

d_phi_A_xi = (1 - eta)
d_phi_B_xi = (1 - eta)
d_phi_C_xi = (1 + eta)
d_phi_D_xi = (1 + eta)
d_phi_A_eta = (1 - xi)
d_phi_B_eta = (1 + xi)
d_phi_C_eta = (1 + xi)
d_phi_D_eta = (1 - xi)

J_1 = (2 / xi) * Matrix([[0.25 * xi, 0], [-0.5, 2]])
det_J =(1/2)*xi
d_omega = 0.25*Matrix([[d_phi_A_xi, d_phi_B_xi, d_phi_C_xi, d_phi_D_xi], [d_phi_A_eta, d_phi_B_eta, d_phi_C_eta, d_phi_D_eta]])
E = Matrix([[lambda_x, 0], [0, lambda_y]])

J_1_T_s = simplify(J_1.T)
J_1_s = simplify(J_1)

d_omega_T_s =simplify(d_omega.T)
d_omega_s =simplify(d_omega)
E_s = simplify(E)

result = (simplify(d_omega_T_s*J_1_T_s*E_s*J_1_s*d_omega_s*det_J))

element = result[1, 3]
latex_output = latex(element)

print(latex_output)

