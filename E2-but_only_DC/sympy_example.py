from sympy import Symbol, simplify, latex


x = Symbol('x', positive=True)
y = Symbol('y', positive=True)
A = Symbol('A', positive=True)

fct_1 = 1.0 / (5.0 * x)
fct_2 = 9.0 * x**2
fct_3 = fct_1 * (1 - y) - fct_2 * A

fct_3 = simplify(fct_3)

print(fct_3)
print(latex(fct_3))


