import numpy as np
from scipy.optimize import minimize

def function(x):
    return x[0]**3 + x[1]**2 + 2 * x[0] * x[1] - 4 * x[0] - 3 * x[1] + 5

def jacobian(x):
    dx = 3 * x[0]**2 + 2 * x[1] - 4
    dy = 2 * x[0] + 2 * x[1] - 3 
    return np.array([dx, dy])

def hessian(x):
    d2x = 6 * x[0]
    d2y = 2
    d2xy = 2
    return np.array([[d2x, d2xy],
                     [d2xy, d2y]])

# Posición inicial
x0 = np.array([0, 0])

# Minimizar la función usando Newton-CG
result_ncg = minimize(function, x0, method='Newton-CG', jac=jacobian)

print("----------\t\t Newton-CG \t\t----------")

# Evaluar la solución de Newton-CG
print("Iteraciones = ", result_ncg.nit)
print("f(x) = ", result_ncg.fun)
print("Solución = ", result_ncg.x)

# Minimizar la función usando Trust-Region Newton-CG
result_tncg = minimize(function, x0, method='trust-ncg', jac=jacobian, hess=hessian)

print("\n----------\t Trust-Region Newton-CG \t----------")

# Evaluar la solución de Trust-Region Newton-CG
print("Iteraciones = ", result_tncg.nit)
print("f(x) = ", result_tncg.fun)
print("Solución = ", result_ncg.x)  # Typo corrected: result_ncg.x -> result_tncg.x