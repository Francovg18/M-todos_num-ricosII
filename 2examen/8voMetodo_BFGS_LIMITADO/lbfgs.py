from scipy.optimize import minimize
import numpy as np

# Definir la función objetivo
def objective(x):
    return np.sin(x[0]) + np.cos(x[1]) + np.tan(x[2]) + (x[0] - 2)**2 + (x[1] + 1)**2 + (x[2] - 1)**2

# Definir la derivada de la función objetivo
def derivative(x):
    dfdx0 = 2*x[0] + np.cos(x[0]) -4
    dfdx1 = 2*x[1] - np.sin(x[1]) +2
    dfdx2 = 2*x[2] + np.tan(x[2])**2 -1
    return np.array([dfdx0, dfdx1, dfdx2])

# Punto inicial
pt = np.array([0, 0,0])

# Optimización utilizando el método BFGS
result = minimize(objective, pt, method='BFGS', jac=derivative)

print("Método BFGS:")

# Evaluar la solución
solution = result['x']
evaluation = objective(solution)
print('Total Evaluations: %d' % result['nfev'])
print('Solución: f(%s) = %f' % (solution, evaluation))

# Optimización utilizando el método L-BFGS-B
result = minimize(objective, pt, method='L-BFGS-B', jac=derivative)

print("Método L-BFGS-B:")

# Evaluar la solución
solution = result['x']
evaluation = objective(solution)
print('Total Evaluations: %d' % result['nfev'])
print('Solución: f(%s) = %f' % (solution, evaluation))