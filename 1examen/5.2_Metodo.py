## CODIGO SIN PRECONDICIONADOR

import numpy as np

# Definición de la matriz A y el vector b
A = np.array([[9, 1, 2], [1, 7, 1], [2, 1, 6]])
b = np.array([12, 13, 14])

# Método de Richardson sin precondicionador con factor de relajación
def richardson(A, b, x0, omega=0.1, tol=1e-6, max_iter=1000):
    x = x0
    for i in range(max_iter):
        r = b - A @ x  # Calcula el residuo
        if np.linalg.norm(r) < tol:  # Condición de convergencia
            return x, i + 1  # Retorna la solución y el número de iteraciones
        x = x + omega * r  # Actualiza x usando el método de Richardson
    return x, max_iter

# Solución inicial
x0 = np.zeros_like(b)

# Resolver el sistema
x, iteraciones = richardson(A, b, x0)
print(f"Solución sin precondicional: {x}, Iteraciones: {iteraciones}")