##CODIGO CON PRECONDICIONADOR

import numpy as np

# Definición de la matriz A y el vector b
A = np.array([[4,-1,1], [-1,5,-1], [1,-1,3]])
b = np.array([7,-4,2])

# Precondicionador de Jacobi (inverso de la diagonal)
D_inv = np.diag(1 / np.diag(A))

# Método de Richardson con precondicionador
def richardson(A, b, x0, M, tol=1e-6, max_iter=1000):
    x = x0
    for i in range(max_iter):
        r = b - A @ x
        if np.linalg.norm(r) < tol:
            return x, i + 1  # Retorna la solución y el número de iteraciones
        x = x + M @ r
    return x, max_iter

# Solución inicial
x0 = np.zeros_like(b)

# Resolver el sistema
x, iteraciones = richardson(A, b, x0, D_inv)
print(f"Solución Precondicional: {x}, Iteraciones: {iteraciones}")