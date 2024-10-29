import numpy as np

def conjugate_gradient(A, b, x0, tol=1e-6):
    x = x0.copy()
    r = b - np.dot(A, x)
    p = r.copy()
    iteration_count = 0  # Inicializar el contador de iteraciones

    while np.linalg.norm(r) > tol:
        alpha = np.dot(r, r) / np.dot(p, np.dot(A, p))
        x = x + alpha * p
        r_new = r - alpha * np.dot(A, p)
        beta = np.dot(r_new, r_new) / np.dot(r, r)
        p = r_new + beta * p
        r = r_new.copy()
        iteration_count += 1  # Incrementar el contador de iteraciones

    print("Número de iteraciones:", iteration_count)
    return x



A = np.array([[4, -1, 1], [-1, 5, -1], [1, -1, 3]])
b = np.array([7, -4, 2])
x0 = np.array([0, 0, 0])

# Aplicar el método del gradiente conjugado
solution = conjugate_gradient(A, b, x0)

print("Solución:", solution)
