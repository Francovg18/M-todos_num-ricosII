import numpy as np

# Parámetros del algoritmo
t = 1.0       # Parámetro de barrera inicial
mu = 10.0     # Factor de crecimiento para t
tol = 1e-6    # Tolerancia de convergencia
max_iter = 100  # Máximo número de iteraciones
beta = 0.5    # Factor de reducción en la búsqueda de línea
epsilon = 1e-6  # Evitar problemas de logaritmos en la región factible

# Función objetivo con barrera
def objective(x, t):
    x1, x2 = x
    if x1 <= epsilon or x2 <= epsilon or (x1 + x2) >= 2 - epsilon or x1 >= 1 - epsilon or x2 >= 1 - epsilon:
        return np.inf  # Penalización si viola las restricciones
    return -x1 - 2*x2 - (1 / t) * (np.log(2 - (x1 + x2)) + np.log(1 - x1) + np.log(1 - x2))

# Gradiente de la función objetivo
def gradient(x, t):
    x1, x2 = x
    if x1 <= epsilon or x2 <= epsilon or (x1 + x2) >= 2 - epsilon or x1 >= 1 - epsilon or x2 >= 1 - epsilon:
        return np.array([np.inf, np.inf])  # Gradiente indefinido fuera de la región factible
    grad_x1 = -1 + (1 / t) * (1 / (2 - (x1 + x2)) - 1 / (1 - x1))
    grad_x2 = -2 + (1 / t) * (1 / (2 - (x1 + x2)) - 1 / (1 - x2))
    return np.array([grad_x1, grad_x2])

# Hessiano de la función objetivo
def hessian(x, t):
    x1, x2 = x
    if x1 <= epsilon or x2 <= epsilon or (x1 + x2) >= 2 - epsilon or x1 >= 1 - epsilon or x2 >= 1 - epsilon:
        return np.array([[np.inf, np.inf], [np.inf, np.inf]])  # Hessiano indefinido
    h11 = (1 / t) * (1 / (2 - (x1 + x2))**2 + 1 / (1 - x1)**2)
    h12 = (1 / t) * (1 / (2 - (x1 + x2))**2)
    h21 = h12
    h22 = (1 / t) * (1 / (2 - (x1 + x2))**2 + 1 / (1 - x2)**2)
    return np.array([[h11, h12], [h21, h22]])

# Método de Newton para el paso de minimización
def newton_step(x, t):
    grad = gradient(x, t)
    hess = hessian(x, t)
    try:
        step = np.linalg.solve(hess, -grad)  # Resolver el sistema Hessiano * dx = -grad
    except np.linalg.LinAlgError:
        return np.zeros_like(x)  # Si la matriz Hessiana no es invertible
    return step

# Búsqueda de línea para asegurar que el nuevo punto esté en la región factible
def busqueda_linea(x, direction):
    step_size = 1.0
    while np.any(x + step_size * direction <= epsilon) or \
          np.sum(x + step_size * direction) >= 2 - epsilon or \
          (x[0] + step_size * direction[0]) >= 1 - epsilon or \
          (x[1] + step_size * direction[1]) >= 1 - epsilon:
        step_size *= beta
    return step_size

# Método de seguimiento de camino
def path_following_method(x0):
    global t
    x = x0
    for iteration in range(max_iter):
        step = newton_step(x, t)
        step_size = busqueda_linea(x, step)
        x_new = x + step_size * step

        # Verifica que no esté fuera de la región factible
        if np.any(x_new <= epsilon) or np.sum(x_new) >= 2 - epsilon or \
           x_new[0] >= 1 - epsilon or x_new[1] >= 1 - epsilon:
            print(f"Iteración {iteration + 1}: Se salió de la región factible.")
            break

        # Condición de convergencia
        if np.linalg.norm(step) < tol:
            print(f"Convergencia alcanzada en la iteración {iteration + 1}")
            break

        x = x_new
        t *= mu  # Aumenta el parámetro de barrera

    return x

# Punto inicial (dentro de la región factible)
x0 = np.array([0.5, 0.5])  # Cambiar este valor si es necesario

# Ejecutar el algoritmo
solucion = path_following_method(x0)
print(f"Solución óptima: x1 = {solucion[0]}, x2 = {solucion[1]}")
