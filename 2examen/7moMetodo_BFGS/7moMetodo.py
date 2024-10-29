#minimizarimport numpy as np
import numpy as np
# Función a minimizar 
def func(x):
    return (x[0] + 2)**2 + x[1]**2 +2
# Gradiente de la función
def grad(x):
    return np.array([2*x[0] +4, # derivada respecto x[0]
                     2*x[1]])    # derivada respecto x[1]

# Algoritmo BFGS
def bfgs(func, grad, x0, tol=1e-5, max_iter=1000):
    n = len(x0)
    I = np.eye(n)  # Matriz identidad de tamaño n x n
    H = I          # Aproximación inicial del Hessiano inverso es la identidad
    x = x0         # Punto inicial
    for i in range(max_iter):
        g = grad(x)  # Calculamos el gradiente en el punto actual
        # Condición de parada: 
        # si la norma del gradiente es menor que la tolerancia, se ha alcanzado un óptimo
        if np.linalg.norm(g) < tol:
            print(f'Convergió en {i} iteraciones')
            return x

        # Dirección de descenso: p = -H * g (usamos la aproximación del Hessiano inverso)
        p = -H @ g
        # Búsqueda de línea para encontrar el mejor paso en la dirección p
        alpha = busqueda_linea(func, grad, x, p)  
        # Actualizamos la posición: x_new = x + alpha * p
        x_new = x + alpha * p
        # Calculamos el nuevo gradiente en la nueva posición
        g_new = grad(x_new)
        # Vector de desplazamiento s y diferencia de gradientes y
        s = x_new - x
        y = g_new - g
        # Calculamos rho, que es el inverso del producto interno de y y s
        rho = 1.0 / (y @ s)
        # Si rho es positivo, actualizamos la matriz H (Hessiano inverso) con la fórmula BFGS
        if rho > 0:  # Evitamos divisiones por cero
            H = (I - rho * np.outer(s, y)) @ H @ (I - rho * np.outer(y, s)) + rho * np.outer(s, s)
        # Actualizamos el punto actual
        x = x_new
    # Si no se alcanza la convergencia después de max_iter iteraciones
    print('No convergió')
    return x



# Búsqueda de línea simple 
# Esta función implementa una búsqueda de línea con retroceso para encontrar el mejor valor de alpha.
# Básicamente, va reduciendo alpha en cada iteración hasta que se cumpla la condición de Armijo (suficiente disminución).
def busqueda_linea(func, grad, x, p, alpha=1, rho=0.8, c=1e-4):
    # Condición de Armijo: buscamos que la nueva posición tenga un valor de función menor que un valor umbral.
    while func(x + alpha * p) > func(x) + c * alpha * np.dot(grad(x), p):
        # Si no cumple la condición, reducimos el paso multiplicando alpha por un factor rho (menos de 1)
        alpha *= rho
    return alpha


# Parámetros iniciales
x0 = np.array([0, 0])

# Llamada al algoritmo BFGS
resultado = bfgs(func, grad, x0)
print(f"Resultado óptimo: {resultado}")
print(f"Valor de la función en el óptimo: {func(resultado)}")
