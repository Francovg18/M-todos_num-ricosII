import numpy as np
""" 2do metodo """
def affine_scaling_lp_fast(c, A, b, tol=1e-6, max_iter=100):
    """
    Método de escalamiento afín optimizado para resolver problemas de programación lineal.
    c: Coeficientes de la función objetivo
    A: Matriz de restricciones
    b: Vector de restricciones
    tol: Tolerancia para la convergencia
    max_iter: Máximo número de iteraciones
    """
    
    # Dimensiones del problema
    n = len(c)
    
    # Inicialización factible (solución inicial)
    x = np.ones(n)
    
    iter_count = 0
    while iter_count < max_iter:
        # Verificar si el sistema ha convergido
        if np.linalg.norm(np.dot(A, x) - b) < tol:
            print(f'Converged in {iter_count} iterations.')
            return x

        # Calcular gradiente de la función objetivo
        grad = c
        
        # Matriz diagonal de x para la dirección de Newton
        X = np.diag(x)
        
        # Calcular dirección de escalamiento afín
        A_inv = np.dot(A, X)
        delta_x_affine = np.dot(np.linalg.pinv(A_inv), (b - np.dot(A, x)))
        
        # Encontrar el tamaño de paso máximo para mantener x >= 0
        alpha = 1
        for i in range(n):
            if delta_x_affine[i] < 0:
                alpha = min(alpha, -x[i] / delta_x_affine[i])
        
        # Actualizar x con el tamaño de paso calculado
        x = x + alpha * delta_x_affine
        
        # Chequear convergencia
        if np.linalg.norm(delta_x_affine) < tol:
            print(f'Converged in {iter_count} iterations.')
            return x
        
        iter_count += 1
    
    print('Reached maximum iterations without convergence.')
    return x

# Ejemplo de uso
c = np.array([4, 3,2])  # Coeficientes de la función objetivo
A = np.array([[1, 2,1], [2, 1,3],[1,1,0]])  # Matriz de restricciones
b = np.array([8, 12,5])  # Vector de restricciones

# Llamada a la función
solucion = affine_scaling_lp_fast(c, A, b)
print("Solución óptima:", solucion)