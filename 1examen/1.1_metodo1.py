import numpy as np

def broyden_method(F, x0, tol=1e-8, max_iter=100):
    """
    """
    x = np.array(x0, dtype=float)
    n = len(x0)
    
    # Función que evalúa el sistema de ecuaciones en el punto x
    def F_evaluate(x):
        return np.array([F_i(x) for F_i in F])
    
    # Inicializa la matriz de Broyden
    B = np.eye(n)
    
    for i in range(max_iter):
        # Evaluar el sistema de ecuaciones
        Fx = F_evaluate(x)
        
        # Actualiza la solución usando la matriz de Broyden
        delta_x = np.linalg.solve(B, -Fx)
        x_new = x + delta_x
        
        # Evaluar el sistema en el nuevo punto
        Fx_new = F_evaluate(x_new)
        
        # Calcula el cambio en el sistema de ecuaciones y en la solución
        delta_Fx = Fx_new - Fx
        delta_x = x_new - x
        
        if np.linalg.norm(delta_x) < tol:
            return x_new, i + 1
        
        # Actualiza la matriz de Broyden
        y = delta_Fx - B @ delta_x
        B += np.outer(y, delta_x) / (delta_x @ delta_x)
        
        x = x_new
    
    raise ValueError("El método de Broyden no convergió después de {} iteraciones.".format(max_iter))

# Definir el sistema de ecuaciones
def F1(x):
    return x[0]**2 -10*x[0] +x[1]**2 +8

def F2(x):
    return x[0]*x[1]**2 +x[0] -10*x[1]+8

# Sistema de ecuaciones
F = [F1, F2]

# Aproximación inicial
x0 = [1.5, 1.5]

# Llamar al método de Broyden
solution, iterations = broyden_method(F, x0)

print("Solución:", solution)
print("Número de iteraciones:", iterations)
