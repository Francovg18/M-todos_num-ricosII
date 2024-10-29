import numpy as np

def newton_system(F, J, x0, tol=1e-6, max_iter=100):
    """
    Resuelve un sistema de ecuaciones no lineales usando el método de Newton-Raphson.
    Parámetros:
    F -- Función vectorial que representa el sistema de ecuaciones no lineales
    J -- Función que devuelve la matriz Jacobiana
    x0 -- Vector de aproximación inicial
    tol -- Tolerancia para la convergencia
    max_iter -- Número máximo de iteraciones
    
    """
    x = x0
    for i in range(max_iter):
        Fx = F(x)  # Evaluar el sistema de ecuaciones en x
        Jx = J(x)  # Evaluar la Jacobiana en x
        
        # Resolver el sistema lineal J(x) * delta_x = -F(x)
        try:
            delta_x = np.linalg.solve(Jx, -Fx)
        except np.linalg.LinAlgError:
            print('La matriz Jacobiana es singular y no se puede invertir.')
            return None
        
        x = x + delta_x  # Actualizar la solución
        
        # Verificar convergencia
        if np.linalg.norm(delta_x, ord=2) < tol:
            print(f'Convergió después de {i+1} iteraciones.')
            return x
    
    print('No convergió después del número máximo de iteraciones.')
    return x

# sistema de ecuaciones no lineales
def F(x):
    return np.array([
        x[0]**2 -x[1],  
        x[0]+x[1]**2 -3       
    ])

# Dmatriz Jacobiana
def J(x):
    return np.array([
        [2*x[0], -1],  
        [1, 2*x[1]]   
    ])

# Aproximación inicial
x0 = np.array([0.5, 0.5])

# Resolver el sistema
solucion = newton_system(F, J, x0)
if solucion is not None:
    print('Solución aproximada:', solucion)
