import numpy as np

def funcion_objetivo(x):
    return x[0]**2+x[1]**2 + 10


def gradiente(x):
    grad = np.zeros_like(x)
    grad[0] = 2*x[0]
    grad[1] = 2*x[1]
    return grad

# Búsqueda de línea: Método para encontrar el mejor tamaño de paso 
def busqueda_linea(f, xk, pk, gk, alpha=1, rho=0.9, c=1e-5):
    # Implementa la condición de Armijo para la búsqueda de línea
    while f(xk + alpha * pk) > f(xk) + c * alpha * np.dot(gk, pk):
        alpha *= rho
    return alpha

# Método BFGS para la minimización de una función
def bfgs(f, grad, x0, tol=1e-5, max_iter=1000):
    xk = x0
    n = len(x0)
    Hk = np.eye(n)  # Matriz identidad inicial (aproximación inicial de la Hessiana)
    
    print(f"Iteración {'k'.ljust(3)} | {'xk'.ljust(30)} | {'f(xk)'.ljust(20)} | {'||gk||'.ljust(10)}")
    print("-" * 80)
    
    for k in range(max_iter):
        gk = grad(xk)  # Calcula el gradiente en el punto actual
        fk = f(xk)  # Calcula el valor de la función objetivo en el punto actual

        # Imprime la iteración, el valor de xk, f(xk) y la norma del gradiente
        print(f"{k:<8} | {xk} | {fk:<20.5f} | {np.linalg.norm(gk):<10.5f}")
        
        # Condición de convergencia: Si el gradiente es cercano a 0, hemos convergido
        if np.linalg.norm(gk) < tol:
            print(f"Convergencia alcanzada en {k} iteraciones.")
            break
        
        # Dirección de búsqueda pk: -Hk * gradiente
        pk = -np.dot(Hk, gk)
        
        # Búsqueda de línea para encontrar el mejor tamaño de paso alpha
        alpha = busqueda_linea(f, xk, pk, gk)
        
        # Actualiza el punto actual
        xk1 = xk + alpha * pk
        
        # Calcula el nuevo gradiente
        gk1 = grad(xk1)
        
        # Variables de actualización para la matriz Hessiana
        sk = xk1 - xk  # Diferencia de posiciones
        yk = gk1 - gk  # Diferencia de gradientes
        
        # Verificación para evitar divisiones por cero
        if np.dot(yk, sk) > 1e-10:
            # Actualización de la matriz Hessiana con la fórmula de BFGS
            rho_k = 1.0 / np.dot(yk, sk)
            I = np.eye(n)
            Vk = I - rho_k * np.outer(sk, yk)
            Hk = np.dot(Vk, np.dot(Hk, Vk.T)) + rho_k * np.outer(sk, sk)
        
        # Actualización de la solución actual
        xk = xk1
    
    return xk

# Ejemplo de uso del método BFGS:
x0 = np.array([1, 1])  # Punto inicial (alejado del mínimo)
solucion = bfgs(funcion_objetivo, gradiente, x0)
print(f"Solución encontrada: {solucion}")