import numpy as np

def karmarkar_verbose(c, A, B,alpha=0.0005, epsilon=1e-10):
    n = len(c)
    k = 1
    x_k = np.ones(n) / n  # Inicialización de x^1
    print( f"Valor X_0 = {x_k}")


    while True:
        print(f"Iteración {k}:")


        # Transformación Proyectiva
        x_bar = np.diag(x_k)

        A_bar = A @ x_bar

        c_bar = x_bar @ c

        # Matriz de restricciones activas
        B = np.vstack((A_bar, np.ones(n)))


        # Calculate B.T @ B
        BTB = np.dot(B,B.T)

        # Proyección del Gradiente
        I = np.identity(B.shape[1])
        projection_matrix = I - (B.T @ np.linalg.inv(BTB) @ B)
        d = -projection_matrix @ c_bar

        # Actualización del Punto
        x_hat = 1/n * np.ones(n) + alpha * d.flatten() / (n * np.linalg.norm(d))
        x_k = np.dot(x_bar, x_hat) / np.dot(np.ones(n), np.dot(x_bar, x_hat))

        print("Solucion de X", x_k)

        # Criterio de Detención
        if np.dot(c, x_k) < epsilon:
            break

        k += 1

    # Redondeo de la solución a una solución entera factible
    x_solution = np.round(x_k,8)

    return x_solution

# Ejemplo de uso
c = np.array([-2, 1, 1])
A = np.array([1,1,-2])
B = np.array([1,0,-1])
solution = karmarkar_verbose(c, A,B)
print("Solución final:")
print(solution)
z = np.dot(c, solution)

print("Solución en la funcion objetivo:")
print(z)