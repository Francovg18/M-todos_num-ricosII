import numpy as np
def gram_schmidt(A):
    """Realiza la factorización QR de la matriz A utilizando el proceso de Gram-Schmidt."""
    # Obtener las dimensiones de la matriz A
    m, n = A.shape

    # Inicializar matrices Q y R
    Q = np.zeros((m, n))
    R = np.zeros((n, n))

    for j in range(n):
        # Copiar la j-ésima columna de A en v
        v = A[:, j]

        for i in range(j):
            # Calcular el coeficiente R[i, j]
            R[i, j] = np.dot(Q[:, i], A[:, j])
            # Restar la proyección de A[:, j] sobre Q[:, i]
            v = v - R[i, j] * Q[:, i]

        # Calcular el valor de R[j, j] como la norma de v
        R[j, j] = np.linalg.norm(v)

        # Normalizar la columna v para obtener Q[:, j]
        Q[:, j] = v / R[j, j]

    return Q, R

def mostrar_resultados(Q,R):
  # Imprimir las matrices Q y R
  print("Matriz Q (Ortogonal):")
  print(Q)
  print("\nMatriz R (Triangular Superior):")
  print(R)

  # Verificar que A = Q * R
  A_reconstructed = np.dot(Q, R)
  print("\nReconstrucción de la Matriz A (Q * R):")
  print(A_reconstructed)

  # Definir una matriz A
A = np.array([[1, 1, 0],
              [1, 0, 1],
              [0, 1, 1]], dtype=float)

# Realizar la factorización QR usando Gram-Schmidt
Q, R = gram_schmidt(A)

# Imprimir las matrices Q y R
mostrar_resultados(Q,R)