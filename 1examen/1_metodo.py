import numpy as np


def Funcion(x, y):
    return np.array([[x**2 -y],
    				 [x + y**2 -3]])
def Diferencia(ant, act):
    suma_diferencias = 0
    for i in range(len(ant)):
        suma_diferencias += abs(act[i][0] - ant[i][0])
    return suma_diferencias
    
x, y = 0.5,0.5
f1 = x**2 -y
f2 = x + y**2 -3
F = np.array([[f1], [f2]])
X = np.array([[x], [y]])

	# PRIMERA ITERACION

	# JACOBIANO
df1x = 2*x
df1y = -1
df2x = 1
df2y = 2*y

JACOBIANO = np.array([[df1x, df1y],
                      [df2x, df2y]])

"""print("JACOBIANO inicial:")
print(JACOBIANO)"""

	# INVERSA DEL JACOBIANO
IJACOBIANO = np.linalg.inv(JACOBIANO)

resultado = X - (IJACOBIANO @ F)
tol = 1e-6


for k in range(25):
    X1 = resultado
    
    print(f"Iteración {k+1}: x = {X1[0][0]:.6f}, y = {X1[1][0]:.6f}")
    
    if Diferencia(X, X1) < tol:
        print("Convergencia lograda.")
        break

    F1 = Funcion(X1[0][0], X1[1][0])
    S = X1 - X
    Y = F1 - F
    
    A = IJACOBIANO + (((S - (IJACOBIANO @ Y)) @ (S.T @ IJACOBIANO)) / (S.T @ IJACOBIANO @ Y))
    
    resultado = X1 - (A @ F1)
    
    IJACOBIANO = A
    X = X1
    F = F1

print("\nSolución final aproximada:")
print(f"x = {X1[0][0]:.6f}, y = {X1[1][0]:.6f}")

