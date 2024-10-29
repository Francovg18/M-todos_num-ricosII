import numpy as np

def Funcion(x):
    x1, x2, lambda1, lambda2 = x
    return np.array([
        [np.sin(x1) + np.cos(x2)],  
        [lambda1 * (x1**2 + x2**2 - 1)], 
        [lambda2 * (x1 - 0.5)]  
    ])

# Función para calcular las diferencias entre dos iteraciones
def Diferencia(ant, act):
    suma_diferencias = 0
    for i in range(len(ant)):
        suma_diferencias += abs(act[i][0] - ant[i][0])
    return suma_diferencias

# Inicialización de las variables
x0, y0, l1, l2 = 0.6, 0.6, 0.1, 0.1  # Valores iniciales
X = np.array([[x0], [y0], [l1], [l2]])

# Evaluar la función en el punto inicial
F = Funcion([x0, y0, l1, l2])

# Calcular el Jacobiano inicial
df1x = np.cos(x0) 
df1y = -np.sin(y0)  
df1l1 = 0
df1l2 = 0  

df2x = 2 * x0  
df2y = 2 * y0 
df2l1 = x0**2 + y0**2 - 1  
df2l2 = 0  

df3x = 1 
df3y = 0 
df3l1 = 0  
df3l2 = x0 - 0.5  

# Construcción del Jacobiano inicial
JACOBIANO = np.array([
    [df1x, df1y, df1l1, df1l2],
    [df2x, df2y, df2l1, df2l2],
    [df3x, df3y, df3l1, df3l2]
])

# Inversa del Jacobiano usando el pseudoinverso
IJACOBIANO = np.linalg.pinv(JACOBIANO)

# Primera aproximación de la solución
resultado = X - (IJACOBIANO @ F)
tol = 1e-6

# Bucle de iteraciones
for k in range(25):
    X1 = resultado

    print(f"Iteración {k+1}: x1 = {X1[0][0]:.6f}, x2 = {X1[1][0]:.6f}, λ1 = {X1[2][0]:.6f}, λ2 = {X1[3][0]:.6f}")

    if Diferencia(X, X1) < tol:
        print("Convergencia lograda.")
        break

    F1 = Funcion(X1.flatten())  # Evaluar la función en el nuevo punto
    S = X1 - X
    Y = F1 - F

    # Actualizar el Jacobiano inverso con la fórmula de Broyden
    A = IJACOBIANO + (((S - (IJACOBIANO @ Y)) @ (S.T @ IJACOBIANO)) / (S.T @ IJACOBIANO @ Y))

    resultado = X1 - (A @ F1)

    IJACOBIANO = A
    X = X1
    F = F1

# Mostrar la solución final aproximada
print("\nSolución final aproximada:")
print(f"x1 = {X1[0][0]:.6f}, x2 = {X1[1][0]:.6f}, λ1 = {X1[2][0]:.6f}, λ2 = {X1[3][0]:.6f}")
