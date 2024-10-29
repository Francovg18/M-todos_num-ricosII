import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tabulate import tabulate

# Definir la función objetivo
def funcion_objetivo(x):
    return np.sin(x[0]) + np.cos(x[1]) + np.tan(x[2]) + (x[0] - 2)**2 + (x[1] + 1)**2 + (x[2] - 1)**2

# Calcular el gradiente de la función objetivo
def gradiente(x):
    dfdx0 = 2*x[0] + np.cos(x[0]) -4
    dfdx1 = 2*x[1] - np.sin(x[1]) +2
    dfdx2 = 2*x[2] + np.tan(x[2])**2 -1
    return np.array([dfdx0, dfdx1, dfdx2])

# Realizar la búsqueda de línea
def busqueda_linea(f, x, p, nabla):
    a = 1
    c1 = 1e-10
    c2 = 0.9
    fx = f(x)
    x_nuevo = x + a * p
    nabla_nuevo = gradiente( x_nuevo)
    while f(x_nuevo) >= fx + (c1*a*nabla.T @ p) or nabla_nuevo.T @ p <= c2*nabla.T @ p:
        a = a * 0.9
        x_nuevo = x + a * p
        nabla_nuevo = gradiente( x_nuevo)
    return a

# Realizar la recursión de dos ciclos
def recursion_dos_ciclos(gradient, s_stored, y_stored, m):
    q = gradient
    longitud = len(q)
    a = np.zeros(m)
    rou = np.array([1/np.dot(y_stored[j, :], s_stored[j, :]) for j in range(m)])
    for i in range(m):
        a[m - 1 - i] = rou[m - 1 - i] * np.dot(s_stored[m - 1 - i, :], q)
        q = q - a[m - 1 - i]*y_stored[m - 1 - i, :]

    H_k0 = (np.dot(s_stored[m - 1], y_stored[m - 1])/np.dot(y_stored[m - 1], y_stored[m - 1]))
    r = H_k0 * q

    for i in range(m):
        beta = rou[i] * np.dot(y_stored[i, :], r)
        r = r + (a[i] - beta) * s_stored[i]
    return r

# Implementar el algoritmo L-BFGS
def L_bfgs(f, x0, max_it, m):
    d = len(x0)
    nabla = gradiente( x0)
    x = x0[:]
    x_store = np.array([x0])
    y_stored = []
    s_stored = []
    p = -nabla
    alpha = busqueda_linea(f, x, p, nabla)
    s_stored.append(alpha * p)
    grad_old = nabla[:]
    x = x + alpha * p
    nabla = gradiente( x)
    y_stored.append(nabla - grad_old)
    m_ = 1
    it = 1
    x_store = np.append(x_store, [x], axis=0)
    datos_iteraciones = [(it, x, f(x))]  # Datos para almacenar las iteraciones

    while np.linalg.norm(nabla) > 1e-5:
        if it > max_it:
            print('¡Se alcanzó el número máximo de iteraciones!')
            break

        if 0 < it and it < m:
            p = -recursion_dos_ciclos(nabla, np.array(s_stored), np.array(y_stored), m_)
            alpha = busqueda_linea(f, x, p, nabla)
            s_stored.append(alpha * p)
            grad_old = nabla[:]
            x = x + alpha * p
            nabla = gradiente(x)
            y_stored.append(nabla - grad_old)
            m_ = m_ + 1
            it = it + 1
            x_store = np.append(x_store, [x], axis=0)
            datos_iteraciones.append((it, x, f(x)))
        else:
            p = -recursion_dos_ciclos(nabla, np.array(s_stored), np.array(y_stored), m)
            alpha = busqueda_linea(f, x, p, nabla)
            s_stored.append(alpha * p)
            s_stored.pop(0)
            grad_old = nabla[:]
            x = x + alpha * p
            nabla = gradiente( x)
            y_stored.append(nabla - grad_old)
            y_stored.pop(0)
            it = it + 1
            x_store = np.append(x_store, [x], axis=0)
            datos_iteraciones.append((it, x, f(x)))

    return x, x_store, datos_iteraciones

x_optimo, xstore, datos_iteraciones = L_bfgs(funcion_objetivo, [0, 0, 0], 100, 15)
print('Valor óptimo:', x_optimo)

# Mostrar las iteraciones en una tabla
headers_tabla = ['Iteración', 'x', 'f(x)']
# Convertir arrays de NumPy a listas antes de pasarlos a tabulate
datos_tabla = [[iteracion[0], iteracion[1].tolist(), iteracion[2]] for iteracion in datos_iteraciones]

# Mostrar la tabla con los datos de las iteraciones
tabla = tabulate(datos_tabla, headers=headers_tabla, tablefmt='grid')
print(tabla)

x_optimo, _, datos_iteraciones = L_bfgs(funcion_objetivo, [0, 0, 0], 100, 15)
ultima_iteracion = datos_iteraciones[-1]
num_iteraciones = ultima_iteracion[0]
punto_optimo = ultima_iteracion[1]
evaluacion_optima = ultima_iteracion[2]

print(f"Número de iteraciones al final: {num_iteraciones}")
print(f"Punto óptimo evaluado en f(x): {punto_optimo} | f(x) = {evaluacion_optima}")