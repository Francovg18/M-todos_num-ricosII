import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def imprimir_tabla(tabla, vars_basicas, num_vars):
    columnas = [f'x{i+1}' for i in range(num_vars)] + \
               [f's{i+1}' for i in range(len(vars_basicas))] + ['RHS']
    df = pd.DataFrame(tabla, columns=columnas)
    df.index = ['Restricción ' + str(i+1) for i in range(len(vars_basicas))] + ['Z']
    print(df)

def graficar_restricciones(coef_restricciones, terminos_independientes):
    x_vals = np.linspace(0, 10, 400)
    
    for i, (coef, t_indep) in enumerate(zip(coef_restricciones, terminos_independientes)):
        y_vals = (t_indep - coef[0] * x_vals) / coef[1]
        plt.plot(x_vals, y_vals, label=f'Restricción {i+1}')

    plt.xlim(0, 10)
    plt.ylim(0, 10)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.axhline(0, color='black',linewidth=1)
    plt.axvline(0, color='black',linewidth=1)
    plt.fill_between(x_vals, np.maximum(0, (terminos_independientes[0] - coef_restricciones[0][0]*x_vals)/coef_restricciones[0][1]), 
                     np.minimum((terminos_independientes[1] - coef_restricciones[1][0]*x_vals)/coef_restricciones[1][1], 10), 
                     where=(x_vals >= 0), color='gray', alpha=0.3)
    plt.legend()
    plt.grid(True)
    plt.title('Región factible y restricciones')
    plt.show()

def simplex(coef_objetivo, coef_restricciones, terminos_independientes, tipo):
    # Número de variables y restricciones
    num_vars = len(coef_objetivo)
    num_restricciones = len(terminos_independientes)

    # Graficar restricciones si son 2 variables
    if num_vars == 2:
        graficar_restricciones(coef_restricciones, terminos_independientes)

    # Construcción de la tabla inicial
    tabla = np.zeros((num_restricciones + 1, num_vars + num_restricciones + 1))

    # Llenado de la tabla
    tabla[:-1, :num_vars] = coef_restricciones
    tabla[:-1, num_vars:num_vars + num_restricciones] = np.identity(num_restricciones)
    tabla[:-1, -1] = terminos_independientes
    
    # Si es minimización, cambiamos los signos de los coeficientes de la función objetivo
    if tipo == 'min':
        tabla[-1, :num_vars] = coef_objetivo
    elif tipo == 'max':
        tabla[-1, :num_vars] = -coef_objetivo

    # Índices de variables básicas
    vars_basicas = list(range(num_vars, num_vars + num_restricciones))

    valor_optimo_anterior = 0
    iteracion = 0

    while True:
        print(f"\nIteración {iteracion}:")
        print("Tabla actual:")
        imprimir_tabla(tabla, vars_basicas, num_vars)

        # Identificar la columna pivote
        col_pivote = np.argmin(tabla[-1, :-1])
        if tabla[-1, col_pivote] >= 0:
            print("No hay coeficientes negativos en la fila Z. Óptimo alcanzado.")
            break  

        # Calcular razones para determinar la fila pivote
        razones = []
        for i in range(num_restricciones):
            elemento = tabla[i, col_pivote]
            if elemento > 0:
                razones.append(tabla[i, -1] / elemento)
            else:
                razones.append(np.inf)

        fila_pivote = np.argmin(razones)
        if razones[fila_pivote] == np.inf:
            raise Exception("El problema no tiene solución óptima finita.")

        print(f"Variable entrante: x{col_pivote + 1}")
        print(f"Variable saliente: x{vars_basicas[fila_pivote] + 1}")
        print(f"Elemento pivote: {tabla[fila_pivote, col_pivote]}")

        # Pivoteo
        elemento_pivote = tabla[fila_pivote, col_pivote]
        tabla[fila_pivote, :] /= elemento_pivote
        for i in range(num_restricciones + 1):
            if i != fila_pivote:
                tabla[i, :] -= tabla[i, col_pivote] * tabla[fila_pivote, :]

        # Actualizar variables básicas
        vars_basicas[fila_pivote] = col_pivote

        # Calcular el error relativo en Z
        valor_optimo_actual = tabla[-1, -1]
        if valor_optimo_actual != 0:
            error_relativo = (valor_optimo_actual - valor_optimo_anterior) / valor_optimo_actual
        else:
            error_relativo = 0
        print(f"Valor de Z en esta iteración: {valor_optimo_actual}")
        print(f"Error relativo en Z: {error_relativo}")

        # Actualizar el valor óptimo anterior
        valor_optimo_anterior = valor_optimo_actual

        iteracion += 1

    # Extraer la solución óptima
    solucion = np.zeros(num_vars)
    for i in range(num_restricciones):
        if vars_basicas[i] < num_vars:
            solucion[vars_basicas[i]] = tabla[i, -1]

    valor_optimo = tabla[-1, -1]
    return solucion, valor_optimo, tabla

# Ejemplo de uso
""" MaxZ = 1x1 + 2x2
1x1 + 3x2 <= 9
2x1 + 1x2 <= 8 
1(3) + 2(2)
"""
tipo_problema = 'max' 

# Coeficientes de la función objetivo
coef_objetivo = np.array([1, 2])

# Coeficientes de las restricciones
coef_restricciones = np.array([[1, 3],
                               [2, 1]])

# Términos independientes
terminos_independientes = np.array([9, 8])

# Ejecutar el método simplex
solucion, valor_optimo, tabla_final = simplex(coef_objetivo, coef_restricciones, terminos_independientes, tipo_problema)

print("Solución óptima:")
print(f"x1 = {solucion[0]}")
print(f"x2 = {solucion[1]}")
print(f"Valor óptimo de Z = {valor_optimo}")
