# Definimos la función
def f(x):
    """
    Esta función retorna el valor de (x - 2)^2 + 1.
    El mínimo de esta función se encuentra en x = 2.
    """
    return (x - 3) ** 2 + 1

# Definimos la derivada de la función
def df(x):
    """
    Esta función retorna la derivada de f(x), que es 2 * (x - 2).
    Se utiliza para calcular la dirección del descenso.
    """

    return 2 * (x - 3)

# Parámetros de la condición de Wolfe
c1 = 1e-4  # Constante de la primera condición ( 0,0001) // descenso 
c2 = 0.9   # Constante de la segunda condición // punto no alejado 90% 

# Función para comprobar la condición de Wolfe
def wolfe_conditions(x, d, alpha, grad):
    """
    Comprueba si se satisfacen las condiciones de Wolfe para el paso alpha.

    - x: valor actual
    - d: dirección de descenso
    - alpha: tamaño del paso
    - grad: gradiente en el punto x
    """
    # Primera condición de Wolfe
    cond1 = f(x + alpha * d) <= f(x) + c1 * alpha * grad
    # Segunda condición de Wolfe
    cond2 = df(x + alpha * d) >= c2 * grad
    return cond1 and cond2

# Método del Gradiente Descendente
def gradient_descent_wolfe(starting_point, alpha=0.1, max_iter=100):
    """
    Realiza el método de gradiente descendente utilizando las condiciones de Wolfe.

    - starting_point: punto de inicio para la búsqueda
    - alpha: tasa de aprendizaje inicial
    - max_iter: número máximo de iteraciones
    """
    x = starting_point  # Inicialización del punto
    iter_count = 0      # Contador de iteraciones

    while iter_count < max_iter:
        grad = df(x)  # Calcula el gradiente en el punto actual
        d = -grad     # Dirección de descenso (negativa del gradiente)

        # Búsqueda del tamaño del paso utilizando condiciones de Wolfe
        alpha = 1.0  # Valor inicial del paso
        while not wolfe_conditions(x, d, alpha, grad):
            alpha *= 0.5  # Reducir el paso si no se cumplen las condiciones

        # Actualización del punto
        x += alpha * d
        iter_count += 1  # Incrementar el contador de iteraciones

        # Criterio de convergencia
        if abs(grad) < 1e-5:
            break

    return x, iter_count  # Retorna el punto mínimo y el número de iteraciones

# Ejecución del algoritmo
resultado_wolfe, iteraciones_wolfe = gradient_descent_wolfe(starting_point=0.0)
print(f"Resultado (Wolfe): {resultado_wolfe}, Iteraciones: {iteraciones_wolfe}")
