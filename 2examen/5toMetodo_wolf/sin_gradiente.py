# Definimos la función
def f(x):
    """
    Esta función retorna el valor de (x - 2)^2 + 1.
    El mínimo de esta función se encuentra en x = 2.
    """
    return (x - 3) ** 2 

# Definimos la derivada de la función
def df(x):
    """
    Esta función retorna la derivada de f(x), que es 2 * (x - 2).
    Se utiliza para calcular la dirección del descenso.
    """
    return 2 * (x - 3)

# Método del Gradiente Descendente Sin Condición de Wolfe
def gradient_descent_simple(starting_point, alpha=0.1, max_iter=100):
    """
    Realiza el método de gradiente descendente sin condiciones de Wolfe.

    - starting_point: punto de inicio para la búsqueda
    - alpha: tasa de aprendizaje
    - max_iter: número máximo de iteraciones
    """
    x = starting_point  # Inicialización del punto
    iter_count = 0      # Contador de iteraciones

    while iter_count < max_iter:
        grad = df(x)  # Calcula el gradiente en el punto actual
        # Actualización del punto en la dirección del descenso
        x -= alpha * grad
        iter_count += 1  # Incrementar el contador de iteraciones

        # Criterio de convergencia
        if abs(grad) < 1e-5:
            break

    return x, iter_count  # Retorna el punto mínimo y el número de iteraciones

# Ejecución del algoritmo
resultado_simple, iteraciones_simple = gradient_descent_simple(starting_point=0.0)
print(f"Resultado (Simple): {resultado_simple}, Iteraciones: {iteraciones_simple}")
