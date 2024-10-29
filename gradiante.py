import sympy as sp

# Definir las variables simbólicas para tres variables
x0, x1, x2 = sp.symbols('x0 x1 x2')

# Definir la función objetivo (puedes cambiarla a lo que desees)
# Ejemplo: f(x0, x1, x2) = sin(x0) + cos(x1) + tan(x2) + (x0 - 2)^2 + (x1 + 1)^2 + (x2 - 1)^2
funcion_objetivo = sp.sin(x0) + sp.cos(x1) + sp.tan(x2) + (x0 - 2)**2 + (x1 + 1)**2 + (x2 - 1)**2

# Calculamos el gradiente (las derivadas parciales con respecto a x0, x1, y x2)
gradiente = [sp.diff(funcion_objetivo, var) for var in (x0, x1, x2)]

# Mostramos el gradiente calculado
print("Gradiente:")
for i, g in enumerate(gradiente):
    print(f"∂f/∂x{i} = {g}")

# Si quieres evaluar el gradiente en un punto específico (por ejemplo, [0, 0, 0]):
punto = {x0: 0.0, x1: 0.0, x2: 0.0}
gradiente_evaluado = [g.evalf(subs=punto) for g in gradiente]

print("\nGradiente evaluado en el punto [0, 0, 0]:")
print(gradiente_evaluado)
