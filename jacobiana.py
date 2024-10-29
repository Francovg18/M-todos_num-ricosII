from sympy import symbols, diff, Matrix, sin, cos

# Definir las variables simbólicas
x, y = symbols('x y')

# Definir las funciones (modifica según lo que necesites)
f1 = x**2 - y
f2 = x**4 + y**4 -x**2 -x + y +2


# Crear una lista con las funciones
funciones = [f1, f2]

# Crear una lista con las variables
variables = [x, y]

# Calcular la matriz Jacobiana
jacobiana = Matrix([[diff(f, var) for var in variables] for f in funciones])

# Imprimir la matriz Jacobiana
print(jacobiana)
