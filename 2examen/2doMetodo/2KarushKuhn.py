import numpy as np 
from scipy.optimize import minimize 
 
# Definimos la función objetivo 
def objective(x, maximize=False): 
    x1, x2 = x 
    value = (x1 - 1)*2 + (x2 - 2)*2 #Aqui ponemos Z 
     
    return -value if maximize else value   
 
# Definimos las restricciones 
def constraint1(x): 
    x1, x2 = x 
    return -x1 -x2 +1 #Aqui ponemos la restriccion, y en caso de haber mas de una duplicamos la funcion 
 
 
# Configuramos las restricciones en el formato que necesita scipy 
cons = [{'type': 'ineq', 'fun': constraint1}] #Si hay mas de una restriccion por una coma anadimos al arreglo {'type': 'ineq', 'fun': constraint1} con el constraint cambiado 
 
# Valores iniciales 
x0 = [0, 0] #Cambiar solo en caso de que obtengamos todo en 0 
 
# Configuración: maximizar o minimizar 
maximize = False  # Cambiar a True si deseas maximizar 
 
# Resolvemos el problema de optimización 
solution = minimize(objective, x0, args=(maximize,), constraints=cons) 
 
# Mostramos los resultados 
if solution.success: 
    print("Solución encontrada:") 
    print(f"x1: {solution.x[0]:.4f}, x2: {solution.x[1]:.4f}") 
    print(f"Valor {'máximo' if maximize else 'mínimo'} de la función objetivo: {solution.fun:.4f}") 
else: 
    print("No se pudo encontrar una solución.")