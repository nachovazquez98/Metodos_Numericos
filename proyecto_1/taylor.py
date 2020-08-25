import sympy as sy
import numpy as np
import math
from sympy.functions import sin, cos, ln
import matplotlib.pyplot as plt
plt.style.use("ggplot")

# Función factorial
def factorial(n):
    if n <= 0:
        return 1
    else:
        return n * factorial(n - 1)

# Taylor aproximacion en x0 de la función 'función'
def taylor(function, x0, n, x = sy.Symbol('x')):
    i = 0
    p = 0
    while i <= n:
        p = p + (function.diff(x, i).subs(x, x0))/(factorial(i))*(x - x0)**i
        i += 1
    return p

def plot(f, x0 = 0, n = 9, h = 2, x_lims = [-5, 5], y_lims = [-5, 5], npoints = 800, x = sy.Symbol('x')):
    x1 = np.linspace(x_lims[0], x_lims[1], npoints)
    # Aproximacion hasta n a partir de 1 y usando pasos de por
    for j in range(1, n + 1, h):
        func = taylor(f, x0, j)
        taylor_lambda = sy.lambdify(x, func, "numpy")
        print('Taylor expansion at n=' + str(j), func)
        plt.plot(x1, taylor_lambda(x1), label = 'Order '+ str(j))
    # Trace la función para aproximar (seno, en este caso)
    func_lambda = sy.lambdify(x, f, "numpy")
    plt.plot(x1, func_lambda(x1), label = 'Funcion de x')
    
    plt.xlim(x_lims)
    plt.ylim(y_lims)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.title('Aproximacion de serie de Taylor')
    plt.show()

# Defina la variable y la función para aproximar
x = sy.Symbol('x')
amplitud = 5
frecuencia = 1.1
f = amplitud*sin(frecuencia*x)
plot(f)
