import math
import numpy as np  
import matplotlib.pyplot as plt 
from scipy import integrate
import scipy.integrate as integrate

def gaussNodes(m,tol=10e-9):
    def legendre(t,m):
        p0 = 1.0; p1 = t
        for k in range(1,m):
            p = ((2.0*k + 1.0)*t*p1 - k*p0)/(1.0 + k )
            p0 = p1; p1 = p
        dp = m*(p0 - t*p1)/(1.0 - t**2)
        return p,dp

    A = np.zeros(m)   
    x = np.zeros(m)   
    nRoots = int((m + 1)/2)         #numero de no neg.
    for i in range(nRoots):
        t = math.cos(math.pi*(i + 0.75)/(m + 0.5)) #raiz aprox
        for j in range(30): 
            p,dp = legendre(t,m)    # Newton-Raphson
            dt = -p/dp; t = t + dt      
            if abs(dt) < tol:
                x[i] = t; x[m-i-1] = -t
                A[i] = 2.0/(1.0 - t**2)/(dp**2) 
                A[m-i-1] = A[i]
                break
    return x,A

def gaussQuad(f,a,b,m): 
    c1 = (b + a)/2.0
    c2 = (b - a)/2.0
    x,A = gaussNodes(m)
    sum = 0.0
    for i in range(len(x)):
        sum = sum + A[i]*f(c1 + c2*x[i])
    return c2*sum    

#grafica
def graph(formula, x_range):  
    x = np.array(x_range)  
    y = np.vectorize(formula)
    plt.plot(x, y(x))  
    plt.text(-3.5,0.25, "Cuadratura de Gauss  %i puntos: %f" % (m, aproxQuad))
    plt.show()

#funcion de casos covid
def f(x): 
    #return (1/np.sqrt(2*np.pi))*(math.e**(-x**2/2))
    return (x**3)*(math.e**x)-2*x+3
#a = -1; b = 0 #valores para la cuadratura de gauss
#m = 5; #numero de puntos
a=0 ; b=np.sqrt(2)
m=3
aproxQuad = gaussQuad(f, a, b, m)
print("Integral con %i puntos: %f" % (m, aproxQuad))

#grafica
#graph(f, np.arange(-4,0, 0.1))

#valor exacto de la solucion analitica
analitica = integrate.quadrature(f, a, b)
print("Valor exacto de la solucion analitica: ",analitica[0])
erp = (abs((abs(analitica[0]-aproxQuad))/analitica[0]))*100
print("ERP(%): ", erp)
