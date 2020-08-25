from math import e
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def F(x,y): #funcion
    F = np.zeros(1)
    F[0] = 37.5-3.5*y[0]
    return F

#y(0) = 50
x = 0.0
y = np.array([50.0])

#valor de la iteracion
xStop = 3.0

#h
h = 0.1875

#eje de las x para graficar
eje_x = np.arange(x,xStop+h,h)

def integrate(F,x,y,xStop,h):
    def runge_kutta4(F,x,y,h):
        K0 = h*F(x,y)
        K1 = h*F(x + h/2.0, y + K0/2.0)
        K2 = h*F(x + h/2.0, y + K1/2.0)
        K3 = h*F(x + h, y + K2)
        return (K0 + 2.0*K1 + 2.0*K2 + K3)/6.0    
    X = []
    Y = []
    X.append(x)
    Y.append(y)
    while x < xStop:
        h = min(h,xStop - x)
        y = y + runge_kutta4(F,x,y,h)
        x = x + h
        X.append(x)
        Y.append(y)
    return np.array(X),np.array(Y)

def printSol(X,Y):
    print("x","y[0.0]", sep="\t")
    for i in range(len(X)):
        print(np.round(X[i], 6), np.round(Y[i],6), sep='\t')

X,Y = integrate(F,x,y,xStop,h) 
printSol(X,Y)       

#graficar
plt.plot(eje_x,Y, color='black')
plt.xlabel('Tiempo en Minutos')
plt.ylabel('Cambio de la concentracion de Sal')
plt.title('Concentracion de Sal')
plt.text(0.5,20,'Sal')
plt.axis()
plt.grid()

plt.show()

