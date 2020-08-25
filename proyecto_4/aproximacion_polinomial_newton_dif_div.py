import numpy as np
from matplotlib import pyplot as plt
import pandas as pd 

data = pd.read_csv("poblacion_inegi.csv")

#datos
#yData = np.array(data[['poblacion']], dtype=np.float32)
#xData = np.array(data[['Periodo']], dtype=np.float32)
#x =  2007 #el punto de interpolacion del arreglo x

xData = np.array([0,3,5,8,13])
yData = np.array([0,69,117,190,303])
x = 11
#algoritmo newton diferencias divididas
n = len(xData) - 1 
def eval_pol(a,xData,x):
    n = len(xData) - 1  # Numero de polinomios
    p = a[n]
    for k in range(1,n+1):
        p = a[n-k] + (x -xData[n-k])*p
    return p

def coef_pol(xData,yData):
    m = len(xData)  # Numero de datos del arreglo x
    a = yData.copy()
    for k in range(1,m):
        a[k:m] = (a[k:m] - a[k-1])/(xData[k:m] - xData[k-1])
    return a

#evaluar interpolacion
a = coef_pol(xData,yData)
Pnx = eval_pol(a,xData,x)
print("P%i(%i): %f" % (n, x, Pnx))
plt.scatter(x, Pnx, c='black')
plt.text(x,Pnx,"P%i(%.2f)= %f" % (n, x, Pnx))

#generar grafica con interpolacion cada 0.5
Yaprox = []
for x in np.arange(xData[0], xData[-1], 0.5):
    Y_ap = eval_pol(a, xData, x)
    Yaprox.append(Y_ap)

#grafica
plt.plot(xData, yData, 'b', label="Grafica con {} puntos".format(len(xData)))
plt.grid()
plt.suptitle("Aproximacion Polinomial de Newton en diferencias divididas")
plt.title("interpolacion del censo de la población del INEGI", fontsize=10)
plt.ylabel("Población")
plt.xlabel("Periodo")
plt.legend(loc="best")
#plt.show()  

