import numpy as np
import matplotlib.pyplot as plt

def f(x):
    # return x**3 + np.log(np.sqrt(x)) + np.sin(x) - 2
    return (9*np.exp(-0.7*x))*(np.cos(4*x))

def signosIguales(x,y):
    if (x<0) and (y>=0):
        return False
    if (y<0) and (x>=0):
        return False 
    return True


def biseccion(a,b,n):
    arreglo_c=[]
    i = 1
    while i<=n:
        c = (a+b)/2
        arreglo_c.append(c)
        print(a,b,c)
        if not signosIguales(f(a),f(c)):
            b = c
        if not signosIguales(f(b),f(c)):
            a = c
        i+=1
    return arreglo_c[n-1]

# a=0.95
# b=1.1
# error=0.01
# n=14

error=0.01
a=0
b=3.5
n= int(round((np.log(b-a/error))/np.log(2)))

print("n: ", n)

k=biseccion(a,b,n)
print(k)

print("Error:", round(f(k), 4))

#grafica
eje_x = np.linspace(a,b,100)
plt.plot(eje_x, f(eje_x))
plt.grid()
plt.scatter(k,f(k))
plt.show()
