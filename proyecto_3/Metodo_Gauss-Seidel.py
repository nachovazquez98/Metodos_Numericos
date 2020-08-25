import numpy as np

def gauss(A, b, x, n):
    print("\n")
    L = np.tril(A)
    U = A - L
    for i in range(n):
        x = np.dot(np.linalg.inv(L), b - np.dot(U, x))
        print(f"{str(i).zfill(3)}: {x}")
    return x

'''___MAIN___'''
A = np.array([[530.0, -220.0, 0.0], [-220.0, 700.0, -240.0], [0.0, -240.0, 510.0]])
b = [10.0, 0.0, 12.0]
x = [1, 1, 1]

print("Sistema de Ecuaciones:")
for i in range(A.shape[0]):
    row = ["{0:3g}*x{1}".format(A[i, j], j + 1) for j in range(A.shape[1])]
    print("[{0}] = [{1:3g}]".format(" + ".join(row), b[i]))
print("\n")

#No. Iteraciones
n = 20

#A representa la matriz
print("A:\n", A)

#representa la solución a Ax = b
print("b:\n", b)
print("\nGauss: ", gauss(A, b, x, n))

