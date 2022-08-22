import numpy as np

tab_x = [0, 1, 2, 4, 7, 10, 20]
tab_y = [1.400e-1, 4.343e-2, 1.348e-2, 1.443e-3, 5.701e-5, 2.406e-6, 7.606e-11]
x1 = 22
def ApproxExp(x, y, x1):
    A0 = np.array([[0., 0.],
                   [0., 0.]])
    A1 = np.array([[0., 0.],
                   [0., 0.]])
    A0[0, 0] = np.sum(np.log(y))
    A0[0, 1] = np.sum(x)
    A0[1, 0] = np.sum(np.log(y)*x)
    A1[0, 0] = float(len(x))
    A1[0, 1] = np.sum(x)
    A1[1, 0] = np.sum(x)
    for i in range(len(x)):
        A0[1, 1] += x[i]**2
        A1[1, 1] += x[i]**2
    a = np.exp(np.linalg.det(A0)/np.linalg.det(A1))
    A0[0, 0] = float(len(x))
    A0[0, 1] = np.sum(np.log(y))
    A0[1, 0] = np.sum(x)
    A0[1, 1] = np.sum(np.log(y)*x)
    A1[0, 0] = float(len(x))
    A1[0, 1] = np.sum(x)
    A1[1, 0] = np.sum(x)
    A1[1, 1] = 0
    for i in range(len(x)):
        A1[1, 1] += x[i]**2
    b = np.linalg.det(A0)/np.linalg.det(A1)
    return(a*np.exp(b*x1))


print(ApproxExp(tab_x, tab_y, x1))