import numpy as np

tab_x = [1, 2, 3, 5, 10]
tab_y = [5.600e-1, 3.550e-1, 2.600e-1, 1.500e-1, 8.000e-2]
x1 = 5
def ApproxExp(x, y, x1):
    A0 = np.array([[0., 0.],
                   [0., 0.]])
    A1 = np.array([[0., 0.],
                   [0., 0.]])
    A0[0, 0] = np.sum(np.log(y))
    A0[0, 1] = np.sum(np.log(x))
    A0[1, 0] = np.sum(np.log(y)*np.log(x))
    A1[0, 0] = float(len(x))
    A1[0, 1] = np.sum(np.log(x))
    A1[1, 0] = np.sum(np.log(x))
    for i in range(len(x)):
        A0[1, 1] += np.log(x[i])**2
        A1[1, 1] += np.log(x[i])**2
    a = np.exp(np.linalg.det(A0)/np.linalg.det(A1))
    print(a)
    A0[0, 0] = float(len(x))
    A0[0, 1] = np.sum(np.log(y))
    A0[1, 0] = np.sum(np.log(x))
    A0[1, 1] = np.sum(np.log(y)*np.log(x))
    A1[0, 0] = float(len(x))
    A1[0, 1] = np.sum(np.log(x))
    A1[1, 0] = np.sum(np.log(x))
    A1[1, 1] = 0
    for i in range(len(x)):
        A1[1, 1] += np.log(x[i])**2
    b = np.linalg.det(A0)/np.linalg.det(A1)
    print(b)
    return(a*x1**b)


print(ApproxExp(tab_x, tab_y, x1))