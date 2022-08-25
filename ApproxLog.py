import numpy as np
tab_x = [0.1, 0.5, 1, 2]
tab_y = [1.180e-7, 1.897e-6, 3.169e-6, 3.386e-6]
x1 = 5
def ApproxLog(x, y, x1):
    A0 = np.array([[0., 0.],
                   [0., 0.]])
    A1 = np.array([[0., 0.],
                   [0., 0.]])
    A0[0, 0] = np.sum(y)
    A0[0, 1] = np.sum(np.log(x))
    A0[1, 0] = np.sum(y*np.log(x))
    A1[0, 0] = float(len(x))
    A1[0, 1] = np.sum(np.log(x))
    A1[1, 0] = np.sum(np.log(x))
    for i in range(len(x)):
        A0[1, 1] += np.log(x[i])**2
        A1[1, 1] += np.log(x[i])**2
    a = np.linalg.det(A0)/np.linalg.det(A1)
    print('a = ', a)
    A0[0, 0] = float(len(x))
    A0[0, 1] = np.sum(y)
    A0[1, 0] = np.sum(np.log(x))
    A0[1, 1] = np.sum(y*np.log(x))
    A1[0, 0] = float(len(x))
    A1[0, 1] = np.sum(np.log(x))
    A1[1, 0] = np.sum(np.log(x))
    A1[1, 1] = 0
    for i in range(len(x)):
        A1[1, 1] += np.log(x[i])**2
    b = np.linalg.det(A0)/np.linalg.det(A1)
    print('b = ', b)
    return(a+b*np.log(x1))

print(ApproxLog(tab_x, tab_y, x1))