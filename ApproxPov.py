import numpy as np
#in case x[0] = 0, approximates with y(x) = a*(x+c)^b
tab_x = [0., 1., 2., 3., 5., 10.]
tab_y = [1.084, 5.600e-1, 3.550e-1, 2.600e-1, 1.500e-1, 8.000e-2]
x1 = 15.73
def ApproxPov(x, y, x1):
    if y[0] == 0:
        return 0
    if x[0] == 0:
        A0 = np.array([[0., 0.],
                       [0., 0.]])
        A1 = np.array([[0., 0.],
                       [0., 0.]])
        ones = np.ones(len(x))
        A0[0, 0] = np.sum(np.log(y))
        A0[0, 1] = np.sum(np.log(x+ones))
        A0[1, 0] = np.sum(np.log(y)*np.log(x+ones))
        A1[0, 0] = float(len(x))
        A1[0, 1] = np.sum(np.log(x+ones))
        A1[1, 0] = np.sum(np.log(x+ones))
        for i in range(len(x)):
            A0[1, 1] += np.log(x[i]+1)**2
            A1[1, 1] += np.log(x[i]+1)**2
        a = np.exp(np.linalg.det(A0)/np.linalg.det(A1))
        print('a = ', a)
        A0[0, 0] = float(len(x))
        A0[0, 1] = np.sum(np.log(y))
        A0[1, 0] = np.sum(np.log(x+ones))
        A0[1, 1] = np.sum(np.log(y)*np.log(x+ones))
        A1[0, 0] = float(len(x))
        A1[0, 1] = np.sum(np.log(x+ones))
        A1[1, 0] = np.sum(np.log(x+ones))
        A1[1, 1] = 0
        for i in range(len(x)):
            A1[1, 1] += np.log(x[i]+1)**2
        b = np.linalg.det(A0)/np.linalg.det(A1)
        print('b = ', b)
        c = (y[0]/a)**(-b)
        print('c = ', c)
        return(a*(x1+c)**b)
    else:
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
        print('a = ', a)
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
        print('b = ', b)
        return(a*x1**b)

print(ApproxPov(tab_x, tab_y, x1))