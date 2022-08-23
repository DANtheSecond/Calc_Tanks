import numpy as np


tab_En = np.array([0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1, 1.5, 2,
                  3, 4, 5, 6, 8, 10])
tab_mub = np.array([4.79, 1.28,	0.512, 0.149, 0.0677, 0.0418, 0.032, 0.0262, 0.0256, 0.0277, 0.0297, 0.0319, 0.0328,
                    0.033, 0.0329, 0.0321, 0.0311, 0.0284, 0.0262, 0.0229, 0.0209, 0.0195, 0.0185, 0.017, 0.0162])
tab_Ga = np.zeros(len(tab_En))
for i in range(len(tab_En)):
    tab_Ga [i] = 1275 * tab_En[i]*tab_mub[i]*3600*1e-12
def ApproxLin(x, y, x1):
    if x1 > x.max():
        x0 = x[-1]
        x2 = x[-2]
    else:
        x0 = x[0]
        x2 = x[1]
        for i in range(len(x)-1):
            if x1 > x[i]:
                x0 = x[i]
                x2 = x[i+1]
            else:
                break
    z = y[np.where(x == x0)]+(y[np.where(x == x2)]-y[np.where(x == x0)])*(x1 - x0)/(x2 - x0)
    return z