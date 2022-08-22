import math
import numpy as np
from scipy.interpolate import CubicSpline

E = float(input('Введите энергию, МэВ: '))
mat = 1.5 #'Вода' #input('Введите материал защиты:')

tab_mat = [1, 1.5, 3]#['Вода','Бетон','Свинец'] #x
tab_E = np.array([0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1, 0.145, 0.15, 0.2, 0.279, 0.3, 0.4, 0.412, 0.5, 0.6, 0.662, 0.8, 1, 1.25, 1.5, 2, 2.75, 3, 4, 5, 6, 8, 10])  #y

st1 = [4.99, 60.3, 1390]
st2 = [1.5, 18.4, 1210]
st3 = [0.707, 7.87, 939]
st4 = [0.325, 2.48, 323]
st5 = [0.238, 1.22, 151]
st6 = [0.207, 0.784, 82.1]
st7 = [0.192, 0.596, 50.8]
st8 = [0.175, 0.442, 23.6]
st9 = [0.165, 0.382, 60.3]
st10 = [0.150, 0.320, 24.6]
st11 = [0.148, 0.317, 21.8]
st12 = [0.136, 0.285, 10.7]
st13 = [0.121, 0.253, 4.65]
st14 = [0.118, 0.246, 4.25]
st15 = [0.106, 0.219, 2.44]
st16 = [0.105, 0.216, 2.32]
st17 = [0.0966, 0.2, 1.7]
st18 = [0.0894, 0.185, 1.33]
st19 = [0.0857, 0.177, 1.18]
st20 = [0.0786, 0.163, 0.952]
st21 = [0.0706, 0.146, 0.771]
st22 = [0.0631, 0.131, 0.658]
st23 = [0.0575, 0.119, 0.577]
st24 = [0.0494, 0.103, 0.508]
st25 = [0.0410, 0.0874, 0.476]
st26 = [0.0397, 0.0837, 0.468]
st27 = [0.0340, 0.0734, 0.472]
st28 = [0.0303, 0.0665, 0.481]
st29 = [0.0277, 0.0619, 0.494]
st30 = [0.0243, 0.0561, 0.520]
st31 = [0.0222, 0.0529, 0.55]

tab_mu = np.array([st1, st2, st3, st4, st5, st6, st7, st8, st9, st10, st11, st12, st13, st14, st15, st16, st17, st18, st19, st20, st21, st22, st23, st24, st25, st26, st27, st28, st29, st30, st31])
print(tab_E.max())
#функция линейной аппроксимации
def lextrap2d(tab_x, tab_y, tab_z, x, y):
    if x >= tab_x[-1]:
        x1 = tab_x[-1]
        x2 = tab_x[-2]
    else:
        for i in range(len(tab_x) - 1):
            if x > tab_x[i]:
                x1 = tab_x[i]
                x2 = tab_x[i+1]
    if y >= tab_y[-1]:
        y1 = tab_y[-1]
        y2 = tab_y[-2]
    else:
        for i in range(len(tab_y) - 1):
            if y > tab_y[i]:
                y1 = tab_y[i]
                y2 = tab_y[i+1]
    z1 = tab_z[tab_y.index(y1), tab_x.index(x1)]+(tab_z[tab_y.index(y1), tab_x.index(x2)]-tab_z[tab_y.index(y1), tab_x.index(x1)])*(x-x1)/(x2-x1)
    z2 = tab_z[tab_y.index(y2), tab_x.index(x1)]+(tab_z[tab_y.index(y2), tab_x.index(x2)]-tab_z[tab_y.index(y2), tab_x.index(x1)])*(x-x1)/(x2-x1)
    z = z1+(z2-z1)*(y-y1)/(y2-y1)
    return z

mut = lextrap2d(tab_mat, tab_E, tab_mu, mat, E)
print('lextrap2d: ', mut)

#функция кубической аппроксимации
def ApproxCub(tab_x, tab_y, tab_z, x, y):
    zx = np.arange(float(len(tab_x)))
    zy = np.arange(float(len(tab_y)))
    for i in range(len(tab_y)):
        for j in range(len(tab_x)):
            zx[j] = tab_z[i, j]
        f = CubicSpline(tab_x, zx, extrapolate = True)
        zy[i] = f(x)
    f = CubicSpline(tab_y, zy, extrapolate = True)
    z = f(y)
    return z

mut = ApproxCub(tab_mat, tab_E, tab_mu, mat, E)
print('ApproxCub: ', mut)

if E > tab_E[-1]:
    E1 = tab_E[-1]
    E2 = tab_E[-2]
else:
    for i in range(len(tab_E)-1):
        if E > tab_E[i]:
            E1 = tab_E[i]
            E2 = tab_E[i+1]

mu = tab_mu[tab_E.index(E1), tab_mat.index(mat)] + (tab_mu[tab_E.index(E2), tab_mat.index(mat)]-tab_mu[tab_E.index(E1), tab_mat.index(mat)])*(E-E1)/(E2-E1)
print('Расчёт: ', mu)
