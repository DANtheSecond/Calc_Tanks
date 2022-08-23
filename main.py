import numpy as np
from scipy.interpolate import CubicSpline


# Input Data
Task = int(input('Task type (1 - Side task; 2 - End task): '))
R = float(input('Radius of the tank, m: '))
H = float(input('Height of the tank, m: '))
if Task == 1:
    h = float(input('Height level of the reference point, m: '))
dt = int(input('Protection material (1 - Concrete; 2 - Lead; 3 - Iron): '))
d = float(input('Protection thickness, mm: '))
a = float(input('Distance from tank to the reference point, m: '))
A = float(input('Activity level, Bq: '))
E = float(input('Gamma energy, MeV: '))
q = float(input('Emission rate, rel.un.: '))

# Several tanks:
# N = int(input('Tanks number: '))
# a = np.zeros(N)
# for i in range(len(a)):
#     print('Distance from tank №', i, 'to the reference point, m: ')
#     a[i] = float(input())


# Tables
from mumatrix import tab_G
from CoeffConv import tab_En, tab_Ga
tab_E = np.array([0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1, 0.145, 0.15, 0.2, 0.279, 0.3, 0.4, 0.412, 0.5,
                  0.6, 0.662, 0.8, 1, 1.25, 1.5, 2, 2.75, 3, 4, 5, 6, 8, 10])
# tab_mu
st1 = [4.99, 60.3, 1390, 1330]
st2 = [1.5, 18.4, 1210, 440]
st3 = [0.707, 7.87, 939, 196]
st4 = [0.325, 2.48, 323, 61.3]
st5 = [0.238, 1.22, 151, 26.8]
st6 = [0.207, 0.784, 82.1, 14.2]
st7 = [0.192, 0.596, 50.8, 8.72]
st8 = [0.175, 0.442, 23.6, 4.22]
st9 = [0.165, 0.382, 60.3, 2.6]
st10 = [0.150, 0.320, 24.6, 1.51]
st11 = [0.148, 0.317, 21.8, 1.39]
st12 = [0.136, 0.285, 10.7, 1.06]
st13 = [0.121, 0.253, 4.65, 0.865]
st14 = [0.118, 0.246, 4.25, 0.833]
st15 = [0.106, 0.219, 2.44, 0.717]
st16 = [0.105, 0.216, 2.32, 0.707]
st17 = [0.0966, 0.2, 1.7, 0.646]
st18 = [0.0894, 0.185, 1.33, 0.595]
st19 = [0.0857, 0.177, 1.18, 0.570]
st20 = [0.0786, 0.163, 0.952, 0.520]
st21 = [0.0706, 0.146, 0.771, 0.467]
st22 = [0.0631, 0.131, 0.658, 0.422]
st23 = [0.0575, 0.119, 0.577, 0.381]
st24 = [0.0494, 0.103, 0.508, 0.333]
st25 = [0.0410, 0.0874, 0.476, 0.291]
st26 = [0.0397, 0.0837, 0.468, 0.284]
st27 = [0.0340, 0.0734, 0.472, 0.260]
st28 = [0.0303, 0.0665, 0.481, 0.248]
st29 = [0.0277, 0.0619, 0.494, 0.240]
st30 = [0.0243, 0.0561, 0.520, 0.234]
st31 = [0.0222, 0.0529, 0.55, 0.234]
tab_mu = np.array(
    [st1, st2, st3, st4, st5, st6, st7, st8, st9, st10, st11, st12, st13, st14, st15, st16, st17, st18, st19, st20,
     st21, st22, st23, st24, st25, st26, st27, st28, st29, st30, st31])
tab_p = np.array([1.25, 1.5, 3, 5, 10])
tab_b = np.array([0, 1, 2, 4, 7, 10, 20])
tab_k = np.array([0, 1, 3, 5, 10])
tab_musR = np.array([0, 1, 2, 3, 5, 10])

# Functions
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
        A0[0, 1] = np.sum(np.log(x + ones))
        A0[1, 0] = np.sum(np.log(y) * np.log(x + ones))
        A1[0, 0] = float(len(x))
        A1[0, 1] = np.sum(np.log(x + ones))
        A1[1, 0] = np.sum(np.log(x + ones))
        for i in range(len(x)):
            A0[1, 1] += np.log(x[i] + 1) ** 2
            A1[1, 1] += np.log(x[i] + 1) ** 2
        a = np.exp(np.linalg.det(A0) / np.linalg.det(A1))
        A0[0, 0] = float(len(x))
        A0[0, 1] = np.sum(np.log(y))
        A0[1, 0] = np.sum(np.log(x + ones))
        A0[1, 1] = np.sum(np.log(y) * np.log(x + ones))
        A1[0, 0] = float(len(x))
        A1[0, 1] = np.sum(np.log(x + ones))
        A1[1, 0] = np.sum(np.log(x + ones))
        A1[1, 1] = 0
        for i in range(len(x)):
            A1[1, 1] += np.log(x[i] + 1) ** 2
        b = np.linalg.det(A0) / np.linalg.det(A1)
        c = (y[0] / a) ** (-b)
        return (a * (x1 + c) ** b)
    else:
        A0 = np.array([[0., 0.],
                       [0., 0.]])
        A1 = np.array([[0., 0.],
                       [0., 0.]])
        A0[0, 0] = np.sum(np.log(y))
        A0[0, 1] = np.sum(np.log(x))
        A0[1, 0] = np.sum(np.log(y) * np.log(x))
        A1[0, 0] = float(len(x))
        A1[0, 1] = np.sum(np.log(x))
        A1[1, 0] = np.sum(np.log(x))
        for i in range(len(x)):
            A0[1, 1] += np.log(x[i]) ** 2
            A1[1, 1] += np.log(x[i]) ** 2
        a = np.exp(np.linalg.det(A0) / np.linalg.det(A1))
        A0[0, 0] = float(len(x))
        A0[0, 1] = np.sum(np.log(y))
        A0[1, 0] = np.sum(np.log(x))
        A0[1, 1] = np.sum(np.log(y) * np.log(x))
        A1[0, 0] = float(len(x))
        A1[0, 1] = np.sum(np.log(x))
        A1[1, 0] = np.sum(np.log(x))
        A1[1, 1] = 0
        for i in range(len(x)):
            A1[1, 1] += np.log(x[i]) ** 2
        b = np.linalg.det(A0) / np.linalg.det(A1)
        return(a * x1 ** b)


def ApproxExp(x, y, x1):
    if y[0] == 0:
        return 0
    A0 = np.array([[0., 0.],
                   [0., 0.]])
    A1 = np.array([[0., 0.],
                   [0., 0.]])
    A0[0, 0] = np.sum(np.log(y))
    A0[0, 1] = np.sum(x)
    A0[1, 0] = np.sum(np.log(y) * x)
    A1[0, 0] = float(len(x))
    A1[0, 1] = np.sum(x)
    A1[1, 0] = np.sum(x)
    for i in range(len(x)):
        A0[1, 1] += x[i] ** 2
        A1[1, 1] += x[i] ** 2
    a = np.exp(np.linalg.det(A0) / np.linalg.det(A1))
    A0[0, 0] = float(len(x))
    A0[0, 1] = np.sum(np.log(y))
    A0[1, 0] = np.sum(x)
    A0[1, 1] = np.sum(np.log(y) * x)
    A1[0, 0] = float(len(x))
    A1[0, 1] = np.sum(x)
    A1[1, 0] = np.sum(x)
    A1[1, 1] = 0
    for i in range(len(x)):
        A1[1, 1] += x[i] ** 2
    b = np.linalg.det(A0) / np.linalg.det(A1)
    return(a * np.exp(b * x1))


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


# Calculations
p = a / R
k1 = (H - h) / R
k2 = h / R
if E > tab_E.max() or E < tab_E.min():
    f = CubicSpline(tab_E, tab_mu[:, 0], extrapolate=True)
    if f(E) > 0:
        mus = f(E)
    else:
        mus = ApproxPov(tab_E, tab_mu[:, 0], E)
    f = CubicSpline(tab_E, tab_mu[:, dt], extrapolate=True)
    if f(E) > 0:
        mud = f(E)
    else:
        mud = ApproxPov(tab_E, tab_mu[:, dt], E)
else:
    mus = ApproxLin(tab_E, tab_mu[:, 0], E)
    mud = ApproxLin(tab_E, tab_mu[:, dt], E)
print('μs =', mus)
print('μd =', mud)
musR = mus*R*100
b = mud*d/10
s = np.zeros(len(tab_musR))
Gpbk = np.zeros((len(tab_k), len(tab_b), len(tab_p)))
for i in range(len(tab_p)):
    for j in range(len(tab_b)):
        for n in range(len(tab_k)):
            for m in range(len(tab_musR)):
                s[m] = tab_G[m, n, j, i]
            Gpbk[n, j, i] = ApproxPov(tab_musR, s, musR)
s = np.zeros(len(tab_k))
Gpb1 = np.zeros((len(tab_b), len(tab_p)))
Gpb2 = np.zeros((len(tab_b), len(tab_p)))
for i in range(len(tab_p)):
    for j in range(len(tab_b)):
        for n in range(len(tab_k)):
            s[n] = Gpbk[n, j, i]
        f = CubicSpline(tab_k, s, extrapolate=True)
        if f(k1) < 0:
            Gpb1[j, i] = ApproxLin(tab_k, s, k1)
        else:
            Gpb1[j, i] = f(k1)
        if f(k2) < 0:
            Gpb2[j, i] = ApproxLin(tab_k, s, k2)
        else:
            Gpb2[j, i] = f(k2)
s = np.zeros(len(tab_b))
Gp1 = np.zeros(len(tab_p))
Gp2 = np.zeros(len(tab_p))
for i in range(len(tab_p)):
    for j in range(len(tab_b)):
        s[j] = Gpb1[j, i]
    Gp1[i] = ApproxExp(tab_b, s, b)
    for j in range(len(tab_b)):
        s[j] = Gpb2[j, i]
    Gp2[i] = ApproxExp(tab_b, s, b)
G1 = ApproxPov(tab_p, Gp1, p)
if k2 == 0:
    G2 = 0
else:
    G2 = ApproxPov(tab_p, Gp2, p)
V = H * 2 * np.pi * R**2
D = 2 * A * ApproxLin(tab_En, tab_Ga, E) * q * R / V * (G1 + G2)

#Results
print('p = ', p)
print('b = ', b)
print('μsR = ', musR)
print("k' = ", k1, "| k'' = ", k2)
print("G' = ", G1, "| G'' = ", G2)
print('Г = ', ApproxLin(tab_En, tab_Ga, E) * q)
print('V = ', V)
print('D = ', D)