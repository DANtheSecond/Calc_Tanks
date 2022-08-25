import numpy as np
from scipy.interpolate import CubicSpline
import openpyxl as op

# Input Data
inp = op.load_workbook("input.xlsx", data_only=True).active
if inp.cell(row=2, column=1).value == 'Radial':
    Task = 1
else:
    Task = 2
R = inp.cell(row=2, column=2).value
H = inp.cell(row=2, column=3).value
if Task == 1:
    h = inp.cell(row=2, column=4).value
a = inp.cell(row=1, column=12).value
dt = np.zeros(a).astype(int)
d = np.zeros(a)
for i in range(2, a + 2):
    dt[i - 2] = inp.cell(row=i, column=12).value
    d[i - 2] = inp.cell(row=i, column=6).value
a = inp.cell(row=2, column=7).value
E = np.zeros(inp.max_row - 1)
q = np.zeros(inp.max_row - 1)
A = np.zeros(inp.max_row - 1)
for i in range(2, inp.max_row + 1):
    E[i - 2] = inp.cell(row=i, column=9).value
    q[i - 2] = inp.cell(row=i, column=10).value
    A[i - 2] = inp.cell(row=i, column=11).value

# Tables
from mumatrix import tab_p, tab_b, tab_k, tab_musR, tab_E, tab_mu, tab_G
from CoeffConv import tab_En, tab_Ga


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
        return (a * x1 ** b)


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
    return (a * np.exp(b * x1))


def ApproxLin(x, y, x1):
    if x1 > x.max():
        x0 = x[-1]
        x2 = x[-2]
    else:
        x0 = x[0]
        x2 = x[1]
        for i in range(len(x) - 1):
            if x1 > x[i]:
                x0 = x[i]
                x2 = x[i + 1]
            else:
                break
    z = y[np.where(x == x0)] + (y[np.where(x == x2)] - y[np.where(x == x0)]) * (x1 - x0) / (x2 - x0)
    return z


# Creating workbook
results = op.Workbook()
res = results.active
res['A1'] = 'Energy, E, MeV'
res['B1'] = 'Gamma emission, q, rel.un.'
res['C1'] = 'Г, μSv*m2/(h*Bq)'
res['D1'] = 'μs, rel.un.'
res['E1'] = 'Protection material'
res['F1'] = 'Protection thickness, mm'
for i in range(2, len(d) + 2):
    res.cell(row=i, column=5).value = dt[i - 2]
    res.cell(row=i, column=6).value = d[i - 2]
res['G1'] = 'Protection μ, rel.un.'
res['H1'] = 'p, rel.un.'
res['I1'] = "b, rel.un."
res['J1'] = "k', rel.un."
res['K1'] = 'k", rel.un.'
res['L1'] = 'μsR, rel.un.'
res['M1'] = "G', rel.un."
res['N1'] = 'G", rel.un.'
res['O1'] = 'D, μSv/h'
res['P1'] = 'Total dose, μSv/h'

# Calculations
D = np.zeros(len(E))
p = a / R
k1 = (H - h) / R
k2 = h / R
res.cell(row=2, column=8).value = p
res.cell(row=2, column=10).value = k1
res.cell(row=2, column=11).value = k2
for e in range(len(E)):
    res.cell(row=e + 2, column=1).value = E[e]
    res.cell(row=e + 2, column=2).value = q[e]
    res.cell(row=e + 2, column=3).value = float(ApproxLin(tab_En, tab_Ga, E[e]) * q[e])
    b = 0.
    mud = np.zeros(len(dt))
    if E[e] > tab_E.max() or E[e] < tab_E.min():
        f = CubicSpline(tab_E, tab_mu[:, 0], extrapolate=True)
        if f(E[e]) > 0:
            mus = f(E[e])
        else:
            mus = ApproxPov(tab_E, tab_mu[:, 0], E[e])
    else:
        mus = ApproxLin(tab_E, tab_mu[:, 0], E[e])
    res.cell(row=e + 2, column=4).value = float(mus)
    print('μs =', mus)
    for i in range(len(dt)):
        if E[e] > tab_E.max() or E[e] < tab_E.min():
            f = CubicSpline(tab_E, tab_mu[:, dt[i]], extrapolate=True)
            if f(E[e]) > 0:
                mud[i] = f(E[e])
            else:
                mud[i] = ApproxPov(tab_E, tab_mu[:, dt[i]], E[e])
        else:
            mud[i] = ApproxLin(tab_E, tab_mu[:, dt[i]], E[e])
        print('μd[', i, '] =', mud[i])
        b += (mud[i] * d[i]) / 10
    res.cell(row=e + 2, column=9).value = b
    musR = float(mus * R * 100)
    res.cell(row=e + 2, column=12).value = musR
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
    res.cell(row=e + 2, column=13).value = G1
    if k2 == 0:
        G2 = 0
    else:
        G2 = ApproxPov(tab_p, Gp2, p)
    res.cell(row=e + 2, column=14).value = G2
    V = H * 2 * np.pi * R ** 2
    D[e] = 2 * A[e] * ApproxLin(tab_En, tab_Ga, E[e]) * q[e] * R / V * (G1 + G2)
    res.cell(row=e + 2, column=15).value = D[e]
    print('D[', e, ']= ', D[e], 'μSv/h')

# Results
    print('p = ', p)
    print('b = ', b)
    print('μsR = ', musR)
    print("k' = ", k1, "| k'' = ", k2)
    print("G' = ", G1, "| G'' = ", G2)
    print('E = ', E[e])
    print('q = ', q[e])
    print('Г = ', float(ApproxLin(tab_En, tab_Ga, E[e]) * q[e]))
    print('V = ', V)
print('Total dose rate, D = ', np.sum(D), 'μSv/h')
res.cell(row=2, column=16).value = np.sum(D)
results.save("Results.xlsx")