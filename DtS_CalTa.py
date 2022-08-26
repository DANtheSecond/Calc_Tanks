import numpy as np
from scipy.interpolate import CubicSpline
import openpyxl as op
from mumatrix import tab_p, tab_b, tab_k, tab_musR, tab_E, tab_mu, tab_G, tab_musH, tab_b1, tab_aH, tab_RH, tab_Z
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


def ApproxLog(x, y, x1):
    A0 = np.array([[0., 0.],
                   [0., 0.]])
    A1 = np.array([[0., 0.],
                   [0., 0.]])
    A0[0, 0] = np.sum(y)
    A0[0, 1] = np.sum(np.log(x))
    A0[1, 0] = np.sum(y * np.log(x))
    A1[0, 0] = float(len(x))
    A1[0, 1] = np.sum(np.log(x))
    A1[1, 0] = np.sum(np.log(x))
    for i in range(len(x)):
        A0[1, 1] += np.log(x[i]) ** 2
        A1[1, 1] += np.log(x[i]) ** 2
    a = np.linalg.det(A0) / np.linalg.det(A1)
    print('a = ', a)
    A0[0, 0] = float(len(x))
    A0[0, 1] = np.sum(y)
    A0[1, 0] = np.sum(np.log(x))
    A0[1, 1] = np.sum(y * np.log(x))
    A1[0, 0] = float(len(x))
    A1[0, 1] = np.sum(np.log(x))
    A1[1, 0] = np.sum(np.log(x))
    A1[1, 1] = 0
    for i in range(len(x)):
        A1[1, 1] += np.log(x[i]) ** 2
    b = np.linalg.det(A0) / np.linalg.det(A1)
    print('b = ', b)
    return (a + b * np.log(x1))


# Input Data
inp = op.load_workbook("input.xlsx", data_only=True)['Tank1']
N = inp.cell(row=3, column=1).value
results = op.Workbook()
res = results.active
res['S2'] = 'Total dose, μSv/h'
res['T2'] = "=SUM(Q:Q)"
T = int(2)
for t in range(N):
    sht = 'Sheet'+str(t+1)
    inp = op.load_workbook("input.xlsx", data_only=True)[sht]
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
        dt[i - 2] = inp.cell(row=1, column=i+11).value
        d[i - 2] = inp.cell(row=i, column=6).value
    a = inp.cell(row=2, column=7).value
    E = np.zeros(inp.max_row - 1)
    q = np.zeros(inp.max_row - 1)
    A = np.zeros(inp.max_row - 1)
    for i in range(2, inp.max_row + 1):
        E[i - 2] = inp.cell(row=i, column=9).value
        q[i - 2] = inp.cell(row=i, column=10).value
        A[i - 2] = inp.cell(row=i, column=11).value


# Creating workbook
    res.cell(row=T-1, column=1).value = 'Tank №' + str(t+1)
    res.cell(row=T, column=1).value = 'Energy, E, MeV'
    res.cell(row=T, column=2).value = 'Gamma emission, q, rel.un.'
    res.cell(row=T, column=3).value = 'Г, μSv*m2/(h*Bq)'
    res.cell(row=T, column=4).value = 'μs, rel.un.'
    res.cell(row=T, column=5).value = 'Protection material'
    res.cell(row=T, column=6).value = 'Protection thickness, δ, mm'
    for i in range(2, len(d) + 2):
        res.cell(row=T+i-1, column=5).value = dt[i - 2]
        res.cell(row=T+i-1, column=6).value = d[i - 2]
    res.cell(row=T, column=7).value = 'Protection μ, rel.un.'
    if Task == 1:
        res.cell(row=T, column=8).value = 'p, rel.un.'
        res.cell(row=T, column=9).value = "b, rel.un."
        res.cell(row=T, column=10).value = "k', rel.un."
        res.cell(row=T, column=11).value = 'k", rel.un.'
        res.cell(row=T, column=12).value = 'μsR, rel.un.'
        res.cell(row=T, column=13).value = "G', rel.un."
        res.cell(row=T, column=14).value = 'G", rel.un.'
    else:
        res.cell(row=T, column=8).value = 'μsh, rel.un.'
        res.cell(row=T, column=9).value = "b, rel.un."
        res.cell(row=T, column=10).value = "a/h, rel.un."
        res.cell(row=T, column=11).value = 'R/h, rel.un.'
        res.cell(row=T, column=13).value = "Z, rel.un."
    res.cell(row=T, column=15).value = 'D, μSv/h'
    res.cell(row=T, column=16).value = 'Dose from tank №' + str(t+1) + ', μSv/h'

    # Calculations
    D = np.zeros(len(E))
    if Task == 1:
        p = a / R
        k1 = (H - h) / R
        k2 = h / R
        res.cell(row=T + 1, column=8).value = p
        res.cell(row=T + 1, column=10).value = k1
        res.cell(row=T + 1, column=11).value = k2
        for e in range(len(E)):
            res.cell(row=T + e + 1, column=1).value = E[e]
            res.cell(row=T + e + 1, column=2).value = q[e]
            res.cell(row=T + e + 1, column=3).value = float(ApproxLin(tab_En, tab_Ga, E[e]) * q[e])
            res.cell(row=T + e + 1, column=7).value = ''
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
            res.cell(row=T + e + 1, column=4).value = float(mus)
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
                res.cell(row=T + e + 1, column=7).value += ' μd' + str(i+1) + ' = ' + str(mud[i])
                print('μd[', i, '] =', mud[i])
                b += (mud[i] * d[i]) / 10
            res.cell(row=T + e + 1, column=9).value = b
            musR = float(mus * R * 100)
            res.cell(row=T + e + 1, column=12).value = musR
            s = np.zeros(len(tab_musR))
            Gpbk = np.zeros((len(tab_k), len(tab_b), len(tab_p)))
            for i in range(len(tab_p)):
                for j in range(len(tab_b)):
                    for n in range(len(tab_k)):
                        for m in range(len(tab_musR)):
                            s[m] = tab_G[m, n, j, i]
                        if musR > tab_musR[-1]:
                            Gpbk[n, j, i] = ApproxPov(tab_musR, s, musR)
                        else:
                            Gpbk[n, j, i] = ApproxLin(tab_musR, s, musR)
            s = np.zeros(len(tab_k))
            Gpb1 = np.zeros((len(tab_b), len(tab_p)))
            Gpb2 = np.zeros((len(tab_b), len(tab_p)))
            for i in range(len(tab_p)):
                for j in range(len(tab_b)):
                    for n in range(len(tab_k)):
                        s[n] = Gpbk[n, j, i]
                    if k1 > tab_k[-1]:
                        Gpb1[j, i] = tab_k[-1]
                    else:
                        Gpb1[j, i] = ApproxLin(tab_k, s, k1)
                    if k2 > tab_k[-1]:
                        Gpb2[j, i] = tab_k[-1]
                    else:
                        Gpb2[j, i] = ApproxLin(tab_k, s, k2)
            s = np.zeros(len(tab_b))
            Gp1 = np.zeros(len(tab_p))
            Gp2 = np.zeros(len(tab_p))
            for i in range(len(tab_p)):
                for j in range(len(tab_b)):
                    s[j] = Gpb1[j, i]
                if b > tab_b[-1]:
                    Gp1[i] = ApproxExp(tab_b, s, b)
                else:
                    Gp1[i] = ApproxLin(tab_b, s, b)
                for j in range(len(tab_b)):
                    s[j] = Gpb2[j, i]
                if b > tab_b[-1]:
                    Gp2[i] = ApproxExp(tab_b, s, b)
                else:
                    Gp2[i] = ApproxLin(tab_b, s, b)
            if p < tab_p[0] or p > tab_p[-1]:
                G1 = float(ApproxPov(tab_p, Gp1, p))
            else:
                G1 = float(ApproxLin(tab_p, Gp1, p))
            res.cell(row=T + e + 1, column=13).value = G1
            if k2 == 0:
                G2 = 0
            else:
                if p < tab_p[0] or p > tab_p[-1]:
                    G2 = float(ApproxPov(tab_p, Gp2, p))
                else:
                    G2 = float(ApproxLin(tab_p, Gp2, p))
            res.cell(row=T + e + 1, column=14).value = G2
            V = H * 2 * np.pi * R ** 2
            D[e] = 2 * A[e] * ApproxLin(tab_En, tab_Ga, E[e]) * q[e] * R / V * (G1 + G2)
            res.cell(row=T + e + 1, column=15).value = D[e]
            print('p = ', p)
            print('b = ', b)
            print('μsR = ', musR)
            print("k' = ", k1, "| k'' = ", k2)
            print("G' = ", G1, "| G'' = ", G2)
            print('E = ', E[e])
            print('q = ', q[e])
            print('Г = ', float(ApproxLin(tab_En, tab_Ga, E[e]) * q[e]))
            print('V = ', V)
            print('D[', e, ']= ', D[e], 'μSv/h')
    else:
        aH = a/H
        RH = R/H
        res.cell(row=T + 1, column=10).value = aH
        res.cell(row=T + 1, column=12).value = RH
        for e in range(len(E)):
            res.cell(row=T + e + 1, column=1).value = E[e]
            res.cell(row=T + e + 1, column=2).value = q[e]
            res.cell(row=T + e + 1, column=3).value = float(ApproxLin(tab_En, tab_Ga, E[e]) * q[e])
            res.cell(row=T + e + 1, column=7).value = ''
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
            res.cell(row=T + e + 1, column=4).value = float(mus)
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
                res.cell(row=T + e + 1, column=7).value += ' μd' + str(i+1) + ' = ' + str(mud[i])
                print('μd[', i, '] =', mud[i])
                b += (mud[i] * d[i]) / 10
            res.cell(row=T + e + 1, column=9).value = b
            musH = float(mus * H * 100)
            res.cell(row=T + e + 1, column=8).value = musH
            s = np.zeros(len(tab_RH))
            Gpbk = np.zeros((len(tab_aH), len(tab_b1), len(tab_musH)))
            for i in range(len(tab_musH)):
                for j in range(len(tab_b1)):
                    for n in range(len(tab_aH)):
                        for m in range(len(tab_RH)):
                            s[m] = tab_Z[m, n, j, i]
                        f = CubicSpline(tab_RH, s, extrapolate=True)
                        if RH > tab_RH[-1] or f(RH) < 0:
                            Gpbk[n, j, i] = ApproxLog(tab_RH, s, RH)
                        else:
                            Gpbk[n, j, i] = f(RH)
            s = np.zeros(len(tab_aH))
            Gpb = np.zeros((len(tab_b1), len(tab_musH)))
            for i in range(len(tab_musH)):
                for j in range(len(tab_b1)):
                    for n in range(len(tab_aH)):
                        s[n] = Gpbk[n, j, i]
                    f = CubicSpline(tab_aH, s, extrapolate=True)
                    if aH > tab_aH[-1] or f(aH) < 0:
                        Gpb[j, i] = ApproxExp(tab_aH, s, aH)
                    else:
                        Gpb[j, i] = f(aH)
            s = np.zeros(len(tab_b1))
            Gp = np.zeros(len(tab_musH))
            for i in range(len(tab_musH)):
                for j in range(len(tab_b1)):
                    s[j] = Gpb[j, i]
                if b > tab_b1[-1]:
                    Gp[i] = ApproxExp(tab_b1, s, b)
                else:
                    Gp[i] = ApproxLin(tab_b1, s, b)
            f = CubicSpline(tab_musH[0:(len(tab_musH)-1)], Gp[0:(len(tab_musH)-1)], extrapolate=True)
            if f(musH) > tab_musH[-2] or f(musH) < 0:
                Z = float(Gp[-1]*np.exp(np.log(Gp[-2] / Gp[-1]) / tab_musH[-2]*musH))
            else:
                Z = float(f(musH))
            res.cell(row=T + e + 1, column=13).value = Z
            V = H * 2 * np.pi * R ** 2
            D[e] = 2 * A[e] * ApproxLin(tab_En, tab_Ga, E[e]) * q[e] * R / V * Z
            res.cell(row=T + e + 1, column=15).value = D[e]
            print('μsH = ', musH)
            print('b = ', b)
            print('R/h = ', RH)
            print("a/h = ", aH)
            print("Z = ", Z)
            print('E = ', E[e])
            print('q = ', q[e])
            print('Г = ', float(ApproxLin(tab_En, tab_Ga, E[e]) * q[e]))
            print('V = ', V)
    print('Total dose rate, D = ', np.sum(D), 'μSv/h')
    res.cell(row=T, column=17).value = np.sum(D)
    T += inp.max_row + 1
results.save("Results.xlsx")