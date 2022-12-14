import numpy as np


tab_En = np.array([0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1, 1.5, 2,
                  3, 4, 5, 6, 8, 10])
tab_mub = np.array([4.87, 1.32,	0.533, 0.154, 0.0701, 0.0431, 0.0328, 0.0264, 0.0256, 0.0275, 0.0294, 0.0317, 0.0325,
                    0.0328, 0.0326, 0.0318, 0.0308, 0.0282, 0.0259, 0.0227, 0.0207, 0.0193, 0.0183, 0.0169, 0.016])
tab_Ga = np.zeros(len(tab_En))
for i in range(len(tab_En)):
    tab_Ga [i] = 1275 * tab_En[i]*tab_mub[i]*3600*1e-12