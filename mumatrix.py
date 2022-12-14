import numpy as np

tab_p = np.array([1.25, 1.5, 3, 5, 10])  # x 4 (cubic/power approximation)
tab_b = np.array([0, 1, 2, 4, 7, 10, 20])  # y 3 (exponential approximation)
tab_k = np.array([0, 1, 3, 5, 10])  # k 2 (cubic approximation)
tab_musR = np.array([0, 1, 2, 3, 5, 10])  # z 1 (power approximation)

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

# tab_G
mukz = np.zeros((7, 5))
st0 = [1.084, 7.366e-1, 1.777e-1, 6.323e-2, 1.573e-2]
st1 = [2.972e-1, 2.265e-1, 6.301e-2, 2.298e-2, 5.571e-3]
st2 = [8.707e-2, 7.130e-2, 2.239e-2, 8.355e-3, 2.117e-3]
st3 = [8.331e-3, 7.434e-3, 2.831e-3, 1.104e-3, 2.848e-4]
st4 = [2.850e-4, 2.724e-4, 1.279e-4, 5.307e-5, 1.406e-5]
st5 = [1.071e-5, 1.064e-5, 5.809e-6, 2.552e-6, 6.937e-7]
st6 = [2.608e-10, 2.738e-10, 2.007e-10, 1.037e-10, 3.060e-11]
muk0 = np.array([st0, st1, st2, st3, st4, st5, st6])

st0 = [1.693, 1.275, 4.247e-1, 1.724e-1, 4.599e-2]
st1 = [3.842e-1, 3.222e-1, 1.356e-1, 5.978e-2, 1.665e-2]
st2 = [1.041e-1, 9.171e-2, 4.406e-2, 2.078e-2, 6.031e-3]
st3 = [9.244e-3, 8.635e-3, 4.842e-3, 2.532e-3, 7.915e-4]
st4 = [3.012e-4, 2.958e-4, 1.893e-4, 1.095e-4, 3.767e-5]
st5 = [1.107e-5, 1.119e-4, 7.809e-6, 4.820e-6, 1.796e-6]
st6 = [2.628e-10, 2.775e-10, 2.298e-10, 1.589e-10, 7.128e-11]
muk1 = np.array([st0, st1, st2, st3, st4, st5, st6])

st0 = [1.75, 1.37, 5.672e-1, 2.470e-1, 7.320e-2]
st1 = [3.927e-1, 3.336e-1, 1.594e-1, 8.107e-2, 2.587e-2]
st2 = [1.049e-1, 9.286e-2, 4.874e-2, 2.670e-2, 9.165e-3]
st3 = [9.254e-3, 8.654e-3, 5.046e-3, 2.999e-3, 1.154e-3]
st4 = [3.012e-4, 2.959e-4, 1.916e-4, 1.204e-4, 5.200e-5]
st5 = [1.107e-5, 1.119e-5, 7.840e-6, 5.086e-6, 2.365e-6]
st6 = [2.628e-10, 2.775e-10, 2.299e-10, 1.604e-10, 8.379e-11]
muk2 = np.array([st0, st1, st2, st3, st4, st5, st6])

st0 = [1.700, 1.420, 6.842e-1, 3.504e-1, 1.237e-1]
st1 = [3.942e-1, 3.359e-1, 1.702e-1, 9.881e-2, 4.046e-2]
st2 = [1.049e-1, 9.296e-2, 4.985e-2, 3.001e-2, 1.340e-2]
st3 = [9.254e-3, 8.654e-3, 5.063e-3, 3.132e-3, 1.519e-3]
st4 = [3.012e-4, 2.959e-4, 1.916e-4, 1.217e-4, 6.170e-5]
st5 = [1.107e-5, 1.119e-5, 7.840e-6, 5.101e-6, 2.635e-6]
st6 = [2.628e-10, 2.775e-10, 2.299e-10, 1.604e-10, 8.601e-11]
muk3 = np.array([st0, st1, st2, st3, st4, st5, st6])

musR0 = np.array([mukz, muk0, muk1, muk2, muk3])

st0 = [5.600e-1, 4.000e-1, 9.400e-2, 3.250e-2, 8.000e-3]
st1 = [1.494e-1, 1.173e-1, 3.262e-2, 1.166e-2, 2.873e-3]
st2 = [4.210e-2, 3.597e-2, 1.154e-2, 4.238e-3, 1.054e-3]
st3 = [3.813e-3, 3.607e-3, 1.446e-3, 5.580e-4, 1.417e-4]
st4 = [1.234e-4, 1.270e-4, 6.456e-5, 2.673e-5, 6.989e-6]
st5 = [4.467e-6, 4.826e-6, 2.899e-6, 1.281e-6, 3.447e-7]
st6 = [1.010e-10, 1.171e-10, 9.706e-11, 5.147e-11, 1.517e-11]
muk0 = np.array([st0, st1, st2, st3, st4, st5, st6])

st0 = [6.800e-1, 5.400e-1, 2.030e-1, 8.400e-2, 2.280e-2]
st1 = [1.668e-1, 1.431e-1, 6.457e-2, 2.297e-2, 8.219e-3]
st2 = [4.522e-2, 4.093e-2, 2.085e-2, 1.014e-2, 2.974e-3]
st3 = [3.966e-3, 3.864e-3, 2.277e-3, 1.229e-3, 3.898e-4]
st4 = [1.260e-4, 1.315e-4, 8.869e-5, 5.281e-5, 1.852e-5]
st5 = [4.521e-6, 4.926e-6, 3.653e-6, 2.313e-6, 8.812e-7]
st6 = [1.013e-10, 1.177e-10, 1.068e-10, 7.564e-11, 3.479e-11]
muk1 = np.array([st0, st1, st2, st3, st4, st5, st6])

st0 = [6.800e-1, 5.600e-1, 2.400e-1, 1.170e-1, 3.550e-2]
st1 = [1.671e-1, 1.438e-1, 7.161e-2, 3.802e-2, 1.257e-2]
st2 = [4.525e-2, 4.099e-2, 2.210e-2, 1.252e-2, 4.449e-3]
st3 = [3.967e-3, 3.865e-3, 2.322e-3, 1.407e-3, 5.593e-4]
st4 = [1.260e-4, 1.315e-4, 8.911e-5, 5.669e-5, 2.515e-5]
st5 = [4.521e-6, 4.926e-6, 3.658e-6, 2.403e-6, 1.142e-6]
st6 = [1.013e-10, 1.177e-10, 1.068e-10, 7.605e-11, 4.035e-11]
muk2 = np.array([st0, st1, st2, st3, st4, st5, st6])

st0 = [6.900e-1, 5.700e-1, 2.630e-1, 1.400e-1, 5.700e-2]
st1 = [1.671e-1, 1.439e-1, 7.335e-2, 4.343e-2, 1.877e-2]
st2 = [4.525e-2, 4.099e-2, 2.225e-2, 1.348e-2, 6.237e-3]
st3 = [3.967e-3, 3.865e-3, 2.324e-3, 1.443e-3, 7.115e-4]
st4 = [1.260e-4, 1.315e-4, 8.912e-5, 5.701e-5, 2.912e-5]
st5 = [4.521e-6, 4.926e-6, 3.658e-6, 2.406e-6, 1.251e-6]
st6 = [1.013e-10, 1.177e-10, 1.068e-10, 7.606e-11, 4.120e-11]
muk3 = np.array([st0, st1, st2, st3, st4, st5, st6])

musR1 = np.array([mukz, muk0, muk1, muk2, muk3])

st0 = [3.550e-1, 2.541e-1, 5.780e-2, 2.008e-2, 4.832e-3]
st1 = [9.157e-2, 7.382e-2, 2.060e-2, 7.239e-3, 1.760e-3]
st2 = [2.512e-2, 2.225e-2, 7.265e-3, 2.626e-3, 6.452e-4]
st3 = [2.183e-3, 2.174e-3, 9.055e-4, 3.457e-4, 8.674e-5]
st4 = [7.501e-5, 7.458e-5, 4.008e-5, 1.652e-5, 4.276e-6]
st5 = [2.661e-6, 2.779e-6, 1.786e-6, 7.901e-7, 2.108e-7]
st6 = [5.704e-11, 6.421e-11, 5.859e-11, 3.153e-11, 9.264e-12]
muk0 = np.array([st0, st1, st2, st3, st4, st5, st6])

st0 = [4.000e-1, 3.349e-1, 1.256e-1, 5.207e-2, 1.395e-2]
st1 = [9.600e-2, 8.281e-2, 3.888e-2, 1.781e-2, 5.010e-3]
st2 = [2.584e-2, 2.391e-2, 1.250e-2, 6.160e-3, 1.812e-2]
st3 = [2.214e-3, 2.245e-3, 1.357e-3, 7.432e-4, 2.373e-4]
st4 = [7.548e-5, 7.563e-5, 5.269e-5, 3.178e-5, 1.126e-5]
st5 = [2.670e-6, 2.800e-6, 2.166e-6, 1.388e-6, 5.350e-7]
st6 = [5.707e-11, 6.431e-11, 6.292e-11, 4.511e-11, 2.105e-11]
muk1 = np.array([st0, st1, st2, st3, st4, st5, st6])

st0 = [4.000e-1, 3.200e-1, 1.420e-1, 7.000e-2, 2.140e-2]
st1 = [9.627e-2, 8.380e-2, 4.215e-2, 2.267e-2, 7.605e-3]
st2 = [2.584e-1, 2.391e-2, 1.303e-2, 7.452e-3, 2.689e-3]
st3 = [2.214e-3, 2.245e-3, 1.374e-3, 8.370e-4, 3.379e-4]
st4 = [7.548e-5, 7.563e-5, 5.281e-5, 3.372e-5, 1.515e-5]
st5 = [2.670e-6, 2.800e-6, 2.167e-6, 1.430e-6, 6.870e-7]
st6 = [5.707e-11, 6.431e-11, 6.292e-11, 4.527e-11, 2.422e-11]
muk2 = np.array([st0, st1, st2, st3, st4, st5, st6])

st0 = [4.000e-1, 3.200e-1, 1.460e-1, 8.762e-2, 3.42e-2]
st1 = [9.627e-2, 8.380e-2, 4.280e-2, 2.534e-2, 1.1e-2]
st2 = [2.584e-3, 2.391e-2, 1.308e-2, 7.910e-3, 3.7e-3]
st3 = [2.214e-3, 2.245e-3, 1.374e-3, 8.526e-4, 4.2e-4]
st4 = [7.548e-5, 7.563e-5, 5.281e-5, 3.385e-5, 1.8e-5]
st5 = [2.670e-6, 2.800e-6, 2.167e-6, 1.431e-6, 7.0e-7]
st6 = [5.707e-11, 6.431e-11, 6.292e-11, 4.527e-11, 2.5e-11]
muk3 = np.array([st0, st1, st2, st3, st4, st5, st6])

musR2 = np.array([mukz, muk0, muk1, muk2, muk3])

st0 = [2.600e-1, 1.800e-1, 3.200e-1, 1.420e-2, 3.400e-3]
st1 = [6.386e-2, 5.263e-2, 1.476e-2, 5.154e-3, 1.240e-3]
st2 = [1.808e-2, 1.570e-2, 5.195e-3, 1.869e-3, 4.548e-4]
st3 = [1.577e-3, 1.510e-3, 6.452e-4, 2.457e-4, 6.113e-5]
st4 = [4.902e-5, 5.098e-5, 2.842e-5, 1.173e-5, 3.013e-6]
st5 = [1.710e-6, 1.972e-6, 1.261e-6, 5.599e-7, 1.485e-7]
st6 = [3.450e-11, 4.657e-11, 4.085e-11, 2.224e-11, 6.519e-12]
muk0 = np.array([st0, st1, st2, st3, st4, st5, st6])

st0 = [2.600e-1, 2.200e-1, 8.500e-1, 3.600e-2, 9.800e-3]
st1 = [6.549e-2, 2.761e-2, 2.711e-2, 1.255e-2, 3.524e-3]
st2 = [1.831e-2, 1.643e-2, 8.690e-3, 4.332e-3, 1.274e-3]
st3 = [1.585e-3, 1.535e-3, 9.405e-4, 5.212e-4, 1.667e-4]
st4 = [4.912e-5, 5.127e-5, 3.643e-5, 2.222e-5, 7.903e-6]
st5 = [1.711e-6, 1.978e-6, 1.495e-6, 9.678e-7, 3.753e-7]
st6 = [3.451e-11, 4.659e-11, 4.329e-11, 3.132e-11, 1.473e-11]
muk1 = np.array([st0, st1, st2, st3, st4, st5, st6])

st0 = [2.700e-1, 2.200e-1, 9.800e-2, 4.850e-2, 1.520e-2]
st1 = [6.549e-2, 5.764e-2, 2.910e-2, 1.582e-2, 5.331e-3]
st2 = [1.831e-2, 1.643e-2, 8.987e-3, 5.193e-3, 1.884e-3]
st3 = [1.585e-3, 1.535e-3, 9.490e-4, 5.823e-4, 2.362e-4]
st4 = [4.912e-5, 5.127e-5, 3.648e-5, 2.344e-5, 1.059e-5]
st5 = [1.711e-6, 1.978e-6, 1.496e-6, 9.935e-7, 4.797e-7]
st6 = [3.451e-11, 4.659e-11, 4.329e-11, 3.141e-11, 1.689e-11]
muk2 = np.array([st0, st1, st2, st3, st4, st5, st6])

st0 = [2.700e-1, 2.200e-1, 1.030e-1, 5.800e-2, 2.350e-2]
st1 = [6.549e-2, 5.764e-2, 2.947e-2, 1.755e-2, 7.741e-3]
st2 = [1.731e-2, 1.643e-2, 9.021e-3, 5.482e-3, 2.573e-3]
st3 = [1.585e-3, 1.535e-3, 9.491e-4, 5.917e-4, 2.938e-4]
st4 = [4.912e-5, 5.127e-5, 3.648e-5, 2.351e-5, 1.206e-5]
st5 = [1.711e-6, 1.978e-6, 1.496e-6, 9.941e-7, 5.189e-7]
st6 = [3.451e-11, 4.659e-11, 4.329e-11, 3.141e-11, 1.716e-11]
muk3 = np.array([st0, st1, st2, st3, st4, st5, st6])

musR3 = np.array([mukz, muk0, muk1, muk2, muk3])

st0 = [1.500e-1, 1.150e-1, 2.620e-2, 8.700e-3, 2.100e-3]
st1 = [4.024e-2, 3.282e-2, 9.295e-3, 3.213e-3, 7.645e-4]
st2 = [1.084e-2, 9.686e-3, 3.265e-3, 1.164e-3, 2.803e-4]
st3 = [9.219e-4, 9.181e-4, 4.041e-4, 1.530e-4, 3.767e-5]
st4 = [2.765e-5, 3.160e-5, 1.771e-5, 7.292e-6, 1.856e-6]
st5 = [9.292e-7, 1.172e-6, 7.825e-7, 3.478e-7, 9.147e-8]
st6 = [1.667e-11, 2.694e-11, 2.507e-11, 1.376e-11, 4.013e-12]
muk0 = np.array([st0, st1, st2, st3, st4, st5, st6])

st0 = [1.550e-1, 1.300e-1, 5.300e-2, 2.250e-2, 6.000e-3]
st1 = [4.062e-2, 3.488e-2, 1.660e-2, 7.741e-3, 2.167e-3]
st2 = [1.087e-2, 9.935e-3, 5.307e-3, 2.667e-3, 7.835e-4]
st3 = [9.225e-4, 9.238e-4, 5.727e-4, 3.200e-4, 1.024e-4]
st4 = [2.766e-5, 3.164e-5, 2.215e-5, 1.360e-5, 4.852e-6]
st5 = [9.293e-7, 1.172e-6, 9.086e-7, 5.911e-7, 2.302e-7]
st6 = [1.667e-11, 2.694e-11, 2.625e-11, 1.907e-11, 9.014e-12]
muk1 = np.array([st0, st1, st2, st3, st4, st5, st6])

st0 = [1.550e-1, 1.320e-1, 5.900e-2, 3.000e-2, 9.400e-3]
st1 = [4.062e-2, 3.488e-2, 1.767e-2, 9.667e-3, 3.268e-3]
st2 = [1.087e-2, 9.935e-3, 5.463e-3, 3.169e-3, 1.154e-3]
st3 = [9.225e-4, 9.238e-4, 5.766e-4, 3.548e-4, 1.446e-4]
st4 = [2.766e-5, 3.164e-5, 2.217e-5, 1.427e-5, 6.473e-6]
st5 = [9.293e-7, 1.172e-6, 9.087e-7, 6.047e-7, 2.929e-7]
st6 = [1.667e-11, 2.694e-11, 2.625e-11, 1.911e-11, 1.028e-11]
muk2 = np.array([st0, st1, st2, st3, st4, st5, st6])

st0 = [1.550e-1, 1.320e-1, 6.600e-2, 3.550e-2, 1.460e-2]
st1 = [4.062e-2, 3.488e-2, 1.785e-2, 1.065e-2, 4.712e-3]
st2 = [1.087e-2, 9.935e-3, 5.474e-3, 3.329e-3, 1.565e-3]
st3 = [9.225e-4, 9.238e-4, 5.766e-4, 3.598e-4, 1.787e-4]
st4 = [2.767e-5, 3.164e-5, 2.217e-5, 1.430e-5, 7.333e-6]
st5 = [9.293e-7, 1.172e-6, 9.087e-7, 6.050e-7, 3.157e-7]
st6 = [1.667e-11, 2.694e-11, 2.625e-11, 1.911e-11, 1.044e-11]
muk3 = np.array([st0, st1, st2, st3, st4, st5, st6])

musR4 = np.array([mukz, muk0, muk1, muk2, muk3])

st0 = [8.000e-2, 6.000e-2, 1.450e-2, 4.500e-3, 1.060e-3]
st1 = [1.982e-2, 1.675e-2, 4.711e-3, 1.620e-3, 3.816e-4]
st2 = [5.272e-3, 4.921e-3, 1.653e-3, 5.872e-4, 1.399e-4]
st3 = [4.342e-4, 4.665e-4, 2.042e-4, 7.713e-5, 1.880e-5]
st4 = [1.222e-5, 1.565e-5, 8.930e-6, 3.674e-6, 9.265e-7]
st5 = [3.816e-7, 5.755e-7, 3.937e-7, 1.751e-7, 4.566e-7]
st6 = [5.302e-12, 1.262e-11, 1.256e-11, 6.920e-12, 2.003e-12]
muk0 = np.array([st0, st1, st2, st3, st4, st5, st6])

st0 = [8.400e-2, 6.600e-2, 2.650e-2, 1.120e-2, 3.000e-3]
st1 = [1.988e-2, 1.745e-2, 8.157e-3, 3.858e-3, 1.079e-3]
st2 = [5.274e-3, 4.990e-3, 2.607e-3, 1.328e-3, 3.902e-4]
st3 = [4.342e-4, 4.674e-4, 2.816e-4, 1.590e-4, 5.099e-5]
st4 = [1.222e-5, 1.565e-5, 1.091e-5, 6.745e-6, 2.414e-6]
st5 = [3.816e-7, 5.755e-7, 4.488e-7, 2.929e-7, 1.145e-7]
st6 = [5.302e-12, 1.262e-11, 1.303e-11, 9.450e-12, 4.475e-12]
muk1 = np.array([st0, st1, st2, st3, st4, st5, st6])

st0 = [8.400e-2, 6.600e-2, 3.200e-2, 1.450e-2, 4.600e-3]
st1 = [1.988e-2, 1.746e-2, 8.825e-3, 4.767e-3, 1.622e-3]
st2 = [5.274e-3, 4.990e-3, 2.668e-3, 1.562e-3, 5.727e-4]
st3 = [4.342e-4, 4.674e-4, 2.830e-4, 1.749e-4, 7.168e-5]
st4 = [1.222e-5, 1.565e-5, 1.092e-5, 7.044e-6, 3.206e-6]
st5 = [3.816e-7, 5.755e-7, 4.488e-7, 2.988e-7, 1.499e-7]
st6 = [5.302e-12, 1.262e-11, 1.303e-11, 9.466e-12, 5.083e-12]
muk2 = np.array([st0, st1, st2, st3, st4, st5, st6])

st0 = [8.400e-2, 6.600e-2, 3.400e-2, 1.860e-2, 7.100e-3]
st1 = [1.988e-2, 1.746e-2, 8.905e-3, 5.197e-3, 2.320e-3]
st2 = [5.274e-3, 4.990e-3, 2.672e-3, 1.632e-3, 7.706e-4]
st3 = [4.342e-4, 4.674e-4, 2.830e-4, 1.771e-4, 8.801e-5]
st4 = [1.222e-5, 1.565e-5, 1.092e-5, 7.0458e-6, 3.614e-6]
st5 = [3.816e-7, 5.755e-7, 4.488e-7, 2.990e-7, 1.557e-7]
st6 = [5.302e-12, 1.262e-11, 1.303e-11, 9.466e-12, 5.157e-12]
muk3 = np.array([st0, st1, st2, st3, st4, st5, st6])

musR5 = np.array([mukz, muk0, muk1, muk2, muk3])

tab_G = np.array([musR0, musR1, musR2, musR3, musR4, musR5])

# tab_Z
tab_musH = np.array([0.1, 0.3, 0.5, 1, 2, 3, 5, np.inf])  # cubic/exp approximation
tab_b1 = np.array([0, 0.2, 0.5, 1, 2, 5, 10])             # exp
tab_aH = np.array([0, 0.1, 0.5, 1])                       # cubic/exp
tab_RH = np.array([0.1, 0.5, 1, 2])                       # cub/log

st0 = [1.504e-2, 4.413e-2, 7.198e-2, 1.368e-1, 2.516e-1, 3.437e-1, 4.988e-1, 1]
st1 = [1.504e-2, 4.413e-2, 7.198e-2, 1.368e-1, 2.516e-1, 3.437e-1, 4.988e-1, 1]
st2 = [1.504e-2, 4.413e-2, 7.198e-2, 1.368e-1, 2.516e-1, 3.437e-1, 4.988e-1, 1]
st3 = [1.504e-2, 4.413e-2, 7.198e-2, 1.368e-1, 2.516e-1, 3.437e-1, 4.988e-1, 1]
st4 = [1.504e-2, 4.413e-2, 7.198e-2, 1.368e-1, 2.516e-1, 3.437e-1, 4.988e-1, 1]
st5 = [1.504e-2, 4.413e-2, 7.198e-2, 1.368e-1, 2.516e-1, 3.437e-1, 4.988e-1, 1]
st6 = [1.504e-2, 4.413e-2, 7.198e-2, 1.368e-1, 2.516e-1, 3.437e-1, 4.988e-1, 1]
muk0 = np.array([st0, st1, st2, st3, st4, st5, st6])

st0 = [3.864e-3, 1.119e-2, 1.803e-2, 3.335e-2, 5.830e-2, 7.808e-2, 1.081e-1, 2.929e-1]
st1 = [3.118e-3, 9.027e-3, 1.454e-2, 2.688e-2, 4.692e-2, 6.276e-2, 8.677e-2, 2.317e-1]
st2 = [2.261e-3, 6.543e-3, 1.054e-2, 1.945e-2, 3.389e-2, 4.526e-2, 6.241e-2, 1.623e-1]
st3 = [1.325e-3, 3.832e-3, 6.166e-3, 1.136e-2, 1.973e-2, 2.628e-2, 3.609e-2, 9.027e-2]
st4 = [4.572e-4, 1.320e-3, 2.120e-3, 3.894e-3, 6.721e-3, 8.905e-3, 1.213e-2, 2.825e-2]
st5 = [1.930e-5, 5.547e-5, 8.875e-5, 1.614e-4, 2.739e-4, 3.578e-4, 4.768e-4, 9.290e-4]
st6 = [1.055e-7, 3.016e-7, 4.799e-7, 8.619e-7, 1.430e-6, 1.832e-6, 2.366e-6, 3.799e-6]
muk1 = np.array([st0, st1, st2, st3, st4, st5, st6])

st0 = [6.394e-4, 1.802e-3, 2.829e-3, 4.927e-3, 7.777e-3, 9.607e-3, 1.185e-2, 1.942e-2]
st1 = [5.230e-4, 1.474e-3, 2.314e-3, 4.029e-3, 6.360e-3, 7.856e-3, 9.685e-3, 1.590e-2]
st2 = [3.869e-4, 1.090e-3, 1.712e-3, 2.980e-3, 4.703e-3, 5.809e-3, 7.160e-3, 1.145e-2]
st3 = [2.341e-4, 6.597e-4, 1.035e-3, 1.803e-3, 2.844e-3, 3.512e-3, 4.328e-3, 7.069e-3]
st4 = [8.570e-5, 2.415e-4, 2.790e-4, 6.597e-4, 1.040e-3, 1.284e-3, 1.581e-3, 2.578e-3]
st5 = [4.206e-6, 1.185e-5, 1.859e-5, 3.232e-5, 5.090e-5, 6.275e-5, 7.713e-5, 1.246e-4]
st6 = [2.768e-8, 7.739e-8, 1.222e-7, 2.122e-7, 3.333e-7, 4.101e-7, 5.025e-7, 8.004e-7]
muk2 = np.array([st0, st1, st2, st3, st4, st5, st6])

st0 = [2.399e-4, 6.682e-4, 1.037e-3, 1.757e-3, 2.644e-3, 3.144e-3, 3.672e-3, 4.963e-3]
st1 = [1.963e-4, 5.469e-4, 8.487e-4, 1.438e-3, 2.164e-3, 2.573e-3, 3.005e-3, 4.070e-3]
st2 = [1.454e-4, 4.050e-4, 6.285e-4, 1.065e-3, 1.602e-3, 1.905e-3, 2.225e-3, 2.725e-3]
st3 = [8.813e-5, 2.454e-4, 3.809e-4, 6.453e-4, 9.712e-4, 1.154e-3, 1.348e-3, 1.820e-3]
st4 = [3.237e-5, 9.016e-5, 1.399e-4, 2.370e-4, 3.567e-4, 4.240e-4, 4.949e-3, 6.687e-4]
st5 = [1.605e-6, 4.468e-6, 6.934e-6, 1.174e-5, 1.767e-5, 2.099e-5, 2.450e-5, 3.303e-5]
st6 = [1.073e-8, 2.988e-8, 4.637e-8, 7.851e-8, 1.180e-7, 1.402e-7, 1.635e-7, 2.198e-7]
muk3 = np.array([st0, st1, st2, st3, st4, st5, st6])

musR0 = np.array([muk0, muk1, muk2, muk3])

st0 = [6.418e-2, 1.796e-1, 2.798e-1, 4.770e-1, 7.185e-1, 8.450e-1, 9.509e-1, 1]
st1 = [6.418e-2, 1.796e-1, 2.798e-1, 4.770e-1, 7.185e-1, 8.450e-1, 9.509e-1, 1]
st2 = [6.418e-2, 1.796e-1, 2.798e-1, 4.770e-1, 7.185e-1, 8.450e-1, 9.509e-1, 1]
st3 = [6.418e-2, 1.796e-1, 2.798e-1, 4.770e-1, 7.185e-1, 8.450e-1, 9.509e-1, 1]
st4 = [6.418e-2, 1.796e-1, 2.798e-1, 4.770e-1, 7.185e-1, 8.450e-1, 9.509e-1, 1]
st5 = [6.418e-2, 1.796e-1, 2.798e-1, 4.770e-1, 7.185e-1, 8.450e-1, 9.509e-1, 1]
st6 = [6.418e-2, 1.796e-1, 2.798e-1, 4.770e-1, 7.185e-1, 8.450e-1, 9.509e-1, 1]
muk0 = np.array([st0, st1, st2, st3, st4, st5, st6])

st0 = [4.000e-2, 1.123e-1, 1.757e-1, 3.030e-1, 4.677e-1, 5.646e-1, 6.655e-1, 8.039e-1]
st1 = [2.960e-2, 8.291e-2, 1.293e-1, 2.216e-1, 3.384e-1, 4.064e-1, 4.707e-1, 5.463e-1]
st2 = [1.928e-2, 5.383e-2, 8.368e-2, 1.422e-1, 2.142e-1, 2.534e-1, 2.899e-1, 3.229e-1]
st3 = [9.843e-3, 2.753e-2, 4.235e-2, 7.125e-2, 1.054e-1, 1.230e-1, 1.381e-1, 1.483e-1]
st4 = [2.803e-3, 7.744e-3, 1.192e-2, 1.979e-2, 2.863e-2, 3.280e-2, 3.599e-2, 3.754e-2]
st5 = [8.616e-5, 2.360e-4, 3.602e-4, 5.864e-4, 8.209e-4, 9.179e-4, 9.785e-4, 9.966e-4]
st6 = [3.588e-7, 9.775e-7, 1.484e-6, 2.388e-6, 3.277e-6, 3.614e-6, 3.794e-6, 3.831e-6]
muk1 = np.array([st0, st1, st2, st3, st4, st5, st6])

st0 = [1.326e-2, 3.707e-2, 5.774e-2, 9.866e-2, 1.507e-1, 1.812e-1, 2.147e-1, 2.929e-1]
st1 = [1.065e-2, 2.975e-2, 4.631e-2, 7.904e-2, 1.205e-1, 1.447e-1, 1.710e-1, 2.317e-1]
st2 = [7.663e-3, 2.139e-2, 3.328e-2, 5.671e-2, 8.621e-2, 1.033e-1, 1.216e-1, 1.623e-1]
st3 = [4.434e-3, 1.237e-2, 1.922e-2, 3.266e-2, 4.941e-2, 5.896e-2, 6.905e-2, 9.027e-2]
st4 = [1.492e-3, 4.151e-3, 6.438e-3, 1.089e-2, 1.633e-2, 1.934e-2, 2.241e-2, 2.825e-2]
st5 = [5.863e-5, 1.623e-4, 2.503e-4, 4.180e-4, 6.128e-4, 7.123e-4, 8.032e-4, 9.290e-4]
st6 = [2.883e-7, 7.931e-7, 1.216e-6, 2.000e-6, 2.856e-6, 3.247e-6, 3.549e-6, 3.799e-6]
muk2 = np.array([st0, st1, st2, st3, st4, st5, st6])

st0 = [5.610e-3, 1.558e-2, 2.410e-2, 4.056e-2, 6.033e-2, 7.107e-2, 8.192e-2, 1.056e-1]
st1 = [4.562e-3, 1.266e-2, 1.959e-2, 3.296e-2, 4.900e-2, 5.770e-2, 6.646e-2, 8.551e-2]
st2 = [3.345e-3, 9.284e-3, 1.436e-2, 2.414e-2, 3.587e-2, 4.220e-2, 4.857e-2, 6.223e-2]
st3 = [1.994e-3, 5.533e-3, 8.557e-3, 1.438e-2, 2.133e-2, 2.507e-2, 2.880e-2, 3.662e-2]
st4 = [7.093e-4, 1.967e-3, 3.040e-3, 5.099e-3, 7.545e-3, 8.848e-3, 1.013e-2, 1.280e-2]
st5 = [3.203e-5, 8.865e-5, 1.367e-4, 2.284e-4, 3.354e-4, 3.909e-4, 4.436e-4, 5.434e-4]
st6 = [1.854e-7, 5.118e-7, 7.874e-7, 1.307e-6, 1.897e-6, 2.190e-6, 2.451e-6, 2.874e-6]
muk3 = np.array([st0, st1, st2, st3, st4, st5, st6])

musR1 = np.array([muk0, muk1, muk2, muk3])

st0 = [1.070e-1, 2.875e-1, 4.312e-1, 6.754e-1, 8.932e-1, 9.645e-1, 9.960e-1, 1]
st1 = [1.070e-1, 2.875e-1, 4.312e-1, 6.754e-1, 8.932e-1, 9.645e-1, 9.960e-1, 1]
st2 = [1.070e-1, 2.875e-1, 4.312e-1, 6.754e-1, 8.932e-1, 9.645e-1, 9.960e-1, 1]
st3 = [1.070e-1, 2.875e-1, 4.312e-1, 6.754e-1, 8.932e-1, 9.645e-1, 9.960e-1, 1]
st4 = [1.070e-1, 2.875e-1, 4.312e-1, 6.754e-1, 8.932e-1, 9.645e-1, 9.960e-1, 1]
st5 = [1.070e-1, 2.875e-1, 4.312e-1, 6.754e-1, 8.932e-1, 9.645e-1, 9.960e-1, 1]
st6 = [1.070e-1, 2.875e-1, 4.312e-1, 6.754e-1, 8.932e-1, 9.645e-1, 9.960e-1, 1]
muk0 = np.array([st0, st1, st2, st3, st4, st5, st6])

st0 = [7.928e-2, 2.156e-1, 3.272e-1, 5.263e-1, 7.265e-1, 8.089e-1, 8.638e-1, 9.005e-1]
st1 = [5.440e-2, 1.474e-1, 2.227e-1, 3.550e-1, 4.829e-1, 5.313e-1, 5.596e-1, 5.709e-1]
st2 = [3.294e-2, 8.892e-2, 1.339e-1, 2.118e-1, 2.845e-1, 3.103e-1, 3.236e-1, 3.265e-1]
st3 = [1.556e-2, 4.189e-2, 6.293e-2, 9.892e-2, 1.315e-1, 1.425e-1, 1.477e-1, 1.485e-1]
st4 = [4.022e-3, 1.081e-2, 1.621e-2, 2.538e-2, 3.354e-2, 3.621e-2, 3.738e-2, 3.754e-2]
st5 = [1.057e-4, 2.844e-4, 4.271e-4, 6.704e-4, 8.888e-4, 9.607e-4, 9.925e-4, 9.966e-4]
st6 = [3.928e-7, 1.061e-6, 1.599e-6, 2.529e-6, 3.386e-6, 3.678e-6, 3.812e-6, 3.831e-6]
muk1 = np.array([st0, st1, st2, st3, st4, st5, st6])

st0 = [3.754e-2, 1.032e-1, 1.583e-1, 2.611e-1, 3.758e-1, 4.320e-1, 4.814e-1, 5.528e-1]
st1 = [2.906e-2, 7.984e-2, 1.223e-1, 2.011e-1, 2.881e-1, 3.299e-1, 3.657e-1, 4.145e-1]
st2 = [1.988e-2, 5.452e-2, 8.339e-2, 1.366e-1, 1.944e-1, 2.214e-1, 2.436e-1, 2.706e-1]
st3 = [1.065e-2, 2.916e-2, 4.449e-2, 7.284e-2, 1.021e-1, 1.154e-1, 1.256e-1, 1.361e-1]
st4 = [3.155e-3, 8.603e-3, 1.308e-2, 2.113e-2, 2.933e-2, 3.273e-2, 3.506e-2, 3.672e-2]
st5 = [9.649e-5, 2.617e-4, 3.958e-4, 6.314e-4, 8.570e-4, 9.398e-4, 9.840e-4, 9.961e-4]
st6 = [3.835e-7, 1.039e-6, 1.569e-6, 2.493e-6, 3.360e-6, 3.663e-6, 3.808e-6, 3.831e-6]
muk2 = np.array([st0, st1, st2, st3, st4, st5, st6])

st0 = [1.898e-2, 5.228e-2, 8.029e-2, 1.328e-1, 1.919e-1, 2.212e-1, 2.477e-1, 2.929e-1]
st1 = [1.517e-2, 4.177e-2, 6.413e-2, 1.059e-1, 1.529e-1, 1.760e-1, 1.967e-1, 2.317e-1]
st2 = [1.085e-2, 2.985e-2, 4.580e-2, 7.557e-2, 1.088e-1, 1.250e-1, 1.392e-1, 1.623e-1]
st3 = [6.212e-3, 1.708e-2, 2.618e-2, 4.309e-2, 6.179e-2, 7.075e-2, 7.847e-2, 9.027e-2]
st4 = [2.047e-3, 5.619e-3, 8.599e-3, 1.410e-2, 2.006e-2, 2.283e-2, 2.511e-2, 2.825e-2]
st5 = [7.606e-5, 2.079e-4, 3.169e-4, 5.144e-4, 7.193e-4, 8.067e-4, 8.692e-4, 9.290e-4]
st6 = [3.469e-7, 9.446e-7, 1.434e-6, 2.306e-6, 3.169e-6, 3.504e-6, 3.702e-6, 3.799e-6]
muk3 = np.array([st0, st1, st2, st3, st4, st5, st6])

musR2 = np.array([muk0, muk1, muk2, muk3])

st0 = [1.583e-1, 4.008e-1, 5.706e-1, 8.075e-1, 9.561e-1, 9.885e-1, 9.990e-1, 1]
st1 = [1.583e-1, 4.008e-1, 5.706e-1, 8.075e-1, 9.561e-1, 9.885e-1, 9.990e-1, 1]
st2 = [1.583e-1, 4.008e-1, 5.706e-1, 8.075e-1, 9.561e-1, 9.885e-1, 9.990e-1, 1]
st3 = [1.583e-1, 4.008e-1, 5.706e-1, 8.075e-1, 9.561e-1, 9.885e-1, 9.990e-1, 1]
st4 = [1.583e-1, 4.008e-1, 5.706e-1, 8.075e-1, 9.561e-1, 9.885e-1, 9.990e-1, 1]
st5 = [1.583e-1, 4.008e-1, 5.706e-1, 8.075e-1, 9.561e-1, 9.885e-1, 9.990e-1, 1]
st6 = [1.583e-1, 4.008e-1, 5.706e-1, 8.075e-1, 9.561e-1, 9.885e-1, 9.990e-1, 1]
muk0 = np.array([st0, st1, st2, st3, st4, st5, st6])

st0 = [1.304e-1, 3.372e-1, 4.892e-1, 7.178e-1, 8.820e-1, 9.248e-1, 9.428e-1, 9.501e-1]
st1 = [8.010e-2, 2.071e-1, 3.003e-1, 4.399e-1, 5.393e-1, 5.642e-1, 5.731e-1, 5.744e-1]
st2 = [4.440e-2, 1.152e-1, 1.676e-1, 2.472e-1, 3.054e-1, 3.206e-1, 3.261e-1, 3.265e-1]
st3 = [1.923e-2, 5.024e-2, 7.354e-2, 1.098e-1, 1.376e-1, 1.452e-1, 1.482e-1, 1.485e-1]
st4 = [4.519e-3, 1.194e-2, 1.764e-2, 2.683e-2, 3.432e-2, 3.653e-2, 3.743e-2, 3.754e-2]
st5 = [1.083e-4, 2.904e-4, 4.347e-4, 6.782e-4, 8.929e-4, 9.623e-4, 9.927e-4, 9.966e-4]
st6 = [3.938e-7, 1.063e-6, 1.602e-6, 2.532e-6, 3.387e-6, 3.678e-6, 3.812e-6, 3.831e-6]
muk1 = np.array([st0, st1, st2, st3, st4, st5, st6])

st0 = [8.037e-2, 2.140e-1, 3.185e-1, 4.922e-1, 6.449e-1, 6.982e-1, 7.299e-1, 7.575e-1]
st1 = [5.782e-2, 1.538e-1, 2.286e-1, 3.521e-1, 4.586e-1, 4.942e-1, 5.138e-1, 5.277e-1]
st2 = [3.603e-2, 9.572e-2, 1.421e-1, 2.183e-1, 2.827e-1, 3.033e-1, 3.134e-1, 3.181e-1]
st3 = [1.711e-2, 4.546e-2, 6.746e-2, 1.034e-1, 1.335e-1, 1.427e-1, 1.468e-1, 1.478e-1]
st4 = [4.323e-3, 1.150e-2, 1.710e-2, 2.631e-2, 3.406e-2, 3.642e-2, 3.740e-2, 3.754e-2]
st5 = [1.079e-4, 2.896e-4, 4.337e-4, 6.772e-4, 8.926e-4, 9.622e-4, 9.927e-4, 9.966e-4]
st6 = [3.938e-7, 1.063e-6, 1.602e-6, 2.532e-6, 3.387e-6, 3.678e-6, 3.812e-6, 3.831e-6]
muk2 = np.array([st0, st1, st2, st3, st4, st5, st6])

st0 = [5.036e-2, 1.359e-1, 2.048e-1, 3.247e-1, 4.402e-1, 4.858e-1, 5.175e-1, 5.528e-1]
st1 = [3.850e-2, 1.038e-1, 1.564e-1, 2.474e-1, 3.343e-1, 3.681e-1, 3.907e-1, 4.145e-1]
st2 = [2.585e-2, 6.967e-2, 1.048e-1, 1.655e-1, 2.226e-1, 2.442e-1, 2.580e-1, 2.706e-1]
st3 = [1.346e-2, 3.624e-2, 5.445e-2, 8.573e-2, 1.147e-1, 1.252e-1, 1.315e-1, 1.361e-1]
st4 = [3.792e-3, 1.020e-2, 1.530e-2, 2.401e-2, 3.197e-2, 3.463e-2, 3.605e-2, 3.672e-2]
st5 = [1.052e-4, 2.831e-4, 4.251e-4, 6.674e-4, 8.856e-4, 9.579e-4, 9.908e-4, 9.961e-4]
st6 = [3.929e-7, 1.061e-6, 1.599e-6, 2.529e-6, 3.386e-6, 3.677e-6, 3.812e-6, 3.831e-6]
muk3 = np.array([st0, st1, st2, st3, st4, st5, st6])

musR3 = np.array([muk0, muk1, muk2, muk3])

tab_Z = np.array([musR0, musR1, musR2, musR3])

# Test:
# # print(tab_G[5, 3, 3, 3])
# from scipy.interpolate import CubicSpline
# def ApproxLog(x, y, x1):
#     A0 = np.array([[0., 0.],
#                    [0., 0.]])
#     A1 = np.array([[0., 0.],
#                    [0., 0.]])
#     A0[0, 0] = np.sum(y)
#     A0[0, 1] = np.sum(np.log(x))
#     A0[1, 0] = np.sum(y*np.log(x))
#     A1[0, 0] = float(len(x))
#     A1[0, 1] = np.sum(np.log(x))
#     A1[1, 0] = np.sum(np.log(x))
#     for i in range(len(x)):
#         A0[1, 1] += np.log(x[i])**2
#         A1[1, 1] += np.log(x[i])**2
#     a = np.linalg.det(A0)/np.linalg.det(A1)
#     print('a = ', a)
#     A0[0, 0] = float(len(x))
#     A0[0, 1] = np.sum(y)
#     A0[1, 0] = np.sum(np.log(x))
#     A0[1, 1] = np.sum(y*np.log(x))
#     A1[0, 0] = float(len(x))
#     A1[0, 1] = np.sum(np.log(x))
#     A1[1, 0] = np.sum(np.log(x))
#     A1[1, 1] = 0
#     for i in range(len(x)):
#         A1[1, 1] += np.log(x[i])**2
#     b = np.linalg.det(A0)/np.linalg.det(A1)
#     print('b = ', b)
#     return(a+b*np.log(x1))
# def ApproxPov(x, y, x1):
#     if y[0] == 0:
#         return 0
#     if x[0] == 0:
#         A0 = np.array([[0., 0.],
#                        [0., 0.]])
#         A1 = np.array([[0., 0.],
#                        [0., 0.]])
#         ones = np.ones(len(x))
#         A0[0, 0] = np.sum(np.log(y))
#         A0[0, 1] = np.sum(np.log(x+ones))
#         A0[1, 0] = np.sum(np.log(y)*np.log(x+ones))
#         A1[0, 0] = float(len(x))
#         A1[0, 1] = np.sum(np.log(x+ones))
#         A1[1, 0] = np.sum(np.log(x+ones))
#         for i in range(len(x)):
#             A0[1, 1] += np.log(x[i]+1)**2
#             A1[1, 1] += np.log(x[i]+1)**2
#         a = np.exp(np.linalg.det(A0)/np.linalg.det(A1))
#         # print('a = ', a)
#         A0[0, 0] = float(len(x))
#         A0[0, 1] = np.sum(np.log(y))
#         A0[1, 0] = np.sum(np.log(x+ones))
#         A0[1, 1] = np.sum(np.log(y)*np.log(x+ones))
#         A1[0, 0] = float(len(x))
#         A1[0, 1] = np.sum(np.log(x+ones))
#         A1[1, 0] = np.sum(np.log(x+ones))
#         A1[1, 1] = 0
#         for i in range(len(x)):
#             A1[1, 1] += np.log(x[i]+1)**2
#         b = np.linalg.det(A0)/np.linalg.det(A1)
#         # print('b = ', b)
#         c = (y[0]/a)**(-b)
#         # print('c = ', c)
#         return(a*(x1+c)**b)
#     else:
#         A0 = np.array([[0., 0.],
#                        [0., 0.]])
#         A1 = np.array([[0., 0.],
#                        [0., 0.]])
#         A0[0, 0] = np.sum(np.log(y))
#         A0[0, 1] = np.sum(np.log(x))
#         A0[1, 0] = np.sum(np.log(y)*np.log(x))
#         A1[0, 0] = float(len(x))
#         A1[0, 1] = np.sum(np.log(x))
#         A1[1, 0] = np.sum(np.log(x))
#         for i in range(len(x)):
#             A0[1, 1] += np.log(x[i])**2
#             A1[1, 1] += np.log(x[i])**2
#         a = np.exp(np.linalg.det(A0)/np.linalg.det(A1))
#         # print('a = ', a)
#         A0[0, 0] = float(len(x))
#         A0[0, 1] = np.sum(np.log(y))
#         A0[1, 0] = np.sum(np.log(x))
#         A0[1, 1] = np.sum(np.log(y)*np.log(x))
#         A1[0, 0] = float(len(x))
#         A1[0, 1] = np.sum(np.log(x))
#         A1[1, 0] = np.sum(np.log(x))
#         A1[1, 1] = 0
#         for i in range(len(x)):
#             A1[1, 1] += np.log(x[i])**2
#         b = np.linalg.det(A0)/np.linalg.det(A1)
#         # print('b = ', b)
#         return(a*x1**b)
# def ApproxExp(x, y, x1):
#     A0 = np.array([[0., 0.],
#                    [0., 0.]])
#     A1 = np.array([[0., 0.],
#                    [0., 0.]])
#     A0[0, 0] = np.sum(np.log(y))
#     A0[0, 1] = np.sum(x)
#     A0[1, 0] = np.sum(np.log(y)*x)
#     A1[0, 0] = float(len(x))
#     A1[0, 1] = np.sum(x)
#     A1[1, 0] = np.sum(x)
#     for i in range(len(x)):
#         A0[1, 1] += x[i]**2
#         A1[1, 1] += x[i]**2
#     a = np.exp(np.linalg.det(A0)/np.linalg.det(A1))
#     A0[0, 0] = float(len(x))
#     A0[0, 1] = np.sum(np.log(y))
#     A0[1, 0] = np.sum(x)
#     A0[1, 1] = np.sum(np.log(y)*x)
#     A1[0, 0] = float(len(x))
#     A1[0, 1] = np.sum(x)
#     A1[1, 0] = np.sum(x)
#     A1[1, 1] = 0
#     for i in range(len(x)):
#         A1[1, 1] += x[i]**2
#     b = np.linalg.det(A0)/np.linalg.det(A1)
#     return(a*np.exp(b*x1))
# def ApproxCub(tab_x, tab_y, tab_z, x, y):
#     zx = np.arange(float(len(tab_x)))
#     zy = np.arange(float(len(tab_y)))
#     for i in range(len(tab_y)):
#         for j in range(len(tab_x)):
#             zx[j] = tab_z[i, j]
#         f = CubicSpline(tab_x, zx, extrapolate = True)
#         zy[i] = f(x)
#     f = CubicSpline(tab_y, zy, extrapolate = True)
#     z = f(y)
#     return z
# def ApproxLin(x, y, x1):
#     if x1 > x.max():
#         x0 = x[-1]
#         x2 = x[-2]
#     else:
#         x0 = x[0]
#         x2 = x[1]
#         for i in range(len(x)-1):
#             if x1 > x[i]:
#                 x0 = x[i]
#                 x2 = x[i+1]
#             else:
#                 break
#     z = y[np.where(x == x0)]+(y[np.where(x == x2)]-y[np.where(x == x0)])*(x1 - x0)/(x2 - x0)
#     return z

# musR = 10
# k = 10
# b = 1
# p = 3
# s = np.zeros(len(tab_musR))
# Gpbk = np.zeros((len(tab_k), len(tab_b), len(tab_p)))
# for i in range(len(tab_p)):
#     for j in range(len(tab_b)):
#         for n in range(len(tab_k)):
#             for m in range(len(tab_musR)):
#                 s[m] = tab_G[m, n, j, i]
#             if musR > tab_musR[-1]:
#                 Gpbk[n, j, i] = ApproxPov(tab_musR, s, musR)
#             else:
#                 Gpbk[n, j, i] = ApproxLin(tab_musR, s, musR)
# s = np.zeros(len(tab_k))
# Gpb = np.zeros((len(tab_b), len(tab_p)))
# for i in range(len(tab_p)):
#     for j in range(len(tab_b)):
#         for n in range(len(tab_k)):
#             s[n] = Gpbk[n, j, i]
#         if k > tab_k[-1]:
#             Gpb[j, i] = tab_k[-1]
#         else:
#             Gpb[j, i] = ApproxLin(tab_k, s, k)
# s = np.zeros(len(tab_b))
# Gp = np.zeros(len(tab_p))
# for i in range(len(tab_p)):
#     for j in range(len(tab_b)):
#         s[j] = Gpb[j, i]
#     if b > tab_b[-1]:
#         Gp[i] = ApproxExp(tab_b, s, b)
#     else:
#         Gp[i] = ApproxLin(tab_b, s, b)
# if p < tab_p[0] or p > tab_p[-1]:
#     G = ApproxPov(tab_p, Gp, p)
# else:
#     G = ApproxLin(tab_p, Gp, p)
# print(G)
#
# musH = 3.441
# b1 = 2.6618
# aH = 1
# RH = 1.25
# s = np.zeros(len(tab_RH))
# Gpbk = np.zeros((len(tab_aH), len(tab_b1), len(tab_musH)))
# for i in range(len(tab_musH)):
#     for j in range(len(tab_b1)):
#         for n in range(len(tab_aH)):
#             for m in range(len(tab_RH)):
#                 s[m] = tab_Z[m, n, j, i]
#             f = CubicSpline(tab_RH, s, extrapolate=True)
#             if RH > tab_RH[-1] or f(RH) < 0:
#                 Gpbk[n, j, i] = ApproxLog(tab_RH, s, RH)
#             else:
#                 Gpbk[n, j, i] = f(RH)
# s = np.zeros(len(tab_aH))
# Gpb = np.zeros((len(tab_b1), len(tab_musH)))
# for i in range(len(tab_musH)):
#     for j in range(len(tab_b1)):
#         for n in range(len(tab_aH)):
#             s[n] = Gpbk[n, j, i]
#         f = CubicSpline(tab_aH, s, extrapolate=True)
#         if aH > tab_aH[-1] or f(aH) < 0:
#             Gpb[j, i] = ApproxExp(tab_aH, s, aH)
#         else:
#             Gpb[j, i] = f(aH)
# s = np.zeros(len(tab_b1))
# Gp = np.zeros(len(tab_musH))
# for i in range(len(tab_musH)):
#     for j in range(len(tab_b1)):
#         s[j] = Gpb[j, i]
#     if b1 > tab_b1[-1]:
#         Gp[i] = ApproxExp(tab_b1, s, b1) #gives inaccuracy
#     else:
#         Gp[i] = ApproxLin(tab_b1, s, b1)
# f = CubicSpline(tab_musH[0:(len(tab_musH)-1)], Gp[0:(len(tab_musH)-1)], extrapolate=True)
# if f(musH) > tab_musH[-2] or f(musH) < 0:
#     Z = Gp[-1]*np.exp(np.log(Gp[-2] / Gp[-1]) / tab_musH[-2]*musH)
# else:
#     Z = f(musH)
# print(Z)