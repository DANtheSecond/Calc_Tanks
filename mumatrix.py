import numpy as np

tab_p = [1.25, 1.5, 3, 5, 10]   #x 3 (cubic approximation)
tab_b = [0, 1, 2, 4, 7, 10, 20] #y 4 (exponential approximation)
tab_k = [1, 3, 5, 10]           #k 2 (cubic approximation)
tab_musR = [0, 1, 2, 3, 5, 10]  #z 1 (power approximation (excluding 0)

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

musR0 = np.array([muk0, muk1, muk2, muk3])

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

musR1 = np.array([muk0, muk1, muk2, muk3])

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

musR2 = np.array([muk0, muk1, muk2, muk3])

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

musR3 = np.array([muk0, muk1, muk2, muk3])

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

musR4 = np.array([muk0, muk1, muk2, muk3])

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

musR5 = np.array([muk0, muk1, muk2, muk3])

tab_G = np.array([musR0, musR1, musR2, musR3, musR4, musR5])

print(tab_G[5, 3, 3, 4])