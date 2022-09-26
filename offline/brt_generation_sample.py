import os
import pickle
import time
import math
import numpy as np
from scipy.integrate import odeint
from sys import getsizeof


def main():
    r_dot0 = np.arange(0.2, 2.0, 0.5)
    z_dot0 = np.arange(-5.0, -2, 0.5)
    brt_data = brt_data_generation(flying_dynamics, r_dot0, z_dot0)


def flying_dynamics(t, x):
    g = 9.81
    dr = x[2]
    dz = x[3]
    dr_dot = 0.0
    dz_dot = - g
    return [dr, dz, dr_dot, dz_dot]


def brt_data_generation(flying_dynamics, r_dot0, z_dot0, path=None):
    st = time.time()
    t = np.linspace(2, 0, 51)
    n_steps = t.shape[0]

    n_r_dot0 = r_dot0.shape[0]
    n_z_dot0 = z_dot0.shape[0]
    n_velo = n_r_dot0 * n_z_dot0

    [r_dot0s, z_dot0s] = np.meshgrid(r_dot0, z_dot0)

    r_dot0s_flat = r_dot0s.flatten()
    z_dot0s_flat = z_dot0s.flatten()

    brt_data = np.zeros((n_velo, n_steps, 4))

    for i in range(n_velo):
        sol = odeint(flying_dynamics, [0, 0, r_dot0s_flat[i], z_dot0s_flat[i]], t, tfirst=True)
        brt_data[i, :, :] = sol

    print(brt_data[0,:])
    brt_data = brt_data.reshape(-1,4)
    print("Original size: ", brt_data.shape[0])
    # filter out data with insane number
    brt_data = brt_data[(brt_data[:, 0] > -10)
                        & (brt_data[:, 1] > -5.0)
                        & (brt_data[:, 1] < 5.0)
                        & (brt_data[:, 2] < 10.0)
                        & (brt_data[:, 3] < 10.0)]
    print("Filtered size: ", brt_data.shape[0])
    if path is not None:
        np.save(path, brt_data)
    print("Generated", n_velo, "flying trajectories in %.3f" % (time.time() - st), "seconds with",
          round(getsizeof(brt_data) / 1024 / 1024,2), "MB")
    return brt_data


if __name__ == '__main__':
    main()
