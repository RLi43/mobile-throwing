import time

import numpy as np
import pickle

st = time.time()
brt_path = "../object_data/brt_gravity_only"
robot_path = "../robot_data/panda_5_joint_dense_1_dataset_15"

brt_data = np.load(brt_path + '/brt_data.npy')
# robot_zs = np.load(robot_path + '/robot_zs.npy')
# robot_gamma = np.load(robot_path + '/robot_gamma.npy')
robot_zs = np.arange(start=0.0, stop=1.10+0.01, step=0.05)
step_robot_zs = 0.05  # get step
robot_gamma = np.arange(start=20.0, stop=70.0+0.01, step=5.0)
robot_gamma *= np.pi/180.0

bzstart = min(robot_zs) - step_robot_zs * np.ceil((min(robot_zs) - min(brt_data[:, 1]))/step_robot_zs)
brt_zs = np.arange(start=bzstart, stop=max(brt_data[:, 1])+0.01, step=step_robot_zs)
num_zs = brt_zs.shape[0]
num_gammas = len(robot_gamma)  # brt_chunk.shape[1]

from bisect import bisect_left
def insert_idx(a, x):
    """

    :param a: sorted array, ascending
    :param x: element
    :return: the idx of the closest value to x
    """
    idx = bisect_left(a, x)
    if idx == 0: return idx
    elif idx == len(a):
        return idx - 1
    else:
        if (x - a[idx-1]) < (a[idx] - x):
            return idx - 1
        else: return idx


brt_chunk = [[[] for j in range(num_gammas)] for i in range(num_zs)]
states_num = 0
for x in brt_data:
    z = x[1]
    gamma = np.arctan2(x[3], x[2])
    # drop some states
    if gamma < min(robot_gamma) or gamma > max(robot_gamma):
        continue
    v = np.sqrt(x[2]**2 + x[3]**2)
    z_idx = insert_idx(brt_zs, z)
    ga_idx = insert_idx(robot_gamma, gamma)
    brt_chunk[z_idx][ga_idx].append(list(x) + [v])
    states_num += 1
# delete empty chunks
remove_i = 0
while True:
    chunk = brt_chunk[remove_i]
    empty = True
    for j in range(num_gammas):
        if len(chunk[j]) > 0:
            empty = False
            break
    if not empty:
        break
    remove_i += 1
brt_chunk = brt_chunk[remove_i:]
brt_zs = brt_zs[remove_i:, ...]
num_zs -= remove_i

brt_tensor = []
l = 0
while True:
    new_layer_brt = np.ones((num_zs, num_gammas, 5))
    stillhasvalue = False
    for i in range(num_zs):
        for k in range(num_gammas):
            if len(brt_chunk[i][k]) < l + 1:
                new_layer_brt[i, k, :] = np.nan
            else:
                stillhasvalue = True
                new_layer_brt[i, k, :] = brt_chunk[i][k][l]
    if not stillhasvalue:
        break
    brt_tensor.append(new_layer_brt)
    l += 1
brt_tensor = np.array(brt_tensor)
brt_tensor = np.moveaxis(brt_tensor, 0, 2)
brt_tensor = np.expand_dims(brt_tensor, axis=1)  # insert Phi dimension
print("Tensor Size: {0} with {1} states( occupation rate {2:0.1f}%)".format(
    brt_tensor.shape, states_num, 100*states_num*5.0/(np.prod(brt_tensor.shape))))

np.save(brt_path + "/brt_tensor1", brt_tensor)
np.save(brt_path + "/brt_zs1", brt_zs)

print("{0:0.2f}".format(time.time() - st))
