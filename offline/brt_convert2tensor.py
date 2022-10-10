import numpy as np
import pickle

brt_path = "../object_data/brt_gravity_only"

with open(brt_path + '/brt_chunk.pkl', 'rb') as fp:
    brt_chunk = pickle.load(fp)
brt_tensor = []
brt_zs = np.load(brt_path + '/brt_zs.npy')
num_z = brt_zs.shape[0]
num_gammas = 11  # brt_chunk.shape[1]

l = 0
while True:
    new_layer_brt = np.ones((num_z, num_gammas, 5))
    stillhasvalue = False
    for i in range(num_z):
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


np.save(brt_path + "/brt_tensor", brt_tensor)