"""
Algorithm to get velocity hedgehog
Input:
* robot -- robot model(forward kinematic and differential forward kinematics)
Output:
* max_z_phi_gamma -- Max velocity in z, phi, gamma cell
* q_z_phi_gamma -- The configuration to archive the max velocity
Data:
* q_min, q_max -- robot joint limit
* q_dot_min, q_dot_max -- robot joint velocity limit
* Delta_q -- joint grid size
* Z, Phi, Gamma -- velocity hedgehog grids
"""
import time

import numpy as np

import pybullet
import pybullet_data


class Robot:
    def __init__(self, q_min, q_max, q_dot_min, q_dot_max, urdf_path):
        self.q_min = q_min
        self.q_max = q_max
        self.q_dot_min = q_dot_min
        self.q_dot_max = q_dot_max
        clid = pybullet.connect(pybullet.DIRECT)
        pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
        self.robot = pybullet.loadURDF(urdf_path, [0, 0, 0], useFixedBase=True,
                                       flags=pybullet.URDF_USE_INERTIA_FROM_FILE)

    def forward(self, q: list) -> (np.ndarray, np.ndarray):
        pybullet.resetJointStatesMultiDof(self.robot, list(range(7)), [[qi] for qi in q])
        AE = np.array(pybullet.getLinkState(self.robot, 11)[0])
        Jl, Ja = pybullet.calculateJacobian(self.robot, 11, [0, 0, 0], q + [0.1, 0.1], [0.0] * 9, [0.0] * 9)
        J = np.vstack((Jl, Ja))
        J = J[:, :7]
        return AE, J

    def forward_l(self, q: list) -> (np.ndarray, np.ndarray):
        pybullet.resetJointStatesMultiDof(self.robot, list(range(7)), [[qi] for qi in q])
        AE = np.array(pybullet.getLinkState(self.robot, 11)[0])
        Jl, Ja = pybullet.calculateJacobian(self.robot, 11, [0, 0, 0], q + [0.1, 0.1], [0.0] * 9, [0.0] * 9)
        J = np.array(Jl)
        J = J[:, :7]
        return AE, J


def main(robot: Robot, delta_q, Z, Phi, Gamma, svthres=0.1):
    num_joint = robot.q_min.shape[0]
    # Build robot dataset
    q_candidates = computeMesh(robot.q_min[1:6], robot.q_max[1:6], delta_q)
    print("q size:", q_candidates.shape)
    # 1. Filter out q with small singular value
    # 2. Group data by Z
    Xz = reformat(q_candidates, robot, svthres, Z)
    # Initialize velocity hedgehog
    num_z = Z.shape[0]
    num_phi = Phi.shape[0]
    num_gamma = Gamma.shape[0]
    vel_max = np.zeros((num_z, num_phi, num_gamma))
    argmax_q = np.zeros((num_z, num_phi, num_gamma, num_joint))

    # Build velocity hedgehog
    for i in range(Z.shape[0]):
        qz = getqat(i, Xz)
        print("height {:.2f} Num(q): {}".format(Z[i], len(qz)))
        if len(qz) == 0:
            continue
        st = time.time()
        # vels = np.zeros((len(qz), num_phi, num_gamma))
        vels = np.array([batchLP(Phi, Gamma, q, robot) for q in qz])
        # for d, q in enumerate(qz):
        #    vels[d, :, :] = batchLP(Phi, Gamma, q, robot)

        vel_max[i, :, :] = np.max(vels, axis=0)
        argmax_q[i, :, :] = np.array(qz)[np.argmax(vels, axis=0), -7:]

        timecost = time.time() - st
        print("\tuse {0:.2f}s for {1} q, {2:.3f} s per q".format(timecost, len(qz), timecost / len(qz)))

    return vel_max, argmax_q


def computeMesh(q_min, q_max, delta_q):
    qs = [np.arange(qn, qa, delta_q) for qn, qa in zip(q_min, q_max)]
    return np.array(np.meshgrid(*qs)).transpose().reshape(-1, q_max.shape[0])


def reformat(Q, robot: Robot, thres, ZList, Z_TOLERANCE=0.05):
    filtered = 0
    farawayfromZ = 0
    zlen = ZList.shape[0]
    pad_zs = np.r_[-np.inf, ZList]
    Xz = [[] for i in range(zlen)]
    num_q = Q.shape[0]
    for i, q in enumerate(Q):
        if i % (num_q // 10) == 0:
            print('... {:.1f}% finished ...'.format(i / num_q * 100))
        q = [0.0] + q.tolist() + [0.0]
        AE, J = robot.forward(q)
        u, s, vh = np.linalg.svd(J)
        if np.min(s) < thres:  # singularity check
            filtered += 1
            continue

        zi = np.argmax(abs(pad_zs - AE[2]) < Z_TOLERANCE)
        if zi != 0:
            Xz[zi - 1].append(q)  # close to our desired z points
        else:
            farawayfromZ += 1

    print('{} were filtered out due to singularity, {} due to z, {} remain.'.format(
        filtered, farawayfromZ, num_q - filtered - farawayfromZ))

    return Xz


def filterBySingularValue(Q, robot: Robot, thres):
    """
    :param Q: sampled q list
    :return: [X, q] (X: x, y, z)
    """
    ret = []
    filtered = 0
    for q in Q:
        q = [0.0] + q.tolist() + [0.0]
        AE, J = robot.forward(q)
        u, s, vh = np.linalg.svd(J)
        if np.min(s) < thres:
            filtered += 1
            continue
        ret.append(AE.tolist() + q)
    print('{} were filtered out.'.format(filtered))
    return ret


def groupBy(X, ZList, Z_TOLERANCE=0.05):
    """

    :param X:
    :param ZList:
    :return:
    note: Z supposed to be sorted ascending
    """
    zlen = ZList.shape[0]
    pad_zs = np.r_[-np.inf, ZList]
    Xz = [[] for i in range(zlen)]
    for x in X:
        zi = np.argmax(abs(pad_zs - x[2]) < Z_TOLERANCE)
        if zi != 0:
            Xz[zi - 1].append(x)
    return Xz


def getqat(i, Xz):
    return Xz[i]


import cvxpy as cp


def batchLP(Phis, Gammas, q, robot):
    AE, J = robot.forward_l(q[-7:])  # we can reuse AE, J; but is it worth to store such huge data?
    fracyx = AE[1] / AE[0]
    Jinv = np.linalg.pinv(J)
    qdmin, qdmax = robot.q_dot_min, robot.q_dot_max

    s = cp.Variable(1)
    v = cp.Variable(3)
    objective = cp.Maximize(s)
    qv = Jinv @ v

    def lp_solve(cons):
        prob = cp.Problem(objective, cons)
        result = prob.solve(warm_start=True)
        return prob.value

    vels = np.array([[lp_solve([qdmin <= qv, qv <= qdmax,
                                v == s * np.array(
                                    [np.cos(gamma) * np.cos(fracyx + phi), np.cos(gamma) * np.sin(fracyx + phi),
                                     np.sin(gamma)])])
                      for gamma in Gammas] for phi in Phis])
    return vels


def LP(phi, gamma, Jinv, fracyx, qdmin, qdmax):
    """
    max s that:
    - q_dot_min <= inv(J(q))*v <= q_dot_max
    - v = s*    [ cos(gamma) * cos(AE_y / AE_x + phi)   ]
                [ cos(gamma) * sin(AE_y / AE_x + phi)   ]
                [ sin(gamma)                            ]
    :param phi:
    :param gamma:
    :param q:
    :param robot:
    :return:
    """

    s = cp.Variable(1)
    v = cp.Variable(3)
    objective = cp.Maximize(s)
    constraints = [qdmin <= Jinv @ v, Jinv @ v <= qdmax,
                   v == s * np.array(
                       [np.cos(gamma) * np.cos(fracyx + phi), np.cos(gamma) * np.sin(fracyx + phi), np.sin(gamma)])]
    prob = cp.Problem(objective, constraints)
    result = prob.solve(warm_start=True)
    return s.value[0]


pandas = Robot(
    np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]),
    np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]),
    np.array([-2.1750, -2.1750, -2.1750, -2.1750, -2.6100, -2.6100, -2.6100]),
    np.array([2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100]),
    "franka_panda/panda.urdf"
)

if __name__ == "__main__":
    # q = (pandas.q_min + pandas.q_max)/2

    # print(LP(0, np.pi/4, q, pandas))
    """
    X = np.random.random((5,3))
    Z = np.arange(0, 1, 0.2)
    Xz = groupBy(X, Z)
    print(Xz)
    for i in range(5):
        print(Z[i], Xz[i])
    """

    # X = filterBySingularValue([q[1:-1]], pandas, 0.01)
    # print(X)

    # print(computeMesh(pandas.q_min[1:6], pandas.q_max[1:6], 0.2))

    Z = np.arange(0, 1.2, 0.05)
    Phi = np.arange(-np.pi / 2, np.pi / 2, np.pi / 12)  # np.pi / 12
    Gamma = np.arange(np.pi / 9, np.pi * 7 / 18, np.pi / 36)  # np.pi / 36
    # delta_q = 0.2  # 4601952
    # delta_q = 0.3  # 686400
    # delta_q = 0.4  # 162000
    delta_q = 0.5  # 162000
    vel_max, argmax_q = main(robot=pandas, delta_q=delta_q, Z=Z, Phi=Phi, Gamma=Gamma)
    np.save("my_vel_max", vel_max)
    np.save("my_argmax_q", argmax_q)
    print(vel_max.shape)
    print(argmax_q.shape)

    # construct q_idx and qs
    print("Constructing q_idx")
    num_z, num_phi, num_gamma = len(Z), len(Phi), len(Gamma)
    qs = []
    qid_iter = 0
    q_idxs = np.zeros((num_z, num_phi, num_gamma))
    for i in range(num_z):
        for j in range(num_phi):
            for k in range(num_gamma):
                q = argmax_q[i, j, k, :]
                exist = False
                for d, qi in enumerate(qs):
                    if np.allclose(qi, q):
                        qid = d
                        exist = True
                        break
                if not exist:
                    qid = qid_iter
                    qid_iter += 1
                    qs.append(q)
                q_idxs[i, j, k] = qid
    np.save('my_qs', np.array(qs))
    np.save('my_q_idxs', q_idxs)
    print("Done.")
