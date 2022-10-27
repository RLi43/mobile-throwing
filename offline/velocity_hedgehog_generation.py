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
        Jl, Ja = pybullet.calculateJacobian(self.robot, 11, [0, 0, 0], list(q) + [0.1, 0.1], [0.0] * 9, [0.0] * 9)
        J = np.vstack((Jl, Ja))
        J = J[:, :7]
        return AE, J


def main(robot: Robot, delta_q, Dis, Z, Phi, Gamma, svthres=0.1):
    num_joint = robot.q_min.shape[0]
    # Build robot dataset
    # The first joint and the last joint don't contribute to the pose
    q_candidates = computeMesh(robot.q_min[1:6], robot.q_max[1:6], delta_q)
    print("q size:", q_candidates.shape)
    # Filter out q with small singular value
    X = filterBySingularValue(q_candidates, robot, svthres)
    # Group data by Z and AE
    Xzd = groupBy(X, Z, Dis)
    # Initialize velocity hedgehog
    num_z = Z.shape[0]
    num_ae = Dis.shape[0]
    num_phi = Phi.shape[0]
    num_gamma = Gamma.shape[0]
    vel_max = np.zeros((num_z, num_ae, num_phi, num_gamma))
    argmax_q = np.zeros((num_z, num_ae, num_phi, num_gamma, num_joint))

    # Build velocity hedgehog
    for i in range(Z.shape[0]):
        for j in range(Dis.shape[0]):
            qzd = Xzd[i][j]
            if len(qzd) == 0:
                continue
            print("height", Z[i], " AE", Dis[j], " Num(q):", len(qzd))

            st = time.time()
            vels = np.zeros((len(qzd), num_phi, num_gamma))
            for k, q in enumerate(qzd):
                AE, J = robot.forward(q[:7])
                fracyx = AE[1]/AE[0]
                Jinv = np.linalg.pinv(J[:3, :])
                qdmin, qdmax = robot.q_dot_min, robot.q_dot_max
                vels[k, :, :] = np.array([[LP(phi, gamma, Jinv, fracyx, qdmin, qdmax) for gamma in Gamma] for phi in Phi])

            vel_max[i, :, :] = np.max(vels, axis=0)
            argmax_q[i, :, :] = np.array(qzd)[np.argmax(vels, axis=0), :7]

            timecost = time.time() - st
            print("use {0:.2f}s for {1} q, {2:.3f} s per q".format(timecost, len(qzd), timecost/len(qzd)))
    return vel_max, argmax_q


def computeMesh(q_min, q_max, delta_q):
    qs = [np.arange(qn, qa, delta_q) for qn, qa in zip(q_min, q_max)]
    return np.array(np.meshgrid(*qs)).transpose().reshape(-1, q_max.shape[0])


def filterBySingularValue(Q, robot: Robot, thres):
    """
    :param Q: sampled q list
    :return: [X, q] (X: x, y, z)
    """
    ret = []
    for q in Q:
        q = [0.0] + q.tolist() + [0.0]
        AE, J = robot.forward(q)
        u, s, vh = np.linalg.svd(J)
        if np.min(s) < thres:
            continue
        ret.append(AE.tolist() + q)
    return np.array(ret)

def insert_idx(a, v):
    idl = np.searchsorted(a, v)  # left
    if idl == 0:
        return idl
    if idl == len(a):
        return idl-1
    if (a[idl] - v) < (v - a[idl-1]):
        return idl
    return idl-1
def groupBy(X, ZList, AEList):
    """

    :param X:
    :param ZList:
    :param AEList:
    :return:
    note: Z and AE supposed to be sorted ascending
    """
    zlen = ZList.shape[0]
    aelen = AEList.shape[0]
    ae = np.linalg.norm(X[:, :2], axis=1)
    Xzd = [[[] for j in range(aelen)] for i in range(zlen)]
    for i, x in enumerate(X):
        zi = insert_idx(ZList, x[2])
        aei = insert_idx(AEList, ae[i])
        Xzd[zi][aei].append(x)
    return Xzd


import cvxpy as cp

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
    result = prob.solve()
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
    Dis = np.arange(0, 1.1, 0.05)
    Phi = np.arange(-np.pi / 2, np.pi / 2, np.pi / 4)  # np.pi / 12
    Gamma = np.arange(np.pi / 6, np.pi / 3, np.pi / 12)  # np.pi / 36
    vel_max, argmax_q = main(robot=pandas, delta_q=0.8, Z=Z, Dis=Dis, Phi=Phi, Gamma=Gamma)
    np.save("vel_max", vel_max)
    np.save("argmax_q", argmax_q)
    print(vel_max.shape)
    print(argmax_q.shape)
