import time

import numpy as np
import pybullet
import pybullet_data

gravity = -9.81
urdf_path = "franka_panda/panda.urdf"
q_ul = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
q_ll = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
q_dot_ul = np.array([2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100])
PANDA_BASE_HEIGHT = 0.5076438625
bottle_height = 0.18

box_position = np.array([1.5, 0.0, 0.5])
desired_theta = -np.pi / 2
desired_vy_end_max = 1.5
desired_vx_end_max = 0.1

# Select a Q
# keep joint 1, 3, 5, 7 = 0
q0 = np.array([0.0, 0.7, 0.0, -0.5, 0.0, 2.0, -135 / 180.0 * np.pi])


def forward(q):
    controlled_joints = [0, 1, 2, 3, 4, 5, 6]
    pybullet.resetJointStatesMultiDof(robot, controlled_joints, [[q0_i] for q0_i in q])
    linkstate = pybullet.getLinkState(robot, 11)
    AE = linkstate[0]
    ori = linkstate[1]
    q = q.tolist()
    Jl, Ja = pybullet.calculateJacobian(robot, 11, [0, 0, 0], q + [0.1, 0.1], [0.0] * 9, [0.0] * 9)
    J = np.vstack((Jl, Ja))
    J = J[:, :7]
    return AE, ori, J


clid = pybullet.connect(pybullet.DIRECT)
pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
robot = pybullet.loadURDF(urdf_path, [0, 0, 0], useFixedBase=True, flags=pybullet.URDF_USE_INERTIA_FROM_FILE)

AE, ori_q, J = forward(q0)
theta_E = np.arccos(ori_q[3]) * 2 * np.sign(ori_q[1])
# print(AE, ori_q, theta_E)
print("r_E:{:.3f}, z_E:{:.3f}, theta_E:{:.3f}".format(AE[0], AE[2], theta_E))

dz = box_position[2] - AE[2] + bottle_height / 2
dr = box_position[0] - AE[0]
dtheta = desired_theta - theta_E
dtheta = dtheta % np.pi
if dtheta > np.pi / 2:
    dtheta -= np.pi
print("dr:{:.3f}, dz:{:.3f}, dtheta:{:.3f}".format(dr, dz, dtheta))

# find avaliable vx, vy
def flying_time(vy):
    return -vy / gravity + np.sqrt((vy / gravity) ** 2 + 2 * dz / gravity)


def qdot4vel(vx, vy):
    ft = flying_time(vy)
    vy_end = vy + gravity * ft
    omega = dtheta / ft

    vels = np.array([[vx, 0, vy, 0, omega, 0]]).transpose()
    q_dot = np.linalg.pinv(J) @ vels
    return q_dot

    # print("vx:{:.3f}, vy:{:.3f}, omega:{:.3f}, flying time::{:.3f}".format(vx, vy, omega, ft))
    # print(q_dot)


def qdotnorm(x):
    q_dot = qdot4vel(x[0], x[1])
    return np.linalg.norm(q_dot)
def qdotmax(x):
    q_dot = qdot4vel(x[0], x[1])
    return np.max(np.abs(q_dot))
def deviation_onx(x):
    return abs(dr - x[0] * flying_time(x[1]))

from scipy.optimize import minimize

# vx >= 0, vy >= 0
x_init = np.array([0, 0])
# obj = deviation_onx
obj = qdotmax
res = minimize(obj, x_init, constraints=(
    {'type': 'ineq', 'fun': lambda x: x[0]},
    {'type': 'ineq', 'fun': lambda x: x[1]},
    {'type': 'ineq', 'fun': lambda x: q_dot_ul * 0.9 - qdot4vel(x[0], x[1])[:, 0]},
    {'type': 'ineq', 'fun': lambda x: q_dot_ul * 0.9 + qdot4vel(x[0], x[1])[:, 0]},
))
vx, vy = res.x
q_dot = qdot4vel(vx, vy)
ft = flying_time(vy)
omega = dtheta / ft
print('vx:{:.3f}, vy:{:.3f}, omega:{:.3f}, flying time:{:.3f}'.format(vx, vy, omega, ft))
deviation = dr - vx * ft
print('translation on x:{:.3f}, deviation:{:.3f}'.format(vx * ft, deviation))
print(q_dot[:, 0])
# if (np.any(np.abs(q_dot) > q_dot_ul)):
#     print('the solution is out of the speed limit')

ANIMATE = True
if ANIMATE:
    # Simulation
    pybullet.disconnect()
    clid = pybullet.connect(pybullet.GUI)
    pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0)
    pybullet.resetDebugVisualizerCamera(cameraDistance=2.0, cameraYaw=10, cameraPitch=-40,
                                        cameraTargetPosition=[-0.75, 0.5, 0.2])
    hz = 1000
    delta_t = 1.0 / hz
    pybullet.setGravity(0, 0, gravity)
    pybullet.setTimeStep(delta_t)
    pybullet.setRealTimeSimulation(0)

    controlled_joints = [3, 4, 5, 6, 7, 8, 9]
    gripper_joints = [12, 13]
    pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
    robotEndEffectorIndex = 14
    robotId = pybullet.loadURDF("../descriptions/rbkairos_description/robots/rbkairos_panda_hand.urdf",
                                [-box_position[0], -box_position[1], 0], useFixedBase=True)

    planeId = pybullet.loadURDF("plane.urdf", [0, 0, 0.0])
    # Create Cylinder
    # bottleId = pybullet.createVisualShape(shapeType=pybullet.GEOM_CYLINDER, radius=0.03, length=0.1)
    # currently the collision size is reduced to avoid collision with the gripper
    bottleId = pybullet.loadURDF("../descriptions/robot_descriptions/objects_description/objects/bottle.urdf",
                                 [-3.0, 0, 3], globalScaling=0.01)
    # soccerballId = pybullet.loadURDF("soccerball.urdf", [-3.0, 0, 3], globalScaling=0.05)
    boxId = pybullet.loadURDF("../descriptions/robot_descriptions/objects_description/objects/box.urdf",
                              [0, 0, PANDA_BASE_HEIGHT + box_position[2]],
                              globalScaling=0.5)
    pybullet.changeDynamics(bottleId, -1, mass=1.0, linearDamping=0.00, angularDamping=0.00,
                            rollingFriction=0.03,
                            spinningFriction=0.03, restitution=0.2, lateralFriction=0.03)
    pybullet.changeDynamics(planeId, -1, restitution=0.9)
    pybullet.changeDynamics(robotId, gripper_joints[0], jointUpperLimit=100)
    pybullet.changeDynamics(robotId, gripper_joints[1], jointUpperLimit=100)

    pybullet.resetBasePositionAndOrientation(robotId, [-box_position[0], -box_position[1], 0], [0, 0, 0, 1])
    pybullet.resetJointStatesMultiDof(robotId, controlled_joints, [[q0_i] for q0_i in q0])
    eef_state = pybullet.getLinkState(robotId, robotEndEffectorIndex, computeLinkVelocity=1)
    pybullet.resetBasePositionAndOrientation(bottleId, eef_state[0], eef_state[1])
    pybullet.resetBaseVelocity(bottleId, linearVelocity=[vx, 0, vy], angularVelocity=[0, omega, 0])
    # eef_state[1])
    pybullet.resetJointState(robotId, gripper_joints[0], 0.03)
    pybullet.resetJointState(robotId, gripper_joints[1], 0.03)
    # pybullet.stepSimulation()
    try:
        while True:
            pybullet.stepSimulation()
            time.sleep(0.001)
    except KeyboardInterrupt:
        pass
