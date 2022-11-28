import time

import pybullet, pybullet_data
import numpy as np

gravity = -9.81
urdf_path = "franka_panda/panda.urdf"
q_ul = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
q_ll = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
q_dot_ul = np.array([2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100])
PANDA_BASE_HEIGHT = 0.5076438625
box_position = np.array([1.0, 0.0, -0.5])
desired_theta = -np.pi / 2


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

# Select a Q
# keep joint 1, 3, 5, 7 = 0
q0 = np.array([0.0, 0.5, 0.0, -0.5, 0.0, 2.0, 45 / 180.0 * np.pi])
AE, ori_q, J = forward(q0)
theta_E = np.arccos(ori_q[0])*2
print(AE, ori_q, theta_E)
print("r_E:{:.3f}, z_E:{:.3f}, theta_E:{:.3f}".format(AE[0], AE[2], theta_E))

# v0 = sqrt(x^2*g/ (x*sin(2theta) -2y cos^2(theta))
dr = box_position[0] - AE[0]
dz = box_position[2] - AE[2]
dtheta = desired_theta - theta_E
v0 = np.sqrt((dr ** 2 * np.abs(gravity)) / (dr * np.sin(2 * theta_E) - 2 * dz * (np.cos(theta_E) ** 2)))
vx = v0 * np.cos(theta_E)
vy = v0 * np.sin(theta_E)
flying_time = dr/vx
dtheta = dtheta % np.pi
if dtheta > np.pi/2:
    dtheta -= np.pi
omega = dtheta / flying_time  # dtheta + n*pi
print("dr:{:.3f}, dz:{:.3f}, dtheta:{:.3f} ".format(dr, dz, dtheta))
print("vx:{:.3f}, vy:{:.3f}, omega:{:.3f}, flying time::{:.3f}".format(vx, vy, omega, flying_time))
# solve q_dot
b = np.array([vx, 0, vy, 0, omega, 0]).transpose()
q_dot = np.linalg.pinv(J) @ b
print(q_dot)
if(np.any(np.abs(q_dot) > q_dot_ul)):
    print('the solution is out of the speed limit')

ANIMATE = False
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
    pybullet.resetBasePositionAndOrientation(bottleId, eef_state[0], pybullet.getQuaternionFromEuler([0, np.pi / 2, 0]))
    # eef_state[1])
    pybullet.resetJointState(robotId, gripper_joints[0], 0.03)
    pybullet.resetJointState(robotId, gripper_joints[1], 0.03)
    pybullet.stepSimulation()
    time.sleep(0.001)

    input("Enter to quit")
