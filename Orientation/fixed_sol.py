import time

import numpy as np
import pybullet, pybullet_data
from ruckig import InputParameter, Ruckig, Trajectory, Result


# the solution found by experiment (destination)
qd = np.array([0.0, 0.4, 0.0, -0.7, 0.0, 2.02, 0.0 + 45/180*np.pi])
qd_dot = np.array([0, -1.10508170520, 0, 2.15504300e+00, 0.0, 2.5504305e+00, 0]) * 0.999

# throw state
clid = pybullet.connect(pybullet.DIRECT)
pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
urdf_path = "franka_panda/panda.urdf"
robot = pybullet.loadURDF(urdf_path, [0, 0, 0], useFixedBase=True, flags=pybullet.URDF_USE_INERTIA_FROM_FILE)
controlled_joints = [0, 1, 2, 3, 4, 5, 6]
pybullet.resetJointStatesMultiDof(robot, controlled_joints, [[q0_i] for q0_i in qd])
AE =pybullet.getLinkState(robot, 11)[0]
q = qd.tolist()
Jxyz, Jrpy = pybullet.calculateJacobian(robot, 11, [0, 0, 0], q+[0.1, 0.1], qd_dot.tolist() + [0.0]*2, [0.0]*9)
throw_state_xyz = np.array(Jxyz)[:, :7] @ qd_dot
throw_state_rpy = np.array(Jrpy)[:, :7] @ qd_dot
with np.printoptions(precision=3, suppress=True):
    print('throw state')
    print('xyz', throw_state_xyz)
    print('rpy', throw_state_rpy)
pybullet.disconnect(clid)


# initial state
q0 = np.array([0.0, 0.2, 0.0, -0.5, 0.0, 1.00, 0.0 + 45/180*np.pi])
q0_dot = np.array([0.0] * 7)
base_position0 = [0.0, 0.0]
base_positiond = base_position0

# find trajectory
inp = InputParameter(9)
zeros2 = np.zeros(2)
inp.current_position = np.concatenate((q0, base_position0))
inp.current_velocity = np.concatenate((q0_dot, zeros2))
inp.current_acceleration = np.zeros(9)

inp.target_position = np.concatenate((qd, base_positiond))
inp.target_velocity = np.concatenate((qd_dot, zeros2))
inp.target_acceleration = np.zeros(9)

inp.max_velocity = np.array([2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100, 2.0, 2.0])
inp.max_acceleration = np.array([15, 7.5, 10, 12.5, 15, 20, 20, 5.0, 5.0]) -1.0
inp.max_jerk = np.array([7500, 3750, 5000, 6250, 7500, 10000, 10000, 1000, 1000]) - 100

otg = Ruckig(9)
trajectory = Trajectory(9)
_ = otg.calculate(inp, trajectory)

# simulate
video_path = None
PANDA_BASE_HEIGHT = 0.5076438625
# box_position = throw_config_full[-1]
clid = pybullet.connect(pybullet.GUI)
pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0)
pybullet.resetDebugVisualizerCamera(cameraDistance=3.0, cameraYaw=160, cameraPitch=-40, cameraTargetPosition=[0.75, -0.75, 0])

# NOTE: need high frequency
hz = 1000
delta_t = 1.0 / hz
pybullet.setGravity(0, 0, -9.81)
pybullet.setTimeStep(delta_t)
pybullet.setRealTimeSimulation(0)

#AE = throw_config_full[-2]
#EB = box_position - AE

controlled_joints = [3, 4, 5, 6, 7, 8, 9]
gripper_joints = [12, 13]
numJoints = len(controlled_joints)
pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
robotEndEffectorIndex = 14
robotId = pybullet.loadURDF("../descriptions/rbkairos_description/robots/rbkairos_panda_hand.urdf",
                     [0, 0, PANDA_BASE_HEIGHT], useFixedBase=True)

planeId = pybullet.loadURDF("plane.urdf", [0, 0, 0.0])
#boxId = pybullet.loadURDF("../descriptions/robot_descriptions/objects_description/objects/box.urdf", [1, 0, 0.5], globalScaling=1.0)
tableId = pybullet.loadURDF("../descriptions/robot_descriptions/objects_description/objects/table.urdf", [1.25, 0, 0], globalScaling=1.0)
# soccerballId = pybullet.loadURDF("soccerball.urdf", [-3.0, 0, 3], globalScaling=0.05)
# pybullet.changeDynamics(soccerballId, -1, mass=1.0, linearDamping=0.00, angularDamping=0.00, rollingFriction=0.03,
#                  spinningFriction=0.03, restitution=0.2, lateralFriction=0.03)
# the base point of the bottle is the center of the bottle
# bottleId = pybullet.loadURDF("../descriptions/robot_descriptions/objects_description/objects/bottle.urdf",
#                                  [-3.0, 0, 3], globalScaling=0.01)
# pybullet.changeDynamics(bottleId, -1, mass=0.2, linearDamping=0.00, angularDamping=0.00,
#                         rollingFriction=0.03,
#                         spinningFriction=0.03, restitution=0.2, lateralFriction=0.03)
water_bottle_id = pybullet.loadURDF("../descriptions/robot_descriptions/objects_description/objects/water_bottle.urdf",
                                    #flags=pybullet.URDF_MERGE_FIXED_LINKS,
                                    globalScaling=0.001,
                                    basePosition=[0, 0, 0], baseOrientation=[-0.7071068, 0, 0, 0.7071068],
                                    )
print(pybullet.getBodyInfo(water_bottle_id))
#print("water bottle 0:", pybullet.getDynamicsInfo(water_bottle_id, 0))
#print("water bottle 1:", pybullet.getDynamicsInfo(water_bottle_id, 1))
bottle_height = 0.18
move_down_distance = 0.21

pybullet.changeDynamics(robotId, gripper_joints[0], jointUpperLimit=100)
pybullet.changeDynamics(robotId, gripper_joints[1], jointUpperLimit=100)
objectId = water_bottle_id

t0, tf = 0, trajectory.duration
plan_time = tf - t0
sample_t = np.arange(0, tf, delta_t)
n_steps = sample_t.shape[0]
traj_data = np.zeros([3, n_steps, 7])
base_traj_data = np.zeros([3, n_steps, 2])
for i in range(n_steps):
    for j in range(3):
        tmp = trajectory.at_time(sample_t[i])[j]
        traj_data[j, i] = tmp[:7]
        base_traj_data[j, i] = tmp[-2:]

# reset the joint
# see https://github.com/bulletphysics/bullet3/issues/2803#issuecomment-770206176
q0 = traj_data[0, 0]
pybullet.resetBasePositionAndOrientation(robotId, np.append(base_traj_data[0, 0], 0.0), [0, 0, 0, 1])
pybullet.resetJointStatesMultiDof(robotId, controlled_joints, [[q0_i] for q0_i in q0])
eef_state = pybullet.getLinkState(robotId, robotEndEffectorIndex, computeLinkVelocity=1)
bt_pos = eef_state[0] + np.array(pybullet.getMatrixFromQuaternion(eef_state[1])).reshape((3, 3)) @ np.array([0, 0, move_down_distance]).T
pybullet.resetBasePositionAndOrientation(objectId, bt_pos, eef_state[1])
pybullet.resetJointState(robotId, gripper_joints[0], 0.03)
pybullet.resetJointState(robotId, gripper_joints[1], 0.03)
tt = 0
traj_finish = False

if not (video_path is None):
    logId = pybullet.startStateLogging(loggingType=pybullet.STATE_LOGGING_VIDEO_MP4, fileName=video_path)
removed = False
while (True):
    if not removed:
        # robot
        if not traj_finish:
            ref_full = trajectory.at_time(tt)
            ref = [ref_full[i][:7] for i in range(3)]
            ref_base = [ref_full[i][-2:] for i in range(3)]
            pybullet.resetJointStatesMultiDof(robotId, controlled_joints, [[q0_i] for q0_i in ref[0]],
                                       targetVelocities=[[q0_i] for q0_i in ref[1]])
            pybullet.resetBasePositionAndOrientation(robotId, np.append(ref_base[0], 0.0), [0, 0, 0, 1])
        else:
            # stop the robot immediately
            ref_full = trajectory.at_time(plan_time)
            ref = [ref_full[i][:7] for i in range(3)]
            ref_base = [ref_full[i][-2:] for i in range(3)]
            pybullet.resetJointStatesMultiDof(robotId, controlled_joints, [[q0_i] for q0_i in ref[0]])
            pybullet.resetBasePositionAndOrientation(robotId, np.append(ref_base[0], 0.0), [0, 0, 0, 1])
    if not removed:
        # gripper and the object
        if tt > plan_time - 1 * delta_t:
            # open the gripper
            pybullet.resetJointState(robotId, gripper_joints[0], 0.05)
            pybullet.resetJointState(robotId, gripper_joints[1], 0.05)
            print('Robot Trajectory finished')
            eef_state = pybullet.getLinkState(robotId, robotEndEffectorIndex, computeLinkVelocity=1)
            print('End Effector', [["{0:.3f}".format(item) for item in eef_state[esi]]for esi in [0, 1, -2, -1]])
            print('bottle', [["{0:.3f}".format(item) for item in value] for value in pybullet.getBasePositionAndOrientation(water_bottle_id)]
                  + [["{0:.3f}".format(item) for item in value] for value in pybullet.getBaseVelocity(water_bottle_id)])
            #pybullet.removeBody(robotId)
            #removed = True
        else:
            # hold the ball
            eef_state = pybullet.getLinkState(robotId, robotEndEffectorIndex, computeLinkVelocity=1)
            pybullet.resetBasePositionAndOrientation(
                objectId,
                eef_state[0] + np.array(pybullet.getMatrixFromQuaternion(eef_state[1])).reshape((3, 3)) @ np.array([0, 0, move_down_distance]).T,
                eef_state[1])
            pybullet.resetBaseVelocity(objectId, linearVelocity=np.array(eef_state[-2]), angularVelocity=eef_state[-1])
    pybullet.stepSimulation()
    tt = tt + delta_t
    if tt > trajectory.duration:
        traj_finish = True
    time.sleep(delta_t*10)
    if tt > 6.0:
        break
if not (video_path is None):
    pybullet.stopStateLogging(logId)
pybullet.disconnect()


