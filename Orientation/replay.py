import pybullet, pybullet_data
import numpy as np
import time
from scipy import interpolate

traj_path = 'robot_state_data_2.npy'
video_path = None  # 'sim1808.mp4'
start_time = 18
slow = 1.0

trajectory = np.load(traj_path)
trajectory[7, :] += np.pi / 180.0 * 45.0
trajectory_duration = trajectory[0, -1]
position_inter = interpolate.interp1d(trajectory[0, :], trajectory[1:8, :])
vel_inter = interpolate.interp1d(trajectory[0, :], trajectory[8:15, :])
plan_time = 20.23605513572693 + 0.050
# 3 - 28.04465413093567 + 0.080

delay_release = 0.020
force = 0.0

PANDA_BASE_HEIGHT = 0.72  # 0.5076438625
# simulate
# box_position = throw_config_full[-1]
clid = pybullet.connect(pybullet.GUI)
pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0)
pybullet.resetDebugVisualizerCamera(cameraDistance=3.0, cameraYaw=160, cameraPitch=-40,
                                    cameraTargetPosition=[0.75, -0.75, 0])

# NOTE: need high frequency
hz = 500
delta_t = 1.0 / hz
pybullet.setGravity(0, 0, -9.81)
pybullet.setTimeStep(delta_t)
pybullet.setRealTimeSimulation(0)

controlled_joints = [0, 1, 2, 3, 4, 5, 6]  # [3, 4, 5, 6, 7, 8, 9]
gripper_joints = [9, 10]
numJoints = len(controlled_joints)
pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
robotEndEffectorIndex = 11  # 14
# robotId = pybullet.loadURDF("../descriptions/rbkairos_description/robots/rbkairos_panda_hand.urdf",
#                      [0, 0, PANDA_BASE_HEIGHT], useFixedBase=True)
robotId = pybullet.loadURDF("../descriptions/franka_panda_bullet_new/panda.urdf",
                            [0, 0, PANDA_BASE_HEIGHT], useFixedBase=True)

planeId = pybullet.loadURDF("plane.urdf", [0, 0, 0.0])
tableId = pybullet.loadURDF("../descriptions/robot_descriptions/objects_description/objects/table.urdf",
                            [0.85, 0, 0], pybullet.getQuaternionFromEuler([0, 0, np.pi / 2]), globalScaling=1.0)
# soccerballId = pybullet.loadURDF("soccerball.urdf", [-3.0, 0, 3], globalScaling=0.05)
# pybullet.changeDynamics(soccerballId, -1, mass=1.0, linearDamping=0.00, angularDamping=0.00, rollingFriction=0.03,
#                  spinningFriction=0.03, restitution=0.2, lateralFriction=0.03)
# the base point of the bottle is the center of the bottle
# bottleId = pybullet.loadURDF("../descriptions/robot_descriptions/objects_description/objects/bottle.urdf",
#                                  [-3.0, 0, 3], globalScaling=0.01)
water_bottle_id = pybullet.loadURDF(
    "../descriptions/robot_descriptions/objects_description/objects/water_bottle.urdf",
    #flags=pybullet.URDF_MERGE_FIXED_LINKS,
    globalScaling=0.001, basePosition=[1, 0, 0.6], baseOrientation=[0, 1, 0, 0])
print(pybullet.getBodyInfo(water_bottle_id))
# print("water bottle 0:", pybullet.getDynamicsInfo(water_bottle_id, 0))
# print("water bottle 1:", pybullet.getDynamicsInfo(water_bottle_id, 1))

move_down_distance = 0.02  # 225 - 206 cap CoM

pybullet.changeDynamics(robotId, gripper_joints[0], jointUpperLimit=100)
pybullet.changeDynamics(robotId, gripper_joints[1], jointUpperLimit=100)
objectId = water_bottle_id

# for i in range(pybullet.getNumJoints(robotId)):
#     index, jName, _, _, _, _, _, _, _, _, _, _, lName, _, _, _, pIndex = pybullet.getJointInfo(robotId, i)
#     print(index, 'joint name', jName, 'link name', lName, 'parent index', pIndex)

# reset the joint
# see https://github.com/bulletphysics/bullet3/issues/2803#issuecomment-770206176

q0 = position_inter(start_time)
pybullet.resetJointStatesMultiDof(robotId, controlled_joints, [[q0_i] for q0_i in q0])

pybullet.resetBasePositionAndOrientation(objectId, [1, 0, 0.6], [0, 0, 0, 1])
object_state = pybullet.getBasePositionAndOrientation(water_bottle_id)

def set_bottle():
    # rotate the bottle with the gripper
    eef_state = pybullet.getLinkState(robotId, robotEndEffectorIndex, computeLinkVelocity=1)
    # the base of the bottle is the bottom
    # -- so the position of the bottle should go along the gripper z for bottle height - grasp length
    bt_pos = eef_state[0] + np.array(pybullet.getMatrixFromQuaternion(eef_state[1])).reshape((3, 3)) \
             @ np.array([0, 0, move_down_distance]).T
    # flip the bottle
    _, bt_ori = pybullet.multiplyTransforms([0, 0, 0], eef_state[1], [0, 0, 0], [0, 1, 0, 0])
    pybullet.resetBasePositionAndOrientation(objectId, bt_pos, bt_ori)
    pybullet.resetBaseVelocity(objectId, linearVelocity=np.array(eef_state[-2]),
                               angularVelocity=eef_state[-1])


set_bottle()
pybullet.resetJointState(robotId, gripper_joints[0], 0.03)
pybullet.resetJointState(robotId, gripper_joints[1], 0.03)
tt = start_time
traj_finish = False

if not (video_path is None):
    logId = pybullet.startStateLogging(loggingType=pybullet.STATE_LOGGING_VIDEO_MP4, fileName=video_path)
removed = False

eef_state = pybullet.getLinkState(robotId, robotEndEffectorIndex, computeLinkVelocity=1)
#object_state = pybullet.getLinkState(water_bottle_id, 0, computeLinkVelocity=1)

#print(np.array(eef_state[0]) - np.array(object_state[0]), np.array(eef_state[-2]) - np.array(object_state[-2]))

object_state = pybullet.getBasePositionAndOrientation(water_bottle_id)

while True:
    if not removed:
        # robot
        if not traj_finish:
            pybullet.resetJointStatesMultiDof(robotId, controlled_joints, [[q0_i] for q0_i in position_inter(tt)],
                                              targetVelocities=[[q0_i] for q0_i in vel_inter(tt)])
        else:
            # stop the robot immediately
            ref = trajectory[:, -1]
            pybullet.resetJointStatesMultiDof(robotId, controlled_joints, [[q0_i] for q0_i in ref[1:8]])
    if not removed:
        # gripper and the object
        if tt < plan_time - delay_release:  # - delay_release * delta_t:  # early release
            # hold the ball
            set_bottle()
            cap_state = pybullet.getBasePositionAndOrientation(objectId), pybullet.getBaseVelocity(objectId)
            water_state = pybullet.getLinkState(objectId, 1, computeLinkVelocity=1)
        elif tt <= plan_time:
            time.sleep(delta_t * 100)
            # open the gripper
            pybullet.resetJointState(robotId, gripper_joints[0], 0.05)
            pybullet.resetJointState(robotId, gripper_joints[1], 0.05)
            eef_state = pybullet.getLinkState(robotId, robotEndEffectorIndex, computeLinkVelocity=1)
            # apply force
            #object_state = pybullet.getLinkState(water_bottle_id, 0, computeLinkVelocity=1)
            #relative_vel = np.array(eef_state[-2]) - np.array(object_state[-2])
            #forcevec = force * relative_vel / np.linalg.norm(relative_vel)
            #pybullet.applyExternalForce(water_bottle_id, 0, forcevec, [0, 0, 0], pybullet.WORLD_FRAME)
            #print('relative position', np.array(eef_state[0]) - np.array(object_state[0]), 'velocity',
            #      np.array(eef_state[-2]) - np.array(object_state[-2]))
            print('bottle', [["{0:.3f}".format(item) for item in value] for value in
                             pybullet.getBasePositionAndOrientation(water_bottle_id)]
                  + [["{0:.3f}".format(item) for item in value] for value in
                     pybullet.getBaseVelocity(water_bottle_id)])
        else:
            time.sleep(delta_t * 10)
            # free motion
            # print('Robot Trajectory finished')

            object_state = pybullet.getLinkState(water_bottle_id, 0, computeLinkVelocity=1)
            print('bottle', pybullet.getBaseVelocity(water_bottle_id)[1])
            # pybullet.removeBody(robotId)
            # removed = True
            pass
    pybullet.stepSimulation()

    tt = tt + delta_t
    if tt > trajectory_duration:
        traj_finish = True

    cur = time.perf_counter()
    while True:
        if time.perf_counter() >= cur + delta_t * slow:
            break
    if tt > 100.0:
        break
if not (video_path is None):
    pybullet.stopStateLogging(logId)
pybullet.disconnect()
