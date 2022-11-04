import argparse
import time
import math
import pickle
import numpy as np
import pybullet as p
import pybullet_data

from pathlib import Path
from sys import path
from ruckig import InputParameter, Ruckig, Trajectory, Result


# Path to the build directory including a file similar to 'ruckig.cpython-37m-x86_64-linux-gnu'.
build_path = Path(__file__).parent.absolute().parent / 'build'
path.insert(0, str(build_path))

TRAJ_GEN = True
ANIMATE = True

def main(box_position):
    # Height of target box relative to panda base, [-0.5, 0.9] is good
    z = box_position[2]
    base0 = -box_position[:2]

    # joint limit of panda, from https://frankaemika.github.io/docs/control_parameters.html
    ul = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
    ll = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])

    # initial joint position
    q0 = 0.5*(ul+ll)
    q0_dot = np.zeros(7)
    # base0 = [1.0, -1.5]
    robot_path = "robot_data/panda_5_joint_dense_1_dataset_15"
    experiment_path = "object_data/brt_gravity_only"
    g = -9.81

    clid = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
    urdf_path = "franka_panda/panda.urdf"
    robot = p.loadURDF(urdf_path, [0, 0, 0], useFixedBase=True, flags=p.URDF_USE_INERTIA_FROM_FILE)
    load_data(robot_path, experiment_path)
    # get initial guess
    q_candidates,phi_candidates,throw_candidates = brt_chunk_robot_data_matching(z)

    n_candidates = q_candidates.shape[0]

    if TRAJ_GEN:
        # get full throwing configuration and trajectories
        traj_durations = []
        trajs = []
        throw_configs = []
        st = time.time()
        for i in range(n_candidates):
            candidate_idx = i
            throw_config_full = get_full_throwing_config(robot, q_candidates[candidate_idx],
                                                         phi_candidates[candidate_idx],
                                                         throw_candidates[candidate_idx])
            # filter out throwing configuration that will hit gripper palm
            if throw_config_full[4][2] < -0.02:
                continue
            # calculate throwing trajectory
            traj_throw = get_traj_from_ruckig(q0=q0, q0_dot=q0_dot,
                                              qd=throw_config_full[0], qd_dot=throw_config_full[3],
                                              base0=base0, based =-throw_config_full[-1][:-1])
            if traj_throw.duration < 1e-10:  # unknown error
                continue
            traj_durations.append(traj_throw.duration)
            trajs.append(traj_throw)
            throw_configs.append(throw_config_full)

        print("Given query z=", "{0:0.2f}".format(z), ", found", len(throw_configs),
              "good throws in", "{0:0.2f}".format(1000 * (time.time() - st)), "ms")

        # select the minimum-time trajectory to simulate
        selected_idx = np.argmin(traj_durations)
        traj_throw = trajs[selected_idx]
        throw_config_full = throw_configs[selected_idx]

        # Other option: select the one with maximum range
        # selected_idx = np.argmin(throw_candidates[:, 0])
        # throw_config_full = get_full_throwing_config(robot, q_candidates[selected_idx],
        #                                                  phi_candidates[selected_idx],
        #                                                  throw_candidates[selected_idx])
        # traj_throw = get_traj_from_ruckig(q0=q0, q0_dot=q0_dot, qd=throw_config_full[0], qd_dot=throw_config_full[3],
        #                                       base0=base0, based =-throw_config_full[-1][:-1])
        p.disconnect()

        print("box_position: ", throw_config_full[-1])
        print("throwing range: ", "{0:0.2f}".format(-throw_config_full[2][0]),
              "throwing height", "{0:0.2f}".format(throw_config_full[2][1]))
    if ANIMATE:
        video_path=experiment_path+"/moving_base/throw"+ str(int(1000*z))+".mp4"
        throw_simulation_mobile(traj_throw, throw_config_full, g) #, video_path=video_path)

from bisect import bisect_left
def insert_idx(a, x):
    """

    :param a: sorted array, ascending
    :param x: element
    :return: the idx of the closest value to x
    """
    idx = bisect_left(a, x)
    if idx == 0:
        return idx
    elif idx == len(a):
        return idx - 1
    else:
        if (x - a[idx - 1]) < (a[idx] - x):
            return idx - 1
        else:
            return idx

robot_phis, robot_zs, brt_zs, num_gammas = None, None, None, None
mesh, robot_phi_gamma_velos_naive, robot_phi_gamma_q_idxs_naive, brt_tensor = None, None, None, None
def load_data(robot_path, brt_path, brt_tensor_name = None, brt_zs_name = None):
    global robot_phis, robot_zs, brt_zs, num_gammas, mesh, robot_phi_gamma_velos_naive, robot_phi_gamma_q_idxs_naive, brt_tensor
    st = time.time()

    # load data - robot
    # # robot_zs = np.load(robot_path + '/robot_zs.npy')
    # robot_zs = np.arange(start=0.0, stop=1.10+0.01, step=0.05)
    # # robot_gamma = np.load(robot_path + '/robot_gamma.npy')
    # robot_gamma = np.arange(start=20.0, stop=70.0+0.01, step=5.0)
    # robot_gamma *= np.pi/180.0
    # # robot_phis = np.load(robot_path + '/robot_phis.npy')
    # robot_phis = np.linspace(-90, 90, 13)
    # mesh = np.load(robot_path+'/qs.npy')
    # robot_phi_gamma_velos_naive = np.load(robot_path + '/phi_gamma_velos_naive.npy')
    # robot_phi_gamma_q_idxs_naive = np.load(robot_path + '/phi_gamma_q_idxs_naive.npy')
    # num_gammas = robot_phi_gamma_q_idxs_naive.shape[2]

    robot_zs = np.arange(0, 1.2, 0.05)
    robot_gamma = np.arange(np.pi / 9, np.pi * 7 / 18, np.pi / 36)
    robot_phis = np.arange(-np.pi / 2, np.pi / 2, np.pi / 12)
    mesh = np.load('offline/my_qs.npy')
    robot_phi_gamma_velos_naive = np.load('offline/my_vel_max.npy')
    robot_phi_gamma_q_idxs_naive = np.load('offline/my_q_idxs.npy')
    num_gammas = robot_phi_gamma_q_idxs_naive.shape[2]

    ct1 = time.time()
    print("Loading robot data cost {0:0.2f} ms".format(1000 * (ct1 - st)))
    # load data - brt
    if brt_tensor_name is not None and brt_zs_name is not None:
        brt_tensor = np.load(brt_path + brt_tensor_name)
        brt_zs = np.load(brt_path + brt_zs_name)
    else:
        # generate tensor from raw data
        brt_data = np.load(brt_path + '/brt_data.npy')

        # generate zs align to robot zs
        step_robot_zs = 0.05 # TODO
        bzstart = min(robot_zs) - step_robot_zs * np.ceil((min(robot_zs) - min(brt_data[:, 1]))/step_robot_zs)
        brt_zs = np.arange(start=bzstart, stop=max(brt_data[:, 1])+0.01, step=step_robot_zs)
        num_zs = brt_zs.shape[0]
        num_gammas = len(robot_gamma)

        brt_chunk = [[[] for j in range(num_gammas)] for i in range(num_zs)]
        states_num = 0
        for x in brt_data:
            z = x[1]
            gamma = np.arctan2(x[3], x[2])
            # drop some states
            # consider the maximum velocity robot can archive
            if gamma < min(robot_gamma) or gamma > max(robot_gamma):
                continue
            v = np.sqrt(x[2] ** 2 + x[3] ** 2)
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
            brt_tensor.shape, states_num, 100 * states_num * 5.0 / (np.prod(brt_tensor.shape))))

        ct2 = time.time()
        print("Generating brt tensor cost {0:0.2f} ms".format(1000 * (ct2 - ct1)))

    ct = time.time()
    print("Loading cost {0:0.2f} ms".format(1000 * (ct - st)))


def brt_chunk_robot_data_matching(z_target_to_base, thres=0.1):
    """

    :param thres:
    :param z_target_to_base:
    :return:
    """
    # Given target position, find out initial guesses of (q, phi, x), that is to be feed to Ruckig

    global robot_phis, robot_zs, brt_zs, num_gammas, mesh, robot_phi_gamma_velos_naive, robot_phi_gamma_q_idxs_naive, brt_tensor
    st = time.time()

    # align the z idx
    num_robot_zs = robot_zs.shape[0]
    num_brt_zs = brt_zs.shape[0]
    brt_z_min, brt_z_max = np.min(brt_zs), np.max(brt_zs)
    if z_target_to_base + brt_z_min > min(robot_zs):
        rzs_idx_start = round((z_target_to_base+brt_z_min) / 0.05)
        bzs_idx_start = 0
    else:
        rzs_idx_start = 0
        bzs_idx_start = -round((z_target_to_base+brt_z_min) / 0.05)
    if z_target_to_base + brt_z_max > max(robot_zs):
        rzs_idx_end = num_robot_zs - 1
        bzs_idx_end = num_brt_zs - 1 - round((z_target_to_base + brt_z_max - max(robot_zs)) / 0.05)
    else:
        rzs_idx_end = num_robot_zs - 1 + round((z_target_to_base + brt_z_max - max(robot_zs)) / 0.05)
        bzs_idx_end = num_brt_zs - 1
    assert bzs_idx_end - bzs_idx_start == rzs_idx_end - rzs_idx_start
    # z_num = bzs_idx_end - bzs_idx_start + 1

    # BRT-Tensor = {z, phi(length=1), gamma, brt states array, x(length=5))}
    brt_tensor = brt_tensor[bzs_idx_start:bzs_idx_end+1, ...]
    robot_tensor = np.expand_dims(robot_phi_gamma_velos_naive[rzs_idx_start: rzs_idx_end+1, ...], axis=3)
    st1 = time.time()

    validate = np.argwhere(robot_tensor - thres - brt_tensor[:, :, :, :, 4] > 0)
    # validate: z, phi, gamma, idx_of_brt

    q_indice = validate[:, :3]
    q_indice[:, 0] += rzs_idx_start
    q_candidates = mesh[robot_phi_gamma_q_idxs_naive[tuple(q_indice.T)].astype(int), :]
    phi_candidates = robot_phis[validate[:, 1]]
    x_candidates = brt_tensor[:, 0, :, :, :][tuple(np.r_['-1', validate[:, :1], validate[:, 2:4]].T)][:, :4]
    ct = time.time()
    print("Given query z=", "{0:0.2f}".format(z_target_to_base), ", found", len(q_candidates),
          "initial guesses in", "{0:0.2f}".format(1000 * (ct - st)), "ms",
          "\n\tcore operation takes {0:0.2f}".format(1000 * (ct - st1)), "ms")

    return q_candidates, phi_candidates, x_candidates

def get_full_throwing_config(robot, q, phi, throw):
    """
    Return full throwing configurations
    :param robot:
    :param q:
    :param phi:
    :param throw:
    :return:
    """
    r_throw = throw[0]
    z_throw = throw[1]
    r_dot = throw[2]
    z_dot = throw[3]

    # bullet fk
    controlled_joints = [0, 1, 2, 3, 4, 5, 6]
    p.resetJointStatesMultiDof(robot, controlled_joints, [[q0_i] for q0_i in q])
    AE =p.getLinkState(robot, 11)[0]
    q = q.tolist()
    J, _ = p.calculateJacobian(robot, 11, [0, 0, 0], q+[0.1, 0.1], [0.0]*9, [0.0]*9)
    J = np.array(J)
    J = J[:,:7]

    throwing_angle = np.arctan2(AE[1], AE[0])+math.pi*phi/180
    EB_dir = np.array([np.cos(throwing_angle), np.sin(throwing_angle)])

    J_xyz = J[:3, :]
    J_xyz_pinv = np.linalg.pinv(J_xyz)

    eef_velo = np.array([EB_dir[0]*r_dot, EB_dir[1]*r_dot, z_dot])
    q_dot = J_xyz_pinv @ eef_velo
    box_position = AE + np.array([-r_throw*EB_dir[0], -r_throw*EB_dir[1], -z_throw])

    # TODO: fix the gripper issue
    # from https://www.programcreek.com/python/example/122109/pybullet.getEulerFromQuaternion
    gripperState = p.getLinkState(robot, 11)
    gripperPos = gripperState[0]
    gripperOrn = gripperState[1]
    invGripperPos, invGripperOrn = p.invertTransform(gripperPos, gripperOrn)
    eef_velo_dir_3d = eef_velo / np.linalg.norm(eef_velo)
    tmp = AE + eef_velo_dir_3d
    blockPosInGripper, _ = p.multiplyTransforms(invGripperPos, invGripperOrn, tmp, [0, 0, 0, 1])
    velo_angle_in_eef = np.arctan2(blockPosInGripper[1], blockPosInGripper[0])

    if (velo_angle_in_eef<0.5*math.pi) and (velo_angle_in_eef>-0.5*math.pi):
        eef_angle_near = velo_angle_in_eef
    elif velo_angle_in_eef>0.5*math.pi:
        eef_angle_near = velo_angle_in_eef - math.pi
    else:
        eef_angle_near = velo_angle_in_eef + math.pi

    q[-1] = eef_angle_near
    return (q, phi, throw, q_dot, blockPosInGripper, eef_velo, AE, box_position)

def get_traj_from_ruckig(q0, q0_dot, qd, qd_dot, base0, based):
    inp = InputParameter(9)
    zeros2 = np.zeros(2)
    inp.current_position = np.concatenate((q0, base0))
    inp.current_velocity = np.concatenate((q0_dot, zeros2))
    inp.current_acceleration = np.zeros(9)

    inp.target_position = np.concatenate((qd, based))
    inp.target_velocity = np.concatenate((qd_dot, zeros2))
    inp.target_acceleration = np.zeros(9)

    inp.max_velocity = np.array([2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100, 2.0, 2.0])
    inp.max_acceleration = np.array([15, 7.5, 10, 12.5, 15, 20, 20, 5.0, 5.0]) -1.0
    inp.max_jerk = np.array([7500, 3750, 5000, 6250, 7500, 10000, 10000, 1000, 1000]) - 100

    otg = Ruckig(9)
    trajectory = Trajectory(9)
    _ = otg.calculate(inp, trajectory)
    return trajectory

def throw_simulation_mobile(trajectory, throw_config_full, g=-9.81, video_path=None):
    PANDA_BASE_HEIGHT = 0.5076438625
    box_position = throw_config_full[-1]
    clid = p.connect(p.GUI)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.resetDebugVisualizerCamera(cameraDistance=3.0, cameraYaw=160, cameraPitch=-40, cameraTargetPosition=[0.75, -0.75, 0])

    # NOTE: need high frequency
    hz = 1000
    delta_t = 1.0 / hz
    p.setGravity(0, 0, g)
    p.setTimeStep(delta_t)
    p.setRealTimeSimulation(0)

    AE = throw_config_full[-2]
    EB = box_position - AE

    controlled_joints = [3, 4, 5, 6, 7, 8, 9]
    gripper_joints = [12, 13]
    numJoints = len(controlled_joints)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    robotEndEffectorIndex = 14
    robotId = p.loadURDF("descriptions/rbkairos_description/robots/rbkairos_panda_hand.urdf", [-box_position[0], -box_position[1], 0], useFixedBase=True)

    planeId = p.loadURDF("plane.urdf", [0, 0, 0.0])
    soccerballId = p.loadURDF("soccerball.urdf", [-3.0, 0, 3], globalScaling=0.05)
    boxId = p.loadURDF("descriptions/robot_descriptions/objects_description/objects/box.urdf",
                       [0, 0, PANDA_BASE_HEIGHT+box_position[2]],
                       globalScaling=0.5)
    p.changeDynamics(soccerballId, -1, mass=1.0, linearDamping=0.00, angularDamping=0.00, rollingFriction=0.03,
                     spinningFriction=0.03, restitution=0.2, lateralFriction=0.03)
    p.changeDynamics(planeId, -1, restitution=0.9)
    p.changeDynamics(robotId, gripper_joints[0], jointUpperLimit=100)
    p.changeDynamics(robotId, gripper_joints[1], jointUpperLimit=100)

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
    p.resetBasePositionAndOrientation(robotId, np.append(base_traj_data[0,0], 0.0), [0, 0, 0,1])
    p.resetJointStatesMultiDof(robotId, controlled_joints, [[q0_i] for q0_i in q0])
    eef_state = p.getLinkState(robotId, robotEndEffectorIndex, computeLinkVelocity=1)
    p.resetBasePositionAndOrientation(soccerballId, eef_state[0], [0, 0, 0, 1])
    p.resetJointState(robotId, gripper_joints[0], 0.03)
    p.resetJointState(robotId, gripper_joints[1], 0.03)
    tt = 0
    flag = True
    if not (video_path is None):
        logId = p.startStateLogging(loggingType=p.STATE_LOGGING_VIDEO_MP4, fileName=video_path)
    while(True):
        if flag:
            ref_full = trajectory.at_time(tt)
            ref = [ref_full[i][:7] for i in range(3)]
            ref_base = [ref_full[i][-2:] for i in range(3)]
            p.resetJointStatesMultiDof(robotId, controlled_joints, [[q0_i] for q0_i in ref[0]], targetVelocities=[[q0_i] for q0_i in ref[1]])
            p.resetBasePositionAndOrientation(robotId, np.append(ref_base[0], 0.0), [0, 0, 0, 1])
        else:
            ref_full = trajectory.at_time(plan_time)
            ref = [ref_full[i][:7] for i in range(3)]
            ref_base = [ref_full[i][-2:] for i in range(3)]
            p.resetJointStatesMultiDof(robotId, controlled_joints, [[q0_i] for q0_i in ref[0]])
            p.resetBasePositionAndOrientation(robotId, np.append(ref_base[0], 0.0), [0, 0, 0, 1])
        if tt > plan_time - 1*delta_t:
            p.resetJointState(robotId, gripper_joints[0], 0.05)
            p.resetJointState(robotId, gripper_joints[1], 0.05)
        else:
            eef_state = p.getLinkState(robotId, robotEndEffectorIndex, computeLinkVelocity=1)
            p.resetBasePositionAndOrientation(soccerballId, eef_state[0], [0, 0, 0, 1])
            p.resetBaseVelocity(soccerballId, linearVelocity=eef_state[-2])
        p.stepSimulation()
        tt = tt + delta_t
        if tt > trajectory.duration:
            flag = False
        time.sleep(delta_t)
        if tt > 6.0:
            break
    if not (video_path is None):
        p.stopStateLogging(logId)
    p.disconnect()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Overall
    parser.add_argument('--box_x', type=float, required=False, default=-3, help="box x position in panda frame")
    parser.add_argument('--box_y', type=float, required=False, default=3, help="box y position in panda frame")
    parser.add_argument('--box_z', type=float, required=False, default=0, help="box z position in panda frame")

    ARGS = parser.parse_args()

    box_position = np.array([ARGS.box_x, ARGS.box_y, ARGS.box_z])
    print(box_position)
    main(box_position)