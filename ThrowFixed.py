import argparse
import time
import math
import numpy as np
import pybullet
import pybullet_data

from pathlib import Path
from sys import path
from ruckig import InputParameter, Ruckig, Trajectory, Result


## Path to the build directory including a file similar to 'ruckig.cpython-37m-x86_64-linux-gnu'.
# build_path = Path(__file__).parent.absolute().parent / 'build'
# path.insert(0, str(build_path))


class ThrowFixed():
    def __init__(self, q_ul, q_ll, robot_path, brt_path,
                 brt_tensor_name=None, brt_zs_name=None,
                 gravity=-9.81, urdf_path="franka_panda/panda.urdf"):
        self.ul = q_ul
        self.ll = q_ll
        # initial joint position
        self.q0 = 0.5 * (self.ul + self.ll)
        self.q0_dot = np.zeros(7)
        self.robot_path = robot_path
        self.brt_path = brt_path
        self.g = gravity
        self.GAMMA_TOLERANCE = 0.2/180.0*np.pi  # 0.2 degree
        self.Z_TOLERANCE = 0.01  # 1cm
        clid = pybullet.connect(pybullet.DIRECT)
        pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
        self.robot = pybullet.loadURDF(urdf_path, [0, 0, 0], useFixedBase=True,
                                       flags=pybullet.URDF_USE_INERTIA_FROM_FILE)

    def solve(self, box_position, animate=False, video_path=None):
        base0 = -box_position[:2]
        q_candidates, phi_candidates, x_candidates = self.brt_robot_data_matching(box_position)
        if len(q_candidates) == 0:
            print("No result found")
            return 0
        trajs, throw_configs = self.generate_throw_config(q_candidates, phi_candidates, x_candidates, base0)
        if len(trajs) == 0:
            print("No trajectory found")
            return 0

        # select the minimum-time trajectory to simulate
        traj_durations = [traj.duration for traj in trajs]
        selected_idx = np.argmin(traj_durations)
        traj_throw = trajs[selected_idx]
        throw_config_full = throw_configs[selected_idx]

        print("box_position: ", box_position)
        print("AB          : ", throw_config_full[-1])
        print("deviation   : ", throw_config_full[-1] - box_position)
        print("\n\tthrowing range: {0:0.2f}".format(-throw_config_full[2][0]),
              "\n\tthrowing height: {0:0.2f}".format(throw_config_full[2][1]))
        if animate:
            pybullet.disconnect()
            self.throw_simulation_mobile(traj_throw, throw_config_full, self.g, video_path=video_path)
        return len(trajs)

    def load_data(self, brt_tensor_name=None, brt_zs_name=None):
        st = time.time()

        # load data - robot
        # TODO: data files
        self.robot_zs = np.load(self.robot_path + '/robot_zs.npy')
        # self.robot_zs = np.arange(0, 1.2, 0.05)
        self.robot_dis = np.load(self.robot_path + '/robot_diss.npy')
        # self.robot_dis = np.arange(0, 1.1, 0.05)
        self.robot_phis = np.load(self.robot_path + '/robot_phis.npy')
        self.robot_gamma = np.load(self.robot_path + '/robot_gammas.npy')
        # self.robot_gamma = np.arange(start=20.0, stop=70.0 + 0.01, step=5.0)
        # self.robot_gamma *= np.pi / 180.0
        self.num_gammas = len(self.robot_gamma)
        # self.robot_phis = np.linspace(-90, 90, 15)
        self.mesh = np.load(self.robot_path + '/q_idx_qs.npy')
        self.robot_phi_gamma_velos_naive = np.load(self.robot_path + '/z_dis_phi_gamma_vel_max.npy')
        self.robot_phi_gamma_q_idxs_naive = np.load(self.robot_path + '/z_dis_phi_gamma_vel_max_q_idxs.npy')
        self.ae = np.load(self.robot_path + '/q_idx_ae.npy')
        """
        self.ae = np.zeros(list(self.robot_phi_gamma_q_idxs_naive.shape) + [3])
        for i in range(self.robot_phi_gamma_q_idxs_naive.shape[0]):
            for j in range(self.robot_phi_gamma_q_idxs_naive.shape[1]):
                for k in range(self.robot_phi_gamma_q_idxs_naive.shape[2]):
                    controlled_joints = [0, 1, 2, 3, 4, 5, 6]
                    pybullet.resetJointStatesMultiDof(self.robot, controlled_joints,
                                                      [[qi] for qi in self.mesh[
                                                          int(self.robot_phi_gamma_q_idxs_naive[i, j, k])]])
                    self.ae[i, j, k, :] = pybullet.getLinkState(self.robot, 11)[0]
        """

        self.zs_step = 0.05  # TODO
        assert self.num_gammas == self.robot_phi_gamma_q_idxs_naive.shape[3]

        ct1 = time.time()
        print("Loading robot data cost {0:0.2f} ms".format(1000 * (ct1 - st)))

        # load data - brt
        if brt_tensor_name is not None and brt_zs_name is not None:
            self.brt_tensor = np.load(self.brt_path + brt_tensor_name)
            self.brt_zs = np.load(self.brt_path + brt_zs_name)
        else:  # generate tensor from raw data
            brt_data = np.load(self.brt_path + '/brt_data.npy')

            # generate zs align to robot zs
            bzstart = min(self.robot_zs) - \
                      self.zs_step * np.ceil((min(self.robot_zs) - min(brt_data[:, 1])) / self.zs_step)
            brt_zs = np.arange(start=bzstart, stop=max(brt_data[:, 1]) + 0.01, step=self.zs_step)
            num_zs = brt_zs.shape[0]

            # generate brt_chunk
            brt_chunk = [[[] for j in range(self.num_gammas)] for i in range(num_zs)]
            states_num = 0
            pad_gamma = np.r_[-np.inf, self.robot_gamma]
            pad_zs = np.r_[-np.inf, brt_zs]
            for x in brt_data:
                z = x[1]
                gamma = np.arctan2(x[3], x[2])
                # drop some states
                # consider the maximum velocity robot can archive
                if gamma < min(self.robot_gamma) or gamma > max(self.robot_gamma):
                    continue
                v = np.sqrt(x[2] ** 2 + x[3] ** 2)
                # if v > max: continue
                # argmax will be faster than where
                gi = np.argmax(abs(pad_gamma - gamma) < self.GAMMA_TOLERANCE)
                if gi == 0: continue
                zi = np.argmax(abs(pad_zs - z) < self.Z_TOLERANCE)
                if zi == 0: continue
                brt_chunk[zi-1][gi-1].append(list(x) + [v])
                states_num += 1
            # delete empty chunks
            remove_i = 0
            while True:
                chunk = brt_chunk[remove_i]
                empty = True
                for j in range(self.num_gammas):
                    if len(chunk[j]) > 0:
                        empty = False
                        break
                if not empty:
                    break
                remove_i += 1
            brt_chunk = brt_chunk[remove_i:]
            self.brt_zs = brt_zs[remove_i:, ...]
            num_zs -= remove_i

            # transform brt_chunk into brt_tensor
            brt_tensor = []
            l = 0
            while True:
                new_layer_brt = np.ones((num_zs, self.num_gammas, 5))
                stillhasvalue = False
                for i in range(num_zs):
                    for k in range(self.num_gammas):
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
            # [z, dis, phi, gamma, index] -> x {r, z, r_dot, z_dot, v}
            self.brt_tensor = np.expand_dims(brt_tensor, axis=(1, 2))  # insert Phi dimension

            ct2 = time.time()
            print("Tensor Size: {0} with {1} states( occupation rate {2:0.1f}%)".format(
                brt_tensor.shape, states_num, 100 * states_num / (np.prod(brt_tensor.shape[:-1]))))
            print("Generating brt tensor cost {0:0.2f} ms".format(1000 * (ct2 - ct1)))

        ct = time.time()
        print("Loading cost {0:0.2f} ms".format(1000 * (ct - st)))

    def brt_robot_data_matching(self, box_position, thres_v=0.1, thres_dis=0.01, thres_phi=0.04):
        """
        Given target position, find out initial guesses of (q, phi, x), that is to be feed to Ruckig
        :param box_position:
        :param thres_dis:
        :param thres_v:
        :return: candidates of q, phi, x
        """
        st = time.time()
        z_target_to_base = box_position[-1]
        AB = box_position[:2]

        # align the z idx
        num_robot_zs = self.robot_zs.shape[0]
        num_brt_zs = self.brt_zs.shape[0]
        brt_z_min, brt_z_max = np.min(self.brt_zs), np.max(self.brt_zs)
        if z_target_to_base + brt_z_min > min(self.robot_zs):
            rzs_idx_start = round((z_target_to_base + brt_z_min) / self.zs_step)
            bzs_idx_start = 0
        else:
            rzs_idx_start = 0
            bzs_idx_start = -round((z_target_to_base + brt_z_min) / self.zs_step)
        if z_target_to_base + brt_z_max > max(self.robot_zs):
            rzs_idx_end = num_robot_zs - 1
            bzs_idx_end = num_brt_zs - 1 - round((z_target_to_base + brt_z_max - max(self.robot_zs)) / self.zs_step)
        else:
            rzs_idx_end = num_robot_zs - 1 + round((z_target_to_base + brt_z_max - max(self.robot_zs)) / self.zs_step)
            bzs_idx_end = num_brt_zs - 1
        assert bzs_idx_end - bzs_idx_start == rzs_idx_end - rzs_idx_start, \
            "bzs: {0}, {1}; rzs: {2}, {3}".format(bzs_idx_start, bzs_idx_end, rzs_idx_start, rzs_idx_end)
        z_num = bzs_idx_end - bzs_idx_start + 1
        if z_num == 0 or rzs_idx_end <= 0:
            return [], [], []

        # BRT-Tensor = {z, dis(length=1), phi(length=1), gamma, idx} -> brt state,
        # x = {r, z, r_dot, z_dot, speed}(length=5)
        brt_tensor = self.brt_tensor[bzs_idx_start:bzs_idx_end + 1, ...]

        # Fixed-base limitation
        # Robot-Tensor = {z, dis, phi, gamma} -> max_v
        robot_tensor_v = np.expand_dims(self.robot_phi_gamma_velos_naive[rzs_idx_start: rzs_idx_end + 1, ...], axis=4)
        # 1. filter on dis:
        # we can only throw 'out'
        b = np.linalg.norm(AB)
        robot_tensor_v = robot_tensor_v[:, np.where(self.robot_dis < b)[0], ...]

        # 2. calculate desired r
        cos_phi = np.cos(self.robot_phis)
        # [dis, phi] -> r
        dcosphi = self.robot_dis[self.robot_dis < b, np.newaxis] @ cos_phi[np.newaxis, :]  # TODO: this can be stored
        # TODO: It's better to use real r to control the error
        r = np.sqrt(b**2 - self.robot_dis[self.robot_dis < b, None]**2 + dcosphi**2) - dcosphi
        r_tensor = r[None, :, :, None, None]
        mask_r = abs(-brt_tensor[:, :, :, :, :, 0] - r_tensor) < thres_dis
        # TODO: re-compute BRT states that
        # brt_tensor[:, :, :, :, :, 0] - r > 0

        st1 = time.time()

        validate = np.argwhere((robot_tensor_v - thres_v - brt_tensor[:, :, :, :, :, 4] > 0)  # velocity satisfy
                               * mask_r
                               )
                            # TODO: re-compute BRT states that
                            # robot_tensor_r < -brt_tensor[:, :, :, :, 0]
        # validate: z, dis, phi, gamma, idx_of_brt

        q_indices = np.copy(validate[:, :4])
        q_indices[:, 0] += rzs_idx_start
        qids = self.robot_phi_gamma_q_idxs_naive[tuple(q_indices.T)].astype(int)
        q_candidates = self.mesh[qids, :]
        q_ae = self.ae[qids]
        phi_candidates = self.robot_phis[validate[:, 2]]
        x_candidates = brt_tensor[:, 0, 0, :, :, :][tuple(np.r_['-1', validate[:, :1], validate[:, 3:5]].T)][:, :4]
        # TODO: unknown error
        error_index = np.nonzero(np.sum(np.isnan(x_candidates), axis=1))
        if error_index[0].shape[0] > 0:
            print("--error!!!")
            a = input("input to continue")
            q_candidates = np.delete(q_candidates, error_index, axis=0)
            q_ae = np.delete(q_ae, error_index, axis=0)
            phi_candidates = np.delete(phi_candidates, error_index, axis=0)
            x_candidates = np.delete(x_candidates, error_index, axis=0)
        # compensate alpha
        beta = np.arctan2(AB[1], AB[0])
        dis = np.linalg.norm(q_ae[:, :2], axis=1)
        alpha = -np.arccos(np.clip((dis-x_candidates[:, 0]*np.cos(phi_candidates))/b, -1, 1))*np.sign(phi_candidates) + beta
        AE_alpha = np.arctan2(q_ae[:, 1], q_ae[:, 0])
        q_candidates[:, 0] += alpha - AE_alpha
        q_candidates[q_candidates[:, 0] > np.pi, 0] -= 2*np.pi
        q_candidates[q_candidates[:, 0] < -np.pi, 0] += 2*np.pi

        ct = time.time()
        print("Given query box_position= {0}, \n\tfound {1} initial guesses in {2:0.2f} ms".format(
            box_position, len(q_candidates), 1000 * (ct - st)),
            "\n\tcore operation takes {0:0.2f} ms".format(1000 * (ct - st1)))

        return q_candidates, phi_candidates, x_candidates

    def generate_throw_config(self, q_candidates, phi_candidates, x_candidates, base0):
        n_candidates = q_candidates.shape[0]

        # get full throwing configuration and trajectories
        traj_durations = []
        trajs = []
        throw_configs = []
        st = time.time()
        num_outlimit, num_hit, num_ruckigerr, num_small_deviation = 0, 0, 0, 0
        for i in range(n_candidates):
            candidate_idx = i
            # q limit -- just check joint 0
            q0 = q_candidates[candidate_idx][0]
            if q0 > self.ul[0] or q0 < self.ll[0]:
                num_outlimit += 1
                continue

            throw_config_full = self.get_full_throwing_config(self.robot, q_candidates[candidate_idx],
                                                              phi_candidates[candidate_idx],
                                                              x_candidates[candidate_idx])
            # filter out throwing configuration that will hit gripper palm
            if throw_config_full[4][2] < -0.02:
                num_hit += 1
                continue
            # calculate throwing trajectory
            traj_throw = self.get_traj_from_ruckig(q0=self.q0, q0_dot=self.q0_dot,
                                                   qd=throw_config_full[0], qd_dot=throw_config_full[3],
                                                   base0=base0, based=-throw_config_full[-1][:-1])
            if traj_throw.duration < 1e-10:  # unknown error
                num_ruckigerr += 1
                continue
            deviation = throw_config_full[-1][:2] + base0
            if np.linalg.norm(deviation) < 0.01:
                num_small_deviation += 1
            #     continue
            traj_durations.append(traj_throw.duration)
            trajs.append(traj_throw)
            throw_configs.append(throw_config_full)

        print("Found {} good throws in {:0.2f} ms".format(len(throw_configs), 1000 * (time.time() - st)))
        print("\t\t out of joint limit: {}, hit the palm: {}, ruckig error: {}".format(
            num_outlimit, num_hit, num_ruckigerr))
        print("\t\t No. solution with small deviation", num_small_deviation)
        return trajs, throw_configs

    @staticmethod
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
        pybullet.resetJointStatesMultiDof(robot, controlled_joints, [[q0_i] for q0_i in q])
        AE = pybullet.getLinkState(robot, 11)[0]
        q = q.tolist()
        J, _ = pybullet.calculateJacobian(robot, 11, [0, 0, 0], q + [0.1, 0.1], [0.0] * 9, [0.0] * 9)
        J = np.array(J)
        J = J[:, :7]

        throwing_angle = np.arctan2(AE[1], AE[0]) + phi
        EB_dir = np.array([np.cos(throwing_angle), np.sin(throwing_angle)])

        J_xyz = J[:3, :]
        J_xyz_pinv = np.linalg.pinv(J_xyz)

        eef_velo = np.array([EB_dir[0] * r_dot, EB_dir[1] * r_dot, z_dot])
        q_dot = J_xyz_pinv @ eef_velo
        box_position = AE + np.array([-r_throw * EB_dir[0], -r_throw * EB_dir[1], -z_throw])

        # TODO: fix the gripper issue
        # from https://www.programcreek.com/python/example/122109/pybullet.getEulerFromQuaternion
        gripperState = pybullet.getLinkState(robot, 11)
        gripperPos = gripperState[0]
        gripperOrn = gripperState[1]
        invGripperPos, invGripperOrn = pybullet.invertTransform(gripperPos, gripperOrn)
        eef_velo_dir_3d = eef_velo / np.linalg.norm(eef_velo)
        tmp = AE + eef_velo_dir_3d
        blockPosInGripper, _ = pybullet.multiplyTransforms(invGripperPos, invGripperOrn, tmp, [0, 0, 0, 1])
        velo_angle_in_eef = np.arctan2(blockPosInGripper[1], blockPosInGripper[0])

        if (velo_angle_in_eef < 0.5 * math.pi) and (velo_angle_in_eef > -0.5 * math.pi):
            eef_angle_near = velo_angle_in_eef
        elif velo_angle_in_eef > 0.5 * math.pi:
            eef_angle_near = velo_angle_in_eef - math.pi
        else:
            eef_angle_near = velo_angle_in_eef + math.pi

        q[-1] = eef_angle_near
        return (q, phi, throw, q_dot, blockPosInGripper, eef_velo, AE, box_position)

    @staticmethod
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
        inp.max_acceleration = np.array([15, 7.5, 10, 12.5, 15, 20, 20, 5.0, 5.0]) - 1.0
        inp.max_jerk = np.array([7500, 3750, 5000, 6250, 7500, 10000, 10000, 1000, 1000]) - 100

        otg = Ruckig(9)
        trajectory = Trajectory(9)
        _ = otg.calculate(inp, trajectory)
        return trajectory

    @staticmethod
    def throw_simulation_mobile(trajectory, throw_config_full, g=-9.81, video_path=None):
        PANDA_BASE_HEIGHT = 0.5076438625
        box_position = throw_config_full[-1]
        clid = pybullet.connect(pybullet.GUI)
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0)
        pybullet.resetDebugVisualizerCamera(cameraDistance=3.0, cameraYaw=160, cameraPitch=-40,
                                            cameraTargetPosition=[0.75, -0.75, 0])

        # NOTE: need high frequency
        hz = 1000
        delta_t = 1.0 / hz
        pybullet.setGravity(0, 0, g)
        pybullet.setTimeStep(delta_t)
        pybullet.setRealTimeSimulation(0)

        AE = throw_config_full[-2]
        EB = box_position - AE

        controlled_joints = [3, 4, 5, 6, 7, 8, 9]
        gripper_joints = [12, 13]
        numJoints = len(controlled_joints)
        pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
        robotEndEffectorIndex = 14
        robotId = pybullet.loadURDF("descriptions/rbkairos_description/robots/rbkairos_panda_hand.urdf",
                                    [-box_position[0], -box_position[1], 0], useFixedBase=True)

        planeId = pybullet.loadURDF("plane.urdf", [0, 0, 0.0])
        soccerballId = pybullet.loadURDF("soccerball.urdf", [-3.0, 0, 3], globalScaling=0.05)
        boxId = pybullet.loadURDF("descriptions/robot_descriptions/objects_description/objects/box.urdf",
                                  [0, 0, PANDA_BASE_HEIGHT + box_position[2]],
                                  globalScaling=0.5)
        pybullet.changeDynamics(soccerballId, -1, mass=1.0, linearDamping=0.00, angularDamping=0.00,
                                rollingFriction=0.03,
                                spinningFriction=0.03, restitution=0.2, lateralFriction=0.03)
        pybullet.changeDynamics(planeId, -1, restitution=0.9)
        pybullet.changeDynamics(robotId, gripper_joints[0], jointUpperLimit=100)
        pybullet.changeDynamics(robotId, gripper_joints[1], jointUpperLimit=100)

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
        pybullet.resetBasePositionAndOrientation(soccerballId, eef_state[0], [0, 0, 0, 1])
        pybullet.resetJointState(robotId, gripper_joints[0], 0.03)
        pybullet.resetJointState(robotId, gripper_joints[1], 0.03)
        tt = 0
        flag = True
        if not (video_path is None):
            logId = pybullet.startStateLogging(loggingType=pybullet.STATE_LOGGING_VIDEO_MP4, fileName=video_path)
        while (True):
            if flag:
                ref_full = trajectory.at_time(tt)
                ref = [ref_full[i][:7] for i in range(3)]
                ref_base = [ref_full[i][-2:] for i in range(3)]
                pybullet.resetJointStatesMultiDof(robotId, controlled_joints, [[q0_i] for q0_i in ref[0]],
                                                  targetVelocities=[[q0_i] for q0_i in ref[1]])
                pybullet.resetBasePositionAndOrientation(robotId, np.append(ref_base[0], 0.0), [0, 0, 0, 1])
            else:
                ref_full = trajectory.at_time(plan_time)
                ref = [ref_full[i][:7] for i in range(3)]
                ref_base = [ref_full[i][-2:] for i in range(3)]
                pybullet.resetJointStatesMultiDof(robotId, controlled_joints, [[q0_i] for q0_i in ref[0]])
                pybullet.resetBasePositionAndOrientation(robotId, np.append(ref_base[0], 0.0), [0, 0, 0, 1])
            if tt > plan_time - 1 * delta_t:
                pybullet.resetJointState(robotId, gripper_joints[0], 0.05)
                pybullet.resetJointState(robotId, gripper_joints[1], 0.05)
            else:
                eef_state = pybullet.getLinkState(robotId, robotEndEffectorIndex, computeLinkVelocity=1)
                pybullet.resetBasePositionAndOrientation(soccerballId, eef_state[0], [0, 0, 0, 1])
                pybullet.resetBaseVelocity(soccerballId, linearVelocity=eef_state[-2])
            pybullet.stepSimulation()
            tt = tt + delta_t
            if tt > trajectory.duration:
                flag = False
            time.sleep(delta_t)
            if tt > 6.0:
                break
        if not (video_path is None):
            pybullet.stopStateLogging(logId)
        pybullet.disconnect()

    def boundary(self, npoints: int, scale=np.array([2, 2, 2]), scale_low=None):
        """
        Find the boundary of the position where there exist a solution
        :param npoints: number of points
        :param scale: start with the position limit
        :param scale_low: lower bound
        :return:
        """
        if scale_low is None:
            scale_low = -scale
        pts = np.random.uniform(scale_low, scale, [npoints, 3])
        n = np.array([self.solve(pt) for pt in pts])

        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        success = np.where(n > 0)[0]
        failed = np.setdiff1d(np.arange(npoints), success)
        scatter_plot = ax.scatter3D(pts[success, 0], pts[success, 1], pts[success, 2], c=n[success], alpha=0.6, cmap=plt.get_cmap('summer'))
        plt.colorbar(scatter_plot)
        ax.scatter3D(pts[failed, 0], pts[failed, 1], pts[failed, 2], c='red', alpha=0.2)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        success_pts = pts[success]
        # most far away one
        id_far = np.argmax(np.linalg.norm(success_pts[:, :2]))
        print("The most far away one: ", success_pts[id_far])
        # highest
        print("The highest one: ", success_pts[np.argmax(success_pts[:, 2])])

        plt.show()

def plot_solution_num_distribution(manipulator, nx, ny, nz):
    x, y, z = np.linspace(-2.0, 2.0+1e-10, nx), np.linspace(-2.0, 2.0+1e-10, ny), np.linspace(-2.0, 1.0+1e-10, nz)
    xx, yy, zz = np.meshgrid(x, y, z)
    n = np.array([[[manipulator.solve(np.array([xx[i, j, k], yy[i, j, k], zz[i, j, k]])) \
                    for k in range(nz)] for j in range(ny)] for i in range(nx)])
    np.save('num_solution', n)
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    levels = np.linspace(n.min(), n.max(), 100)
    for i in range(nz):
        scatter_plot = ax.contourf(x, y, n[:, :, i], alpha=0.4, levels=levels, zdir='z', offset=z[i], cmap=plt.get_cmap('rainbow'))

    #scatter_plot = ax.scatter3D(xx[idx], yy[idx], zz[idx], c=n[idx], alpha=0.6)
    # , cmap=plt.get_cmap('spring'))
    plt.colorbar(scatter_plot)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_zlim3d(z.min(), z.max())
    plt.title("Number of solutions")
    plt.show()

if __name__ == "__main__":
    manipulator = ThrowFixed(np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]),
                             np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]),
                             "robot_data/panda_5_joint_fix_0.3",
                             "object_data/brt_gravity_only")
    manipulator.load_data()

    plot_solution_num_distribution(manipulator, 21, 21, 11)

    # manipulator.solve(np.array([-1.2, -0.4,  0.7]), animate=True)#, video_path="./fixed-sample.mp4")

    """
    """

    """
    while True:
        # input box position
        print("Input query box position\n")
        x = input("x(default=-1):")
        if x == "":
            x = -3.0
        else:
            x = float(x)
        y = input("y(default=1):")
        if y == "":
            y = 3.0
        else:
            y = float(y)
        z = input("z(default=0):")
        if z == "":
            z = 0.0
        else:
            z = float(z)
        manipulator.solve(box_position=np.array([x, y, z]))
        if input("Press Q to escape").upper() == "Q":
            break
    """