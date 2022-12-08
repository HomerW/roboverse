from abc import abstractmethod
import numpy as np
from PIL import Image
import gym

import roboverse.bullet as bullet
from roboverse.bullet.control import deg_to_quat, quat_to_deg
from roboverse.bullet.serializable import Serializable
from roboverse.bullet.sawyer.sawyer_queries import has_fixed_root, format_sim_query, get_index_by_attribute, get_link_state
from roboverse.bullet.sawyer.sawyer_ik import sawyer_position_ik, step_ik, position_control
from roboverse.envs import objects

class SawyerEnv(gym.Env, Serializable):

    def __init__(self,
                 DoF=4,
                 gui=False,

                 action_scale=0.05,
                 action_repeat=10,
                 timestep=1./240,
                 solver_iterations=150,
                 ddeg_scale=5,
                 gripper_bounds=[-1, 1],

                 pos_init = [0.6, -0.15, -0.2],
                 pos_low = [0.45, -0.25, -.36],
                 pos_high = [0.85, 0.25, -0.1],
                 quat_init = bullet.deg_to_quat([180, 0, 0]),
                 max_force=100.,

                 env_obs_img_dim=196,
                 obs_img_dim=48,
                 downsample=True,
                 transpose_image=False,

                 
                 ):
        assert DoF in [4]

        self.DoF = DoF
        self.gui = gui

        self._action_scale = action_scale
        self._action_repeat = action_repeat
        self._timestep = timestep
        self._solver_iterations = solver_iterations
        self._ddeg_scale = ddeg_scale
        self._gripper_bounds = gripper_bounds

        self._pos_init = pos_init
        self._pos_low = pos_low
        self._pos_high = pos_high
        self._quat_init = quat_init
        self._max_force = max_force

        self._id = 'SawyerBaseEnv'
        self._uid = bullet.connect_headless(self.gui)

        self._transpose_image = transpose_image
        self.env_obs_img_dim = env_obs_img_dim
        self.obs_img_dim = obs_img_dim
        self._downsample = downsample
        self.image_shape = (obs_img_dim, obs_img_dim)
        self.image_length = np.prod(self.image_shape) * 3
        self._projection_matrix_obs = bullet.get_projection_matrix(
            self.obs_img_dim, self.obs_img_dim)

        self._view_matrix_obs = bullet.get_view_matrix(
            target_pos=[0.7, 0, -0.25], distance=0.5,
            yaw=90, pitch=-27, roll=0, up_axis_index=2)

        self._set_action_space()
        self._set_observation_space()
    
    def _load_meshes(self):
        self.robot_id = objects.sawyer()
        self.table_id = objects.table()
        self.objects = {}
        self.end_effector_id = get_index_by_attribute(
            self.robot_id, 'link_name', 'gripper_site')

    def _reset_gripper(self):
        init_pos = np.array(self._pos_init)
        init_theta = np.array(self._quat_init)

        position_control(self.robot_id,
                            self.end_effector_id,
                            init_pos,
                            init_theta)

    def _format_state_query(self):
        ## position and orientation of body root
        bodies = [v for k, v in self.objects.items(
        ) if not has_fixed_root(v)]
        ## position and orientation of link
        links = [(self.robot_id, self.end_effector_id)]
        ## position and velocity of prismatic joint
        joints = [(self.robot_id, None)]
        self._state_query = format_sim_query(
            bodies, links, joints)

    def reset(self):
        bullet.reset()
        bullet.setup_headless()
        self._load_meshes()
        self._format_state_query()
        self._reset_gripper()

        return self.get_observation()
    
    def _format_action(self, *action):
        if len(action) == 1:
            action = np.clip(action[0], a_min=-1, a_max=1)
            delta_pos, delta_yaw, gripper = action[:3], action[3:4], action[-1]
        elif len(action) == 3:
            action[0] = np.clip(action[0], a_min=-1, a_max=1)
            action[1] = np.clip(action[1], a_min=-1, a_max=1)
            action[2] = np.clip(action[2], a_min=-1, a_max=1)
            delta_pos, delta_yaw, gripper = action[0], action[1], action[2]
        else:
            raise RuntimeError('Unrecognized action: {}'.format(action))

        delta_angle = [0, 0, delta_yaw[0]]
        return np.array(delta_pos), np.array(delta_angle), gripper

    def _simulate(self, pos, theta, gripper):
        for _ in range(self._action_repeat):
            sawyer_position_ik(
                self.robot_id, self.end_effector_id,
                pos, theta,
                gripper, gripper_bounds=self._gripper_bounds,
                discrete_gripper=False, max_force=self._max_force
            )
            step_ik(body=self.robot_id)

    def step(self, *action):
        # Get positional information
        pos = get_link_state(
            self.robot_id, self.end_effector_id, 'pos')
        curr_angle = get_link_state(
            self.robot_id, self.end_effector_id, 'theta')
        default_angle = quat_to_deg(self._quat_init)

        # Keep necessary degrees of theta fixed
        angle = np.append(default_angle[:2], [curr_angle[2]])

        # If angle is part of action, use it
        delta_pos, delta_angle, gripper = self._format_action(*action)
        angle += delta_angle * self._ddeg_scale

        # Update position and theta
        pos += delta_pos * self._action_scale
        pos = np.clip(pos, self._pos_low, self._pos_high)
        theta = deg_to_quat(angle)
        self._simulate(pos, theta, gripper)

        # Get tuple information
        observation = self.get_observation()
        info = self.get_info()
        reward = self.get_reward(info)
        done = False

        return observation, reward, done, info

    def _get_end_effector_pos(self):
        return get_link_state(self.robot_id, self.end_effector_id, 'pos')

    def _get_end_effector_theta(self):
        return get_link_state(self.robot_id, self.end_effector_id, 'theta')

    def get_observation(self):
        ## Gripper State
        left_tip_pos = get_link_state(
            self.robot_id, 'right_gripper_l_finger_joint', keys='pos')
        right_tip_pos = get_link_state(
            self.robot_id, 'right_gripper_r_finger_joint', keys='pos')
        left_tip_pos = np.asarray(left_tip_pos)
        right_tip_pos = np.asarray(right_tip_pos)
        gripper_tips_distance = [np.linalg.norm(
            left_tip_pos - right_tip_pos)]
        hand_theta = get_link_state(self.robot_id, self.end_effector_id,
                                           'theta', quat_to_deg=False)
        end_effector_pos = self._get_end_effector_pos()
        gripper_state = np.concatenate((
            end_effector_pos, hand_theta, gripper_tips_distance,
        ))

        ## Image
        image_observation = self.render_obs()

        ## Observation
        observation = {
            'state': gripper_state,
            'image': image_observation,
        }

        return observation

    def get_reward(self, info):
        return 0

    def get_info(self):
        return {}

    def render_obs(self):
        img, depth, segmentation = bullet.render(
            self.env_obs_img_dim, self.env_obs_img_dim, self._view_matrix_obs,
            self._projection_matrix_obs, shadow=0)

        if self._downsample:
            im = Image.fromarray(np.uint8(img), 'RGB').resize(
                self.image_shape, resample=Image.ANTIALIAS)
            img = np.array(im)
        if self._transpose_image:
            img = np.transpose(img, (2, 0, 1))
        return img

    def _set_action_space(self):
        self.action_dim = self.DoF + 1
        act_bound = 1
        act_high = np.ones(self.action_dim) * act_bound
        self.action_space = gym.spaces.Box(-act_high, act_high)

    def _set_observation_space(self):
        img_space = gym.spaces.Box(0, 1, (self.image_length,),
                                    dtype=np.float32)
        robot_state_dim = 8  # XYZ + QUAT + GRIPPER_STATE
        obs_bound = 100
        obs_high = np.ones(robot_state_dim) * obs_bound
        state_space = gym.spaces.Box(-obs_high, obs_high)

        spaces = {'image': img_space, 
                    'state': state_space}
        self.observation_space = gym.spaces.Dict(spaces)

    def close(self):
        bullet.disconnect()