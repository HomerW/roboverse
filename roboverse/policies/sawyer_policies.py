import numpy as np

import roboverse.bullet as bullet
from roboverse.bullet.drawer_utils import get_drawer_handle_pos
from roboverse.bullet.sawyer.sawyer_queries import get_link_state

class SawyerDrawerPickPlacePush:

    def __init__(self, env):
        self.env = env
        self.drawer_policy = SawyerDrawer(env)
        self.push_policy = SawyerPush(env)
        self.pick_place_policy = SawyerPickPlace(env)
    
    def reset(self):
        if self.env.target_object == 'drawer':
            self.drawer_policy.reset()
        elif self.env.target_object == 'push_obj':
            self.push_policy.reset()
        elif self.env.target_object == 'pickplace_obj':
            self.pick_place_policy.reset()
        else:
            raise ValueError("Target object doesn't exist")
    
    def get_action(self):
        if self.env.target_object == 'drawer':
            self.drawer_policy.get_action()
        elif self.env.target_object == 'push_obj':
            self.push_policy.get_action()
        elif self.env.target_object == 'pickplace_obj':
            self.pick_place_policy.get_action()
        else:
            raise ValueError("Target object doesn't exist")

class SawyerDrawer:

    def __init__(self, env):
        self.env = env
    
    def reset(self):
        self.target_pos = self.env.target_position
        self.drawer_yaw = get_link_state(
            self.env.robot_id, 
            self.env.target_object_id, 
            'theta'
        )[2]
        self.grip = -1.
        self.gripper_at_push_point = False
        self.gripper_has_been_above = False
    
    def get_action(self):
        ee_pos = self.env._get_end_effector_pos()
        ee_yaw = self.env._get_end_effector_theta()[2]

        drawer_handle_pos = get_drawer_handle_pos(self.env.target_object_id)
        target_ee_pos_early = drawer_handle_pos - 0.0125 * \
            np.array([np.sin((self.drawer_yaw+180) * np.pi / 180), -
                     np.cos((self.drawer_yaw+180) * np.pi / 180), 0])
        
        if 0 <= self.drawer_yaw < 90:
            target_ee_yaw = self.drawer_yaw
        elif 90 <= self.drawer_yaw < 270:
            target_ee_yaw = self.drawer_yaw - 180
        else:
            target_ee_yaw = self.drawer_yaw - 360

        gripper_yaw_aligned = np.linalg.norm(target_ee_yaw - ee_yaw) > 5
        gripper_pos_xy_aligned = np.linalg.norm(
            target_ee_pos_early[:2] - ee_pos[:2]) < .01
        gripper_pos_z_aligned = np.linalg.norm(
            target_ee_pos_early[2] - ee_pos[2]) < .0175
        gripper_above = ee_pos[2] >= -0.105
        if not self.gripper_has_been_above and gripper_above:
            self.gripper_has_been_above = True
        done = np.linalg.norm(self.target_pos - drawer_handle_pos) < 0.01

        # Stage 1: if gripper is too low, raise it
        if not self.gripper_at_push_point and not self.gripper_has_been_above:
            action = np.array([0, 0, 1, 0])
        # Do stage 2 and 3 at the same time
        elif not self.gripper_at_push_point and (gripper_yaw_aligned or not gripper_pos_xy_aligned):
            # Stage 2: align gripper yaw
            action = np.zeros((4,))
            if gripper_yaw_aligned:
                if target_ee_yaw > ee_yaw:
                    action[3] = 1
                else:
                    action[3] = -1
            # Stage 3: align gripper position with handle position
            if not gripper_pos_xy_aligned:
                xy_action = (target_ee_pos_early - ee_pos) * 6 * 2
                action[0] = xy_action[0]
                action[1] = xy_action[1]
        # Stage 4: lower gripper around handle
        elif not self.gripper_at_push_point and (gripper_pos_xy_aligned and not gripper_pos_z_aligned):
            xy_action = (target_ee_pos_early - ee_pos) * 6 * 2
            action = np.array([xy_action[0], xy_action[1], xy_action[2]*3, 0])
        # Stage 5: open/close drawer
        else:
            if not self.gripper_at_push_point:
                self.gripper_at_push_point = True
            xy_action = self.td_goal - drawer_handle_pos
            action = 12*np.array([xy_action[0], xy_action[1], 0, 0])

        if done:
            action = np.array([0, 0, 1, 0])

        action = np.append(action, [self.grip])
        action = np.clip(action, a_min=-1, a_max=1)
        return action

class SawyerPush:

    def __init__(self, env):
        self.env = env
    
    def reset(self):
        self.target_pos = self.env.target_position
        self.grip = -1.
        self.gripper_at_push_point = False
        self.gripper_has_been_above = False
    
    def get_action(self):
        object_pos, _ = bullet.get_object_position(
            self.env.target_object_id
        )
        ee_pos = self.env._get_end_effector_pos()
        ee_yaw = self.env._get_end_effector_theta()[2]

        vec = self.target_pos[:2] - object_pos[:2]
        direction = (np.arctan2(vec[1], vec[0]) * 180 / np.pi + 360 + 90) % 360
        target_ee_yaw_opts = [direction + 90, direction - 90,
                            direction + 270, direction - 270, direction + 420]
        target_ee_yaw = min(target_ee_yaw_opts,
                          key=lambda x: np.linalg.norm(x - ee_yaw))
        
        target_ee_pos_early = object_pos - 0.11 * \
            np.array([np.sin(direction * np.pi / 180), -
                     np.cos(direction * np.pi / 180), 0])

        gripper_yaw_aligned = np.linalg.norm(target_ee_yaw - ee_yaw) > 5
        gripper_pos_xy_aligned = np.linalg.norm(
            target_ee_pos_early[:2] - ee_pos[:2]) < .005
        gripper_pos_z_aligned = np.linalg.norm(
            target_ee_pos_early[2] - ee_pos[2]) < .0375
        gripper_above = ee_pos[2] >= -0.105
        if not self.gripper_has_been_above and gripper_above:
            self.gripper_has_been_above = True

        done_xy = np.linalg.norm(object_pos[:2] - self.target_pos[:2]) < 0.05
        done = done_xy and np.linalg.norm(object_pos[2] - self.target_pos[2]) < 0.03

        # Stage 1: if gripper is too low, raise it
        if not self.gripper_has_been_above:
            action = np.array([0, 0, 1, 0])

            if target_ee_yaw > ee_yaw:
                action[3] = 1
            else:
                action[3] = -1
        elif (not self.gripper_at_push_point and gripper_yaw_aligned) or (not self.gripper_at_push_point and not gripper_pos_xy_aligned):
            # Stage 2: align gripper yaw
            action = np.zeros((4,))
            if gripper_yaw_aligned:  
                if target_ee_yaw > ee_yaw:
                    action[3] = 1
                else:
                    action[3] = -1
            # Stage 3: align gripper position with handle position
            if not self.gripper_at_push_point and not gripper_pos_xy_aligned: 
                xy_action = (target_ee_pos_early - ee_pos) * 6 * 2
                action[0] = xy_action[0]
                action[1] = xy_action[1]
        # Stage 4: lower gripper around handle
        elif gripper_pos_xy_aligned and not gripper_pos_z_aligned: 
            xy_action = (target_ee_pos_early - ee_pos) * 6 * 2
            action = np.array([xy_action[0], xy_action[1], xy_action[2]*3, 0])
        # Stage 5: open/close drawer
        else:
            if not self.gripper_at_push_point:
                self.gripper_at_push_point = True
            xy_action = self.target_pos - object_pos
            xy_action *= 6
            action = np.array([xy_action[0], xy_action[1], 0, 0])

        if done:
            action = np.array([0, 0, 1, 0])

        action = np.append(action, [self.grip])
        action = np.clip(action, a_min=-1, a_max=1)
        return action


class SawyerPickPlace:
    
    def __init__(self, env):
        self.env = env
    
    def reset(self):
        self.drop_point = self.env.target_position
        self.grip = -1.
        self.object_lifted = False
        self.place_attempted = False

    def get_action(self):
        object_pos, _ = bullet.get_object_position(
            self.env.target_object_id
        )
        target_ee_yaw = np.random.uniform(0, 360)
        ee_pos = self.env._get_end_effector_pos()
        ee_yaw = self.env._get_end_effector_theta()[2]
        
        aligned = np.linalg.norm(object_pos[:2] - ee_pos[:2]) < 0.035
        enclosed = np.linalg.norm(object_pos[2] - ee_pos[2]) < 0.05
        done = np.linalg.norm(object_pos[:2] - self.drop_point[:2]) < 0.025
        above = ee_pos[2] >= -0.125
        ee_yaw_aligned = np.linalg.norm(target_ee_yaw - ee_yaw) > 10

        if not aligned and not above:
            action = np.array([0., 0., 1., 0.])
            self.grip = -1.

            if target_ee_yaw > ee_yaw:
                action[3] = 1
            else:
                action[3] = -1
        elif (not self.object_lifted and ee_yaw_aligned) or not aligned:
            action = np.zeros((4,))
            if not self.object_lifted and ee_yaw_aligned:
                if target_ee_yaw > ee_yaw:
                    action[3] = 1
                else:
                    action[3] = -1
            if not aligned:
                diff = (object_pos - ee_pos) * 3.0 * 2.0
                action[0] = diff[0]
                action[1] = diff[1]
                self.grip = -1.
        elif aligned and not enclosed and self.grip < 1:
            diff = object_pos - ee_pos
            action = np.array([diff[0], diff[1], diff[2], 0.])
            action[2] -= 0.03
            action *= 3.0
            action[2] *= 1.5
            self.grip = -1.
        elif enclosed and self.grip < 1:
            if not self.object_lifted:
                self.object_lifted = True
            diff = object_pos - ee_pos
            action = np.array([diff[0], diff[1], diff[2], 0.])
            action[2] -= 0.03
            action *= 3.0
            action[2] *= 2.0
            self.grip += 0.5
        elif not self.place_attempted and not above:
            action = np.array([0., 0., 1., 0.])
            self.grip = 1.
        elif not done:
            if not self.place_attempted:
                self.place_attempted = True
            diff = self.drop_point - ee_pos
            action = np.array([diff[0], diff[1], diff[2], 0.])
            action[2] = 0
            action *= 3.0
            self.grip = 1.
        else:
            action = np.array([0., 0., 0., 0.])
            self.grip = -1
        
        action = np.append(action, [self.grip])
        action = np.clip(action, a_min=-1, a_max=1)
        return action