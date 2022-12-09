import numpy as np
import pybullet as p

from roboverse.bullet.sawyer.sawyer_object_utils import SawyerDrawerWithTrayObjectUtil, SawyerPickPlaceObjectUtil, SawyerPushObjectUtil
from roboverse.bullet.drawer_utils import get_drawer_pos, get_drawer_frame_pos, get_drawer_handle_pos
from roboverse.bullet.control import get_object_position, step_simulation
import roboverse.bullet as bullet

MAX_ATTEMPTS_TO_GENERATE_OBJECT_POSITIONS = 200

class SawyerUtil():
    def __init__(
        self,
        gripper_pos_low,
        gripper_pos_high,
        num_sim_steps_reset=50,
    ):
        self.gripper_pos_low = gripper_pos_low
        self.gripper_pos_high = gripper_pos_high
        self.num_sim_steps_reset = num_sim_steps_reset

        util_kwargs = {
            'gripper_pos_low': gripper_pos_low,
            'gripper_pos_high': gripper_pos_high,
        }
        self.drawer_util = SawyerDrawerWithTrayObjectUtil(**util_kwargs)
        self.push_obj_util = SawyerPushObjectUtil(**util_kwargs)
        self.pickplace_obj_util = SawyerPickPlaceObjectUtil(**util_kwargs)

    def generate_object_positions(
        self, 
        min_distances={
            'drawer_push': 0.2,
            'pnp_push': 0.1,
            'drawer_pnp': 0.15,
        }
    ):
        i = 0
        valid = False
        while not valid:
            if i > 0:
                p.removeBody(drawer_id)
                p.removeBody(tray_id)
                p.removeBody(push_obj_id)
                p.removeBody(pickplace_obj_id)

            drawer_id, tray_id = self.drawer_util.spawn_obj()
            push_obj_id = self.push_obj_util.spawn_obj()
            pickplace_obj_id = self.pickplace_obj_util.spawn_obj()
            step_simulation(self.num_sim_steps_reset)

            valid = True
            # Check all objects within workspace
            for pos in [
                get_drawer_pos(drawer_id),
                get_drawer_frame_pos(drawer_id),
                get_drawer_handle_pos(drawer_id),
                get_object_position(push_obj_id)[0],
                get_object_position(pickplace_obj_id)[0],
            ]:
                valid = valid and self._pos_in_gripper_workspace(pos)

            # Check collision between push object and drawer
            valid = valid and \
                np.linalg.norm(
                    get_drawer_pos(drawer_id) - get_object_position(push_obj_id)[0]
                ) > min_distances['drawer_push']
            valid = valid and \
                np.linalg.norm(
                    get_drawer_frame_pos(drawer_id) - get_object_position(push_obj_id)[0]
                ) > min_distances['drawer_push']
            
            # Check collision between pnp object and push object
            # valid = valid and \
            #     np.linalg.norm(
            #         get_object_position(push_obj_id)[0] - get_object_position(pickplace_obj_id)[0]
            #     ) > min_distances['pnp_push']
            
            # Check collision between pnp object and drawer handle
            # valid = valid and \
            #     np.linalg.norm(
            #         get_drawer_handle_pos(drawer_id) - get_object_position(pickplace_obj_id)[0]
            #     ) > min_distances['drawer_pnp']

            if i > MAX_ATTEMPTS_TO_GENERATE_OBJECT_POSITIONS:
                raise ValueError('Could not spawn objects')
            i += 1
        
        objects = {
            'drawer': drawer_id,
            'tray': tray_id,
            'push_obj': push_obj_id,
            'pickplace_obj': pickplace_obj_id
        }
        object_positions = {
            'drawer': get_drawer_handle_pos(drawer_id),
            'push_obj': get_object_position(push_obj_id)[0],
            'pickplace_obj': get_object_position(pickplace_obj_id)[0],
        }
        target_object, target_object_id, target_position = self.generate_target(objects)

        return objects, object_positions, target_object, target_object_id, target_position
    
    def generate_target(self, objects):
        drawer_id = objects['drawer']
        push_obj_id = objects['push_obj']
        pickplace_obj_id = objects['pickplace_obj']

        drawer_target_position = self.drawer_util.generate_target(drawer_id)
        push_obj_target_position = self.push_obj_util.generate_target(push_obj_id)
        pickplace_obj_target_position = self.pickplace_obj_util.generate_target(pickplace_obj_id)

        target_object_i = np.random.choice(4)
        if target_object_i in [0, 1]:
            target_object = 'drawer'
            target_object_id = drawer_id
            target_position = drawer_target_position
        elif target_object_i == 2:
            target_object = 'push_obj'
            target_object_id = push_obj_id
            target_position = push_obj_target_position
        else:
            target_object = 'pickplace_obj'
            target_object_id = pickplace_obj_id
            target_position = pickplace_obj_target_position
        
        return target_object, target_object_id, target_position

    def _pos_in_gripper_workspace(self, pos):
        x_within_bounds = self.gripper_pos_low[0] <= pos[0] <= self.gripper_pos_high[0]
        y_within_bounds = self.gripper_pos_low[1] <= pos[1] <= self.gripper_pos_high[1]
        z_within_bounds = self.gripper_pos_low[2] <= pos[2] <= self.gripper_pos_high[2]
        return x_within_bounds and y_within_bounds and z_within_bounds