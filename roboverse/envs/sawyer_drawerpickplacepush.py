import numpy as np

import roboverse.bullet as bullet
from roboverse.bullet.sawyer.sawyer_util import SawyerUtil
from roboverse.bullet.sawyer.sawyer_queries import get_index_by_attribute
from roboverse.envs import objects
from roboverse.envs.sawyer import SawyerEnv

class SawyerDrawerPickPlacePushEnv(SawyerEnv):

    def __init__(self,         
                 *args,
                 **kwargs
                 ):
        
        super().__init__(*args, **kwargs)

        self.sawyer_util = SawyerUtil(
            gripper_pos_low=self._pos_low,
            gripper_pos_high=self._pos_high,
        )

        self.target_object = None
        self.target_position = None
        self.object_positions = {
            'drawer': None,
            'push_obj': None,
            'pickplace_obj': None,
        }

    def _load_meshes(self):
        self.robot_id = objects.sawyer()
        self.table_id = objects.table()
        self.wall_id = objects.wall()
        self.objects, self.object_positions, self.target_object, self.target_object_id, self.target_position \
            = self.sawyer_util.generate_object_positions()
        self.end_effector_id = get_index_by_attribute(
            self.robot_id, 'link_name', 'gripper_site')
    
    def get_reward(self, info):
        return float(info['success'])

    def get_info(self):
        info = super(SawyerDrawerPickPlacePushEnv, self).get_info()

        info['target_object'] = self.target_object
        info['target_position'] = self.target_position
        info['drawer_position'] = self.object_positions['drawer']
        info['push_obj_position'] = self.object_positions['push_obj']
        info['pickplace_obj_position'] = self.object_positions['pickplace_obj']
        if self.target_object == 'drawer':
            info['distance'] = np.linalg.norm(info['drawer_position'] - info['target_position'])
            info['success'] = np.linalg.norm(info['drawer_position'] - info['target_position']) < 0.065
        elif self.target_object == 'push_obj':
            info['distance'] = np.linalg.norm(info['push_obj_position'] - info['target_position'])
            info['success'] = np.linalg.norm(info['push_obj_position'] - info['target_position']) < 0.065
        elif self.target_object == 'pickplace_obj':
            info['distance'] = np.linalg.norm(info['pickplace_obj_position'] - info['target_position'])
            info['success'] = np.linalg.norm(info['pickplace_obj_position'] - info['target_position']) < 0.08
        else:
            info['distance'] = -1
            info['success'] = -1

        return info