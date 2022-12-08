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

    def _load_meshes(self):
        self.robot_id = objects.sawyer()
        self.table_id = objects.table()
        self.wall_id = objects.wall()
        self.objects, self.target_object, self.target_object_id, self.target_position \
            = self.sawyer_util.generate_object_positions()
        self.end_effector_id = get_index_by_attribute(
            self.robot_id, 'link_name', 'gripper_site')
    
    #TODO(Patrick)
    def get_reward(self, info):
        return 0

    #TODO(Patrick)
    def get_info(self):
        info = super(SawyerEnv, self).get_info()

        return info