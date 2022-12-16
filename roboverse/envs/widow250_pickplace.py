from roboverse.envs.widow250 import Widow250Env
from roboverse.bullet import object_utils
import roboverse.bullet as bullet
from roboverse.envs import objects
from .multi_object import MultiObjectEnv, MultiObjectMultiContainerEnv
from roboverse.assets.shapenet_object_lists import CONTAINER_CONFIGS
import os.path as osp
import pybullet as p
import numpy as np

OBJECT_IN_GRIPPER_PATH = osp.join(osp.dirname(osp.dirname(osp.realpath(__file__))),
                'assets/bullet-objects/bullet_saved_states/objects_in_gripper/')
MAX_OBJECTS = 20

class Widow250PickPlaceEnv(Widow250Env):

    def __init__(self,
                 container_name='bowl_small',
                 fixed_container_position=False,
                 start_object_in_gripper=False,
                 **kwargs
                 ):
        self.container_name = container_name

        container_config = CONTAINER_CONFIGS[self.container_name]
        self.fixed_container_position = fixed_container_position
        if self.fixed_container_position:
            self.container_position_low = container_config['container_position_default']
            self.container_position_high = container_config['container_position_default']
        else:
            self.container_position_low = container_config['container_position_low']
            self.container_position_high = container_config['container_position_high']
        self.container_position_z = container_config['container_position_z']
        self.container_orientation = container_config['container_orientation']
        self.container_scale = container_config['container_scale']
        self.min_distance_from_object = container_config['min_distance_from_object']

        self.place_success_height_threshold = container_config['place_success_height_threshold']
        self.place_success_radius_threshold = container_config['place_success_radius_threshold']

        self.start_object_in_gripper = start_object_in_gripper
        super(Widow250PickPlaceEnv, self).__init__(**kwargs)

    def _load_meshes(self):
        self.table_id = objects.table()
        self.robot_id = objects.widow250()
        self.objects = {}

        """
        TODO(avi) This needs to be cleaned up, generate function should only
                  take in (x,y) positions instead.
        """
        assert self.container_position_low[2] == self.object_position_low[2]

        if self.num_objects == 2:
            self.container_position, self.original_object_positions = \
                object_utils.generate_object_positions_v2(
                    self.object_position_low, self.object_position_high,
                    self.container_position_low, self.container_position_high,
                    min_distance_small_obj=0.07,
                    min_distance_large_obj=self.min_distance_from_object,
                )
        elif self.num_objects == 1:
            self.container_position, self.original_object_positions = \
                object_utils.generate_object_positions_single(
                    self.object_position_low, self.object_position_high,
                    self.container_position_low, self.container_position_high,
                    min_distance_large_obj=self.min_distance_from_object,
                )
        else:
            raise NotImplementedError

        # TODO(avi) Need to clean up
        self.container_position[-1] = self.container_position_z
        self.container_id = object_utils.load_object(self.container_name,
                                                     self.container_position,
                                                     self.container_orientation,
                                                     self.container_scale)
        bullet.step_simulation(self.num_sim_steps_reset)
        for object_name, object_position in zip(self.object_names,
                                                self.original_object_positions):
            self.objects[object_name] = object_utils.load_object(
                object_name,
                object_position,
                object_quat=self.object_orientations[object_name],
                scale=self.object_scales[object_name])
            bullet.step_simulation(self.num_sim_steps_reset)

    def reset(self):
        super(Widow250PickPlaceEnv, self).reset()
        ee_pos_init, ee_quat_init = bullet.get_link_state(
            self.robot_id, self.end_effector_index)
        ee_pos_init[2] -= 0.05

        if self.start_object_in_gripper:
            bullet.load_state(osp.join(OBJECT_IN_GRIPPER_PATH,
                'object_in_gripper_reset.bullet'))
            self.is_gripper_open = False

        return self.get_observation()

    def get_reward(self, info):
        if self.reward_type == 'pick_place':
            reward = float(info['place_success_target'])
        elif self.reward_type == 'grasp':
            reward = float(info['grasp_success_target'])
        else:
            raise NotImplementedError
        return reward

    def get_info(self):
        info = super(Widow250PickPlaceEnv, self).get_info()

        info['place_success'] = False
        for object_name in self.object_names:
            place_success = object_utils.check_in_container(
                object_name, self.objects, self.container_position,
                self.place_success_height_threshold,
                self.place_success_radius_threshold)
            if place_success:
                info['place_success'] = place_success

        info['place_success_target'] = object_utils.check_in_container(
            self.target_object, self.objects, self.container_position,
            self.place_success_height_threshold,
            self.place_success_radius_threshold)

        return info

class Widow250PickPlacePositionEnv(Widow250Env):

    def __init__(self,
                 container_position_low=(.72, 0.28, -.3),
                 container_position_high=(.48, 0.12, -.3),
                 container_position_z=-0.36,
                 min_distance_between_objects=0.07,
                 min_distance_from_object=0.11,
                 place_success_height_threshold=-0.32,
                 place_success_radius_threshold=0.05,
                 start_object_in_gripper=False,
                 show_place_target=False,
                 random_object_pose=False,
                 **kwargs
                 ):
        self.container_position_low = container_position_low
        self.container_position_high = container_position_high
        self.container_position_z = container_position_z
        self.min_distance_between_objects = min_distance_between_objects
        self.min_distance_from_object = min_distance_from_object
        self.place_success_height_threshold = place_success_height_threshold
        self.place_success_radius_threshold = place_success_radius_threshold
        self.start_object_in_gripper = start_object_in_gripper
        self.show_place_target = show_place_target
        self.random_object_pose = random_object_pose
        super(Widow250PickPlacePositionEnv, self).__init__(**kwargs)

    def _load_meshes(self, original_object_positions=None, target_position=None, **kwargs):
        self.table_id = objects.table()
        self.robot_id = objects.widow250()
        self.objects = {}

        if self.load_tray:
            self.tray_id = objects.tray_no_divider_scaled()

        """
        TODO(avi) This needs to be cleaned up, generate function should only
                  take in (x,y) positions instead.
        """
        assert self.container_position_low[2] == self.object_position_low[2]

        if original_object_positions is None or target_position is None:
            self.container_position, self.original_object_positions = \
                object_utils.generate_object_positions_v3(
                    self.num_objects, self.object_position_low, self.object_position_high,
                    self.container_position_low, self.container_position_high,
                    min_distance=self.min_distance_between_objects,
                    min_distance_target=self.min_distance_from_object
                )
        else:
            self.container_position = target_position
            self.original_object_positions = original_object_positions

        # TODO(avi) Need to clean up
        self.container_position[-1] = self.container_position_z
        if self.show_place_target:
            visualShapeId = p.createVisualShape(shapeType=p.GEOM_CYLINDER,
                                                rgbaColor=[1, 1, 1, 1],
                                                length=0.001,
                                                radius=self.place_success_radius_threshold)
            p.createMultiBody(baseVisualShapeIndex=visualShapeId,
                              basePosition=self.container_position)
        bullet.step_simulation(self.num_sim_steps_reset)
        for object_name, object_position in zip(self.object_names,
                                                self.original_object_positions):
            if self.random_object_pose:
                object_quat = tuple(np.random.uniform(low=0, high=1, size=4))
            else:
                object_quat = self.object_orientations[object_name]
            self.objects[object_name] = object_utils.load_object(
                object_name,
                object_position,
                object_quat=object_quat,
                scale=self.object_scales[object_name])
            bullet.step_simulation(self.num_sim_steps_reset)

    def reset(self, **kwargs):
        super(Widow250PickPlacePositionEnv, self).reset(**kwargs)
        ee_pos_init, ee_quat_init = bullet.get_link_state(
            self.robot_id, self.end_effector_index)
        ee_pos_init[2] -= 0.05

        if self.start_object_in_gripper:
            bullet.load_state(osp.join(OBJECT_IN_GRIPPER_PATH,
                'object_in_gripper_reset.bullet'))
            self.is_gripper_open = False

        return self.get_observation()

    def get_reward(self, info):
        if self.reward_type == 'pick_place':
            reward = float(info['place_success_target'])
        elif self.reward_type == 'grasp':
            reward = float(info['grasp_success_target'])
        else:
            raise NotImplementedError
        return reward

    def get_info(self):
        info = super(Widow250PickPlacePositionEnv, self).get_info()

        info['place_success'] = False
        for object_name in self.object_names:
            place_success = object_utils.check_in_container(
                object_name, self.objects, self.container_position,
                self.place_success_height_threshold,
                self.place_success_radius_threshold)
            if place_success:
                info['place_success'] = place_success

        info['place_success_target'] = object_utils.check_in_container(
            self.target_object, self.objects, self.container_position,
            self.place_success_height_threshold,
            self.place_success_radius_threshold)
        
        ## Keeping these entries the same length or else hdf5 saving errors during data collection.
        object_names = list(self.object_names)
        object_names = tuple(
            object_names \
            + ['' for _ in range(MAX_OBJECTS - len(object_names))]
        )
        original_object_positions = list(self.original_object_positions)
        original_object_positions = tuple(
            original_object_positions + \
            [-np.ones(3) for _ in range(MAX_OBJECTS - len(original_object_positions))]
        )

        info['object_names'] = object_names
        info['target_object'] = self.target_object
        info['initial_positions'] = original_object_positions
        info['target_position'] = self.container_position

        return info


class Widow250PickPlaceMultiObjectEnv(MultiObjectEnv, Widow250PickPlaceEnv):
    """Grasping Env but with a random object each time."""

class Widow250PickPlacePositionMultiObjectEnv(MultiObjectEnv, Widow250PickPlacePositionEnv):
    """Grasping Env but with a random object each time."""

class Widow250PickPlaceMultiObjectMultiContainerEnv(
    MultiObjectMultiContainerEnv, Widow250PickPlaceEnv):
    """Grasping Env but with a random object each time."""


if __name__ == "__main__":

    # Fixed container position
    env = Widow250PickPlaceEnv(
        reward_type='pick_place',
        control_mode='discrete_gripper',
        object_names=('shed', 'two_handled_vase'),
        object_scales=(0.7, 0.6),
        target_object='shed',
        load_tray=False,
        object_position_low=(.49, .18, -.20),
        object_position_high=(.59, .27, -.20),

        container_name='cube',
        container_position_low=(.72, 0.23, -.20),
        container_position_high=(.72, 0.23, -.20),
        container_position_z=-0.34,
        container_orientation=(0, 0, 0.707107, 0.707107),
        container_scale=0.05,

        camera_distance=0.29,
        camera_target_pos=(0.6, 0.2, -0.28),
        gui=True
    )

    # env = Widow250PickPlaceEnv(
    #     reward_type='pick_place',
    #     control_mode='discrete_gripper',
    #     object_names=('shed',),
    #     object_scales=(0.7,),
    #     target_object='shed',
    #     load_tray=False,
    #     object_position_low=(.5, .18, -.25),
    #     object_position_high=(.7, .27, -.25),
    #
    #     container_name='bowl_small',
    #     container_position_low=(.5, 0.26, -.25),
    #     container_position_high=(.7, 0.26, -.25),
    #     container_orientation=(0, 0, 0.707107, 0.707107),
    #     container_scale=0.07,
    #
    #     camera_distance=0.29,
    #     camera_target_pos=(0.6, 0.2, -0.28),
    #     gui=True
    # )

    import time
    for _ in range(10):
        env.reset()
        for _ in range(5):
            env.step(env.action_space.sample()*0.1)
            time.sleep(0.1)
