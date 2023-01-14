from roboverse.assets.shapenet_object_lists import (
    TRAIN_OBJECTS, TRAIN_CONTAINERS, OBJECT_SCALINGS, OBJECT_ORIENTATIONS,
    CONTAINER_CONFIGS)
from roboverse.envs.widow250 import Widow250Env
from roboverse.bullet import object_utils
import roboverse.bullet as bullet
from roboverse.envs import objects
from .multi_object import MultiObjectEnv, MultiObjectMultiContainerEnv
from roboverse.assets.shapenet_object_lists import CONTAINER_CONFIGS
import os.path as osp
import pybullet as p
import numpy as np

MAX_OBJECTS = 20



class Widow250LanguageEnv(Widow250Env):

    def __init__(self,
                 object_position_high=(.7, .27, -.30),
                 object_position_low=(.5, .18, -.30),
                 min_distance_between_objects=0.07,
                 min_distance_from_object=0.11,
                 place_success_height_threshold=-0.32,
                 place_success_radius_threshold=0.05,
                 start_object_in_gripper=False,
                 show_place_target=False,
                 random_object_pose=False,
                 num_objects=2,
                 num_containers=2,
                 possible_objects=TRAIN_OBJECTS[:10],
                 possible_containers=TRAIN_CONTAINERS[:3],
                 observation_img_dim=128,
                 **kwargs
                 ):
        self.possible_objects = np.asarray(possible_objects)
        self.possible_containers = np.asarray(possible_containers)
        self.min_distance_between_objects = min_distance_between_objects
        self.min_distance_from_object = min_distance_from_object
        self.place_success_height_threshold = place_success_height_threshold
        self.place_success_radius_threshold = place_success_radius_threshold
        self.start_object_in_gripper = start_object_in_gripper
        self.show_place_target = show_place_target
        self.random_object_pose = random_object_pose
        self.num_containers = num_containers
        self.num_objects = num_objects
        
        self.target_container = possible_containers[0]
        self.container_names = [self.target_container]
        container_config = CONTAINER_CONFIGS[self.target_container]
        self.container_position_low = container_config['container_position_low']
        self.container_position_high = container_config['container_position_high']
        self.container_position_z = container_config['container_position_z']
        self.container_orientation = container_config['container_orientation']
        self.container_scale = container_config['container_scale']
        self.min_distance_from_object = container_config['min_distance_from_object']
        self.object_position_low = object_position_low
        self.object_position_high = object_position_high


        super().__init__(object_position_low=object_position_low, object_position_high=object_position_high,
                         observation_img_dim=observation_img_dim, num_objects=num_objects, **kwargs)


    def _load_meshes(self, original_object_positions=None, target_position=None, **kwargs):
        self.table_id = objects.table()
        self.robot_id = objects.widow250()
        self.objects = {}

        if self.load_tray:
            self.tray_id = objects.tray_no_divider_scaled()

        #assert self.container_position_low[2] == self.object_position_low[2]

        if original_object_positions is None or target_position is None:
            extra_positions = self.num_objects + self.num_containers - 1
            self.container_position, self.original_object_positions = \
                object_utils.generate_object_positions_v3(
                    extra_positions, self.object_position_low, self.object_position_high,
                    self.container_position_low, self.container_position_high,
                    min_distance=self.min_distance_between_objects,
                    min_distance_target=self.min_distance_from_object
                )
        else:
            self.container_position = target_position
            self.original_object_positions = original_object_positions

        self.container_position[-1] = self.container_position_z
        self.container_id = object_utils.load_object(self.target_container,
                                                     self.container_position,
                                                     self.container_orientation,
                                                     self.container_scale)
        bullet.step_simulation(self.num_sim_steps_reset)

        for i, container_name in enumerate(self.container_names[1:]):
            container_position = list(self.original_object_positions[i])
            container_config = CONTAINER_CONFIGS[container_name]
            container_position[-1] = container_config['container_position_z']
            self.objects[container_name] = object_utils.load_object(
                container_name,
                container_position,
                container_config['container_orientation'],
                container_config['container_scale'])
            bullet.step_simulation(self.num_sim_steps_reset)

        for i in range(self.num_objects):
            object_name = self.object_names[i]
            object_position = self.original_object_positions[i + self.num_containers - 1]
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
        chosen_container_idx = np.random.permutation(len(self.possible_containers))[:self.num_containers]
        self.container_names = tuple(self.possible_containers[chosen_container_idx])
        self.target_container = self.container_names[0]
        container_config = CONTAINER_CONFIGS[self.target_container]

        self.container_position_low = self.object_position_low
        self.container_position_high = self.object_position_high
        self.container_position_z = container_config['container_position_z']
        self.container_orientation = container_config['container_orientation']
        self.container_scale = container_config['container_scale']
        self.min_distance_from_object = container_config['min_distance_from_object']
        self.place_success_height_threshold = container_config['place_success_height_threshold']
        self.place_success_radius_threshold = container_config['place_success_radius_threshold']

        chosen_obj_idx = np.random.permutation(len(self.possible_objects))[:self.num_objects]
        self.object_names = tuple(self.possible_objects[chosen_obj_idx])
        self.object_scales = dict()
        self.object_orientations = dict()
        for object_name in self.object_names:
            self.object_orientations[object_name] = OBJECT_ORIENTATIONS[object_name]
            self.object_scales[object_name] = OBJECT_SCALINGS[object_name]
        self.target_object = self.object_names[0]

        description = f'move the {self.target_object} onto the {self.target_container}'
        self.description = description.replace('_', ' ')

        super().reset(**kwargs)
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
        info = super().get_info()

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





