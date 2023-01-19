from roboverse.envs.widow250 import Widow250Env
from roboverse.bullet import object_utils
from roboverse.assets.shapenet_object_lists import (
    TRAIN_OBJECTS, TRAIN_CONTAINERS, OBJECT_SCALINGS, OBJECT_ORIENTATIONS,
    CONTAINER_CONFIGS)

import roboverse.bullet as bullet
from roboverse.envs import objects
import pybullet as p
import numpy as np

class Widow250TableArrangementEnv(Widow250Env):

    def __init__(self,
                 possible_container_objects=['steel_pot_pushable'],
                 possible_utensil_objects=['spoon'],
                 possible_pickplace_objects=['spam'],
                 possible_push_objects=['tomato_can_pushable'],

                 container_position_low=(.72, 0.28, -.3),
                 container_position_high=(.48, 0.12, -.3),
                 container_position_z=-0.36,
                 min_distance_between_objects=0.07,
                 min_distance_from_object=0.0,
                 place_success_height_threshold=-0.3,
                 place_success_radius_threshold=0.07,
                 show_place_target=False,
                 **kwargs,
    ):
        self.possible_container_objects = possible_container_objects
        self.possible_utensil_objects = possible_utensil_objects
        self.possible_pickplace_objects = possible_pickplace_objects
        self.possible_push_objects = possible_push_objects
        self.objects_loaded = False

        self.container_position_low = container_position_low
        self.container_position_high = container_position_high
        self.container_position_z = container_position_z
        self.min_distance_between_objects = min_distance_between_objects
        self.min_distance_from_object = min_distance_from_object
        self.place_success_height_threshold = place_success_height_threshold
        self.place_success_radius_threshold = place_success_radius_threshold
        self.show_place_target = show_place_target

        super(Widow250TableArrangementEnv, self).__init__(**kwargs)

        assert self.num_objects == 4

    def _load_meshes(self, original_object_positions=None, target_position=None, **kwargs):
        self.table_id = objects.table_centered()
        self.robot_id = objects.widow250()
        self.wall_id = objects.back_wall()
        self.objects = {}

        if self.load_tray:
            self.tray_id = objects.tray_no_divider_scaled()

        if not self.objects_loaded:
            return

        """
        TODO(avi) This needs to be cleaned up, generate function should only
                  take in (x,y) positions instead.
        """
        assert self.container_position_low[2] == self.object_position_low[2]

        if original_object_positions is None or target_position is None:
            target_object, self.container_position, original_object_positions = \
                object_utils.generate_object_positions_table_arrangement(
                    self.object_position_low, self.object_position_high,
                    self.container_position_low, self.container_position_high,
                    min_distance=self.min_distance_between_objects,
                    min_distance_target=self.min_distance_from_object
                )

            target_object_dict = {
                ## Important: Same order
                'push': self.push_object_name,
                'container': self.container_object_name,
                'utensil': self.utensil_object_name,
                'pickplace': self.pickplace_object_name,
            }
            self.target_object = target_object_dict[target_object]

            self.original_object_positions = [
                ## Important: Same order
                original_object_positions['push'],
                original_object_positions['container'],
                original_object_positions['utensil'],
                original_object_positions['pickplace'],
            ]
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
        
        for object_name, object_position in zip(
        [
            ## Important: Same order
            self.push_object_name,
            self.container_object_name,
            self.utensil_object_name,
            self.pickplace_object_name,
        ],
        self.original_object_positions
        ):
            default_object_quat = self.object_orientations[object_name]
            object_deg = bullet.quat_to_deg(default_object_quat)
            object_deg[2] = np.random.uniform(low=0, high=360)
            object_quat = bullet.deg_to_quat(object_deg)
            self.objects[object_name] = object_utils.load_object(
                object_name,
                object_position,
                object_quat=object_quat,
                scale=self.object_scales[object_name])
            bullet.step_simulation(self.num_sim_steps_reset)

    def reset(
        self, 
        container_object_name=None, 
        utensil_object_name=None, 
        pickplace_object_name=None, 
        push_object_name=None, 
        target_object=None,
        **kwargs
    ):
        if container_object_name is None \
            or utensil_object_name is None \
            or pickplace_object_name is None \
            or push_object_name is None \
            or target_object is None:
            self.container_object_name = str(np.random.choice(self.possible_container_objects))
            self.utensil_object_name = str(np.random.choice(self.possible_utensil_objects))
            self.pickplace_object_name = str(np.random.choice(self.possible_pickplace_objects))
            self.push_object_name = str(np.random.choice(self.possible_push_objects))
            ## Overwritten in _load_meshes
            self.target_object = None
        else:
            assert target_object in [
                container_object_name, 
                utensil_object_name, 
                pickplace_object_name, 
                push_object_name
            ]
            self.container_object_name = container_object_name
            self.utensil_object_name = utensil_object_name
            self.pickplace_object_name = pickplace_object_name
            self.push_object_name = push_object_name
            self.target_object = target_object
        
        self.object_names = [
            self.container_object_name, 
            self.utensil_object_name, 
            self.pickplace_object_name, 
            self.push_object_name
        ]
        
        self.object_scales = dict()
        self.object_orientations = dict()
        for object_name in self.object_names:
            self.object_orientations[object_name] = OBJECT_ORIENTATIONS.get(object_name, (0, 0, 1, 0))
            self.object_scales[object_name] = OBJECT_SCALINGS.get(object_name, 1.0)
        self.objects_loaded = True
        return super().reset(**kwargs)

    def get_reward(self, info):
        if self.reward_type == 'pick_place':
            reward = float(info['place_success_target'])
        else:
            raise NotImplementedError
        return reward

    def get_info(self):
        info = dict()

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
    
        original_object_positions = list(self.original_object_positions)

        info['container_object_name'] = self.container_object_name
        info['utensil_object_name'] = self.utensil_object_name
        info['pickplace_object_name'] = self.pickplace_object_name
        info['push_object_name'] = self.push_object_name
        info['target_object'] = self.target_object
        info['initial_positions'] = original_object_positions
        info['target_position'] = self.container_position

        return info