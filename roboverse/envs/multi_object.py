from roboverse.assets.shapenet_object_lists import (
    TRAIN_OBJECTS, TRAIN_CONTAINERS, OBJECT_SCALINGS, OBJECT_ORIENTATIONS,
    CONTAINER_CONFIGS)

import numpy as np


class MultiObjectEnv:
    """
    Generalization env. Randomly samples one of the following objects
    every time the env resets.
    """
    def __init__(self,
                 possible_objects=TRAIN_OBJECTS[:10],
                 **kwargs):
        assert isinstance(possible_objects, list)
        self.possible_objects = np.asarray(possible_objects)
        super().__init__(**kwargs)

    def reset(self, object_names=None, target_object=None, **kwargs):
        if object_names is None or target_object is None:
            chosen_obj_idx = np.random.choice(np.arange((len(self.possible_objects))),
                size=self.num_objects, replace=False)
            self.object_names = tuple(self.possible_objects[chosen_obj_idx].tolist())
            self.target_object = self.object_names[0]
        else:
            assert target_object in object_names
            self.object_names = object_names
            self.target_object = target_object

        self.object_scales = dict()
        self.object_orientations = dict()
        for object_name in self.object_names:
            self.object_orientations[object_name] = OBJECT_ORIENTATIONS[object_name]
            self.object_scales[object_name] = OBJECT_SCALINGS[object_name]
        return super().reset(**kwargs)


class MultiObjectMultiContainerEnv:
    """
    Generalization env. Randomly samples one of the following objects
    every time the env resets.
    """
    def __init__(self,
                 num_objects=1,
                 possible_objects=TRAIN_OBJECTS[:10],
                 possible_containers=TRAIN_CONTAINERS[:3],
                 **kwargs):
        assert isinstance(possible_objects, list)
        self.possible_objects = np.asarray(possible_objects)
        self.possible_containers = np.asarray(possible_containers)

        super().__init__(**kwargs)
        self.num_objects = num_objects

    def reset(self):

        chosen_container_idx = np.random.randint(0, len(self.possible_containers))
        self.container_name = self.possible_containers[chosen_container_idx]
        container_config = CONTAINER_CONFIGS[self.container_name]
        self.container_position_low = container_config['container_position_low']
        self.container_position_high = container_config['container_position_high']
        self.container_position_z = container_config['container_position_z']
        self.container_orientation = container_config['container_orientation']
        self.container_scale = container_config['container_scale']
        self.min_distance_from_object = container_config['min_distance_from_object']
        self.place_success_height_threshold = container_config['place_success_height_threshold']
        self.place_success_radius_threshold = container_config['place_success_radius_threshold']

        chosen_obj_idx = np.random.randint(0, len(self.possible_objects),
                                           size=self.num_objects)
        self.object_names = tuple(self.possible_objects[chosen_obj_idx])
        self.object_scales = dict()
        self.object_orientations = dict()
        for object_name in self.object_names:
            self.object_orientations[object_name] = OBJECT_ORIENTATIONS[object_name]
            self.object_scales[object_name] = OBJECT_SCALINGS[object_name]
        self.target_object = self.object_names[0]
        return super().reset()
