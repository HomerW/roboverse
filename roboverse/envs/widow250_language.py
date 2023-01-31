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
import random
import re
import itertools

MAX_OBJECTS = 20

REL_POS = dict(
    left=(1., 0., 0.),
    right=(-1., 0., 0.),
    front=(0., 1., 0.),
    back=(0., -1., 0.),
) 

class LanguageTask:
    def __init__(self, object_pos, container_pos, target, goal, rel):
        assert goal in container_pos or goal in object_pos
        assert target in object_pos
        assert not (set(object_pos) & set(container_pos))
        assert rel is None or goal in object_pos
        assert goal not in object_pos or rel in REL_POS

        self.object_pos = object_pos
        self.container_pos = container_pos
        self.target = target
        self.goal = goal
        self.rel = rel

        if goal == target:
            self.description = f'move the {self.target} toward the {self.rel}'
        elif goal in object_pos:
            self.description = f'move the {self.target} to the {self.rel} of the {self.goal}'
        else:
            self.description = f'move the {self.target} onto the {self.goal}'

        self.description = self.description.replace("_", " ")

        if rel is None:
            self.goal_pos = np.array(self.container_pos[self.goal])
        else:
            delta = REL_POS[self.rel]
            self.goal_pos = np.array(self.object_pos[self.goal]) + np.array(delta) / 15

    def __str__(self):
        return self.description

    def __repr__(self):
        return f'LanguageTask({repr(self.object_pos)}, {repr(self.container_pos)}, {repr(self.target)}, {repr(self.goal)}, {repr(self.rel)})'

    @classmethod
    def sample(cls, env):
        containers = random.sample(list(env.possible_containers), env.num_containers)
        objects = random.sample(list(env.possible_objects), env.num_objects)
        target = objects[0]
        goal = np.random.choice(objects + containers)
        rel = random.choice(list(REL_POS)) if goal in objects else None

        return cls.randomize_locations(env, objects, containers, target, goal, rel)

    @classmethod
    def enumerate(cls, env):
        for obj1 in env.possible_objects:
            for obj2 in env.possible_objects:
                for rel in REL_POS:
                    if obj1 == obj2:
                        yield f"move the {obj1} toward the {rel}".replace("_", " ")
                    else:
                        yield f"move the {obj1} to the {rel} of the {obj2}".replace("_", " ")
        for obj in env.possible_objects:
            for cont in env.possible_containers:
                yield f"move the {obj} onto the {cont}".replace("_", " ")

    @classmethod
    def randomize_locations(cls, env, objects, containers, target, goal, rel):
        assert target in env.possible_objects
        assert goal in env.possible_objects or goal in env.possible_containers

        extra_positions = env.num_objects + env.num_containers - 1
        container_position, original_object_positions = \
            object_utils.generate_object_positions_v3(
                extra_positions, env.object_position_low, env.object_position_high,
                env.object_position_low, env.object_position_low,
                min_distance=env.min_distance_between_objects,
                min_distance_target=env.min_distance_from_object
            )
        original_object_positions.append(container_position)

        object_pos = {obj: pos for obj, pos in zip(objects, original_object_positions[:env.num_objects])}
        container_pos = {cont: pos for cont, pos in zip(containers, original_object_positions[env.num_objects:])}

        return cls(object_pos, container_pos, target, goal, rel)

    @classmethod
    def parse(cls, description, env):
        match1 = re.search(r"^move the ([a-zA-Z ]+) to the ([a-zA-Z ]+) of the ([a-zA-Z ]+)$", description)
        match2 = re.search(r"^move the ([a-zA-Z ]+) toward the ([a-zA-Z ]+)$", description)
        match3 = re.search(r"^move the ([a-zA-Z ]+) onto the ([a-zA-Z ]+)$", description)

        if match1:
            target = match1.group(1).replace(" ", "_")
            rel = match1.group(2)
            goal = match1.group(3).replace(" ", "_")
            objects = [target, goal] + random.sample(set(env.possible_objects) - {target, goal}, env.num_objects - 2)
            containers = random.sample(set(env.possible_containers), env.num_containers)
        elif match2:
            target = match2.group(1).replace(" ", "_")
            rel = match2.group(2)
            goal = match2.group(1).replace(" ", "_")
            objects = [target] + random.sample(set(env.possible_objects) - {target}, env.num_objects - 1)
            containers = random.sample(set(env.possible_containers), env.num_containers)
        elif match3:
            target = match3.group(1).replace(" ", "_")
            rel = None
            goal = match3.group(2).replace(" ", "_")
            objects = [target] + random.sample(set(env.possible_objects) - {target}, env.num_objects - 1)
            containers = [goal] + random.sample(set(env.possible_containers) - {goal}, env.num_containers - 1)
        else:
            return None

        return cls.randomize_locations(env, objects, containers, target, goal, rel)

    def spawn(self, env):
        for container_name, container_position in self.container_pos.items():
            container_config = CONTAINER_CONFIGS[container_name]
            container_position[-1] = container_config['container_position_z']
            env.objects[container_name] = object_utils.load_object(
                container_name,
                container_position,
                container_config['container_orientation'],
                container_config['container_scale'])
            bullet.step_simulation(env.num_sim_steps_reset)

        for object_name, object_position in self.object_pos.items():
            object_quat = OBJECT_ORIENTATIONS.get(object_name, (0, 0, 1, 0))
            env.objects[object_name] = object_utils.load_object(
                object_name,
                object_position,
                object_quat=object_quat,
                scale=OBJECT_SCALINGS.get(object_name, 0.6))
            bullet.step_simulation(env.num_sim_steps_reset)

        env.container_position = self.goal_pos
        env.target_object = self.target
        env.original_object_positions = [*self.object_pos.values(), *self.container_pos.values()]


class Widow250LanguageEnv(Widow250Env):

    def __init__(self,
                 object_position_high=(.7, .27, -.30),
                 object_position_low=(.5, .18, -.30),
                 min_distance_between_objects=0.1,
                 min_distance_from_object=0.09,
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
        
        self.task = None
        self.object_position_low = object_position_low
        self.object_position_high = object_position_high
        self.goal_object = None
        self.rel_pos = None

        super().__init__(object_position_low=object_position_low, object_position_high=object_position_high,
                         observation_img_dim=observation_img_dim, num_objects=num_objects, **kwargs)


    def _load_meshes(self, original_object_positions=None, target_position=None, **kwargs):
        self.table_id = objects.table()
        self.robot_id = objects.widow250()
        self.objects = {}

        if self.load_tray:
            self.tray_id = objects.tray_no_divider_scaled()

        if self.task:
            self.task.spawn(self)
        else:
            self.container_position = target_position
            self.original_object_positions = original_object_positions

        bullet.step_simulation(self.num_sim_steps_reset)


    def reset(self, task=None, **kwargs):

        if task is None:
            task = LanguageTask.sample(self)
        self.task = task
        super().reset(**kwargs)

        ee_pos_init, ee_quat_init = bullet.get_link_state(
            self.robot_id, self.end_effector_index)
        ee_pos_init[2] -= 0.05

        self.container_names = tuple(self.task.container_pos)
        self.target_container = self.task.goal if self.task.goal in self.task.container_pos else self.container_names[0]
        container_config = CONTAINER_CONFIGS[self.target_container]
        self.place_success_height_threshold = container_config['place_success_height_threshold']
        self.place_success_radius_threshold = container_config['place_success_radius_threshold']

        self.object_names = tuple(self.task.object_pos)
        self.target_object = self.task.target

        self.description = self.task.description

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





