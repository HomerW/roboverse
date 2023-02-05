import pybullet_data
import pybullet as p
import os
import importlib.util
import numpy as np
from .control import get_object_position, get_link_state
from roboverse.bullet.drawer_utils import *
from roboverse.bullet.button_utils import *
from itertools import combinations

CUR_PATH = os.path.dirname(os.path.realpath(__file__))
ASSET_PATH = os.path.join(CUR_PATH, '../assets')
SHAPENET_ASSET_PATH = os.path.join(ASSET_PATH, 'bullet-objects/ShapeNetCore')
BASE_ASSET_PATH = os.path.join(ASSET_PATH, 'bullet-objects')
BULLET3_ASSET_PATH = os.path.join(BASE_ASSET_PATH, 'bullet3')

MAX_ATTEMPTS_TO_GENERATE_OBJECT_POSITIONS = 200
SHAPENET_SCALE = 0.5


def check_in_container(object_name,
                       object_id_map,
                       container_pos,
                       place_success_height_threshold,
                       place_success_radius_threshold,
                       ):
    object_pos, _ = get_object_position(object_id_map[object_name])
    object_height = object_pos[2]
    object_xy = object_pos[:2]
    container_center_xy = container_pos[:2]
    success = False
    if object_height < place_success_height_threshold:
        object_container_distance = np.linalg.norm(object_xy - container_center_xy)
        if object_container_distance < place_success_radius_threshold:
            success = True

    return success


def check_grasp(object_name,
                object_id_map,
                robot_id,
                end_effector_index,
                grasp_success_height_threshold,
                grasp_success_object_gripper_threshold,
                ):
    object_pos, _ = get_object_position(object_id_map[object_name])
    object_height = object_pos[2]
    success = False
    if object_height > grasp_success_height_threshold:
        ee_pos, _ = get_link_state(
            robot_id, end_effector_index)
        object_gripper_distance = np.linalg.norm(
            object_pos - ee_pos)
        if object_gripper_distance < \
                grasp_success_object_gripper_threshold:
            success = True

    return success

def check_displacement(initial_object_positions,
                       object_id_map,
                       displacement_threshold):
    displaced = False
    for k in object_id_map.keys():
        assert k in initial_object_positions.keys()
    
        initial_obj_pos = initial_object_positions[k]
        obj_pos, _ = get_object_position(object_id_map[k])
        if np.linalg.norm(initial_obj_pos - obj_pos) > displacement_threshold:
            displaced = True
    
    return displaced

# TODO(avi) Need to clean unify these object position functions
def generate_object_positions_single(
        small_object_position_low, small_object_position_high,
        large_object_position_low, large_object_position_high,
        min_distance_large_obj=0.1):

    valid = False
    max_attempts = MAX_ATTEMPTS_TO_GENERATE_OBJECT_POSITIONS
    i = 0
    while not valid:
        large_object_position = np.random.uniform(
                low=large_object_position_low, high=large_object_position_high)
        small_object_positions = []
        small_object_position = np.random.uniform(
            low=small_object_position_low, high=small_object_position_high)
        small_object_positions.append(small_object_position)
        valid = np.linalg.norm(small_object_positions[0] - large_object_position) > min_distance_large_obj
        if i > max_attempts:
            raise ValueError('Min distance could not be assured')

    return large_object_position, small_object_positions


def generate_object_positions_table_arrangement(
    object_position_low, object_position_high, 
    target_position_low, target_position_high, 
    min_distance=0.07, min_distance_target=0.1):

    valid = False
    max_attempts = MAX_ATTEMPTS_TO_GENERATE_OBJECT_POSITIONS
    i = 0
    pickplace_object_in_pot = np.random.uniform() < 0.5
    # ## Debugging
    # pickplace_object_in_pot = True
    while not valid:
        target_position = np.random.uniform(
            low=target_position_low, high=target_position_high)
        object_positions = dict()
        if pickplace_object_in_pot:
            possible_target_objects = ['container', 'utensil', 'push']
            for object in possible_target_objects:
                object_position = np.random.uniform(
                    low=object_position_low, high=object_position_high)
                object_positions[object] = object_position
            object_positions['pickplace'] = object_positions['container'] 
            target_object = np.random.choice(possible_target_objects)

            # ## Debugging
            # target_object = 'container'
        else:
            possible_target_objects = ['container', 'utensil', 'push', 'pickplace']
            for object in possible_target_objects:
                object_position = np.random.uniform(
                    low=object_position_low, high=object_position_high)
                object_positions[object] = object_position
            target_object = np.random.choice(possible_target_objects)

            # ## Debugging
            # target_object = 'container'

            # If pickplace object not in container, target position will be in container
            if target_object == 'pickplace':
                target_position = object_positions['container']

        valid = True
        for (obj1, obj2) in combinations(possible_target_objects, r=2):
            pos1 = object_positions[obj1]
            pos2 = object_positions[obj2]
            valid = valid and \
                np.linalg.norm(pos1 - pos2) > min_distance
        if pickplace_object_in_pot or target_object != 'pickplace':
            for pos in object_positions.values():
                valid = valid and \
                    np.linalg.norm(pos - target_position) > min_distance_target

        if i > max_attempts:
            raise ValueError('Min distance could not be assured')

    return target_object, target_position, object_positions

def generate_object_positions_v3(
    num_objects, object_position_low, object_position_high, 
    target_position_low, target_position_high, 
    min_distance=0.07, min_distance_target=0.1):

    valid = False
    max_attempts = MAX_ATTEMPTS_TO_GENERATE_OBJECT_POSITIONS
    i = 0
    while not valid:
        target_position = np.random.uniform(
                low=target_position_low, high=target_position_high)
        object_positions = []
        for _ in range(num_objects):
            object_position = np.random.uniform(
                low=object_position_low, high=object_position_high)
            object_positions.append(object_position)

        valid = True
        for (pos1, pos2) in combinations(object_positions, r=2):
            valid = valid and \
                np.linalg.norm(pos1 - pos2) > min_distance
        for pos in object_positions:
            valid = valid and \
                np.linalg.norm(pos - target_position) > min_distance_target

        if i > max_attempts:
            raise ValueError('Min distance could not be assured')

    return target_position, object_positions

def generate_object_positions_v2(
        small_object_position_low, small_object_position_high,
        large_object_position_low, large_object_position_high,
        min_distance_small_obj=0.07, min_distance_large_obj=0.1):

    valid = False
    max_attempts = MAX_ATTEMPTS_TO_GENERATE_OBJECT_POSITIONS
    i = 0
    while not valid:
        large_object_position = np.random.uniform(
                low=large_object_position_low, high=large_object_position_high)
        # large_object_position = np.reshape(large_object_position, (1, 3))

        small_object_positions = []
        for _ in range(2):
            small_object_position = np.random.uniform(
                low=small_object_position_low, high=small_object_position_high)
            small_object_positions.append(small_object_position)

        valid_1 = np.linalg.norm(small_object_positions[0] - small_object_positions[1]) > min_distance_small_obj
        valid_2 = np.linalg.norm(small_object_positions[0] - large_object_position) > min_distance_large_obj
        valid_3 = np.linalg.norm(small_object_positions[1] - large_object_position) > min_distance_large_obj

        valid = valid_1 and valid_2 and valid_3
        if i > max_attempts:
            raise ValueError('Min distance could not be assured')

    return large_object_position, small_object_positions


def generate_object_positions(object_position_low, object_position_high,
                              num_objects, min_distance=0.07,
                              current_positions=None):
    if current_positions is None:
        object_positions = np.random.uniform(
            low=object_position_low, high=object_position_high)
        object_positions = np.reshape(object_positions, (1, 3))
    else:
        object_positions = current_positions

    max_attempts = MAX_ATTEMPTS_TO_GENERATE_OBJECT_POSITIONS
    i = 0
    while object_positions.shape[0] < num_objects:
        i += 1
        object_position_candidate = np.random.uniform(
            low=object_position_low, high=object_position_high)
        object_position_candidate = np.reshape(
            object_position_candidate, (1, 3))
        min_distance_so_far = []
        for o in object_positions:
            dist = np.linalg.norm(o - object_position_candidate)
            min_distance_so_far.append(dist)
        min_distance_so_far = np.array(min_distance_so_far)
        if (min_distance_so_far > min_distance).any():
            object_positions = np.concatenate(
                (object_positions, object_position_candidate), axis=0)

        if i > max_attempts:
            raise ValueError('Min distance could not be assured')

    return object_positions


def import_metadata(asset_path):
    metadata_spec = importlib.util.spec_from_file_location(
        "metadata", os.path.join(asset_path, "metadata.py"))
    metadata = importlib.util.module_from_spec(metadata_spec)
    metadata_spec.loader.exec_module(metadata)
    return metadata.obj_path_map, metadata.path_scaling_map


def import_shapenet_metadata():
    return import_metadata(SHAPENET_ASSET_PATH)


# TODO(avi, albert) This should be cleaned up
shapenet_obj_path_map, shapenet_path_scaling_map = import_shapenet_metadata()


def load_object(object_name, object_position, object_quat, scale=1.0):
    if object_name in shapenet_obj_path_map.keys():
        return load_shapenet_object(object_name, object_position,
                                    object_quat=object_quat, scale=scale)
    elif object_name in BULLET_OBJECT_SPECS.keys():
        return load_bullet_object(object_name,
                                  basePosition=object_position,
                                  baseOrientation=object_quat,
                                  globalScaling=scale)
    elif object_name in CYLINDER_COLORS.keys():
        return load_programmatic_object(object_name, object_position,
                                        object_quat=object_quat)
    else:
        print(object_name)
        raise NotImplementedError


def load_shapenet_object(object_name, object_position,
                         object_quat=(1, -1, 0, 0),  scale=1.0):
    object_path = shapenet_obj_path_map[object_name]
    path = object_path.split('/')
    dir_name = path[-2]
    object_name = path[-1]
    filepath_collision = os.path.join(
        SHAPENET_ASSET_PATH,
        'ShapeNetCore_vhacd/{0}/{1}/model.obj'.format(dir_name, object_name))
    filepath_visual = os.path.join(
        SHAPENET_ASSET_PATH,
        'ShapeNetCore.v2/{0}/{1}/models/model_normalized.obj'.format(
            dir_name, object_name))
    scale = SHAPENET_SCALE * scale * shapenet_path_scaling_map[object_path]
    collisionid = p.createCollisionShape(p.GEOM_MESH,
                                         fileName=filepath_collision,
                                         meshScale=scale * np.array([1, 1, 1]))
    visualid = p.createVisualShape(p.GEOM_MESH, fileName=filepath_visual,
                                   meshScale=scale * np.array([1, 1, 1]))
    body = p.createMultiBody(0.05, collisionid, visualid)
    p.resetBasePositionAndOrientation(body, object_position, object_quat)
    return body


def load_bullet_object(object_name, **kwargs):
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    object_specs = BULLET_OBJECT_SPECS[object_name]
    object_specs.update(**kwargs)
    object_id = p.loadURDF(**object_specs)
    return object_id

def load_programmatic_object(object_name, object_position,
                             object_quat=(1, -1, 0, 0)):  
    collisionid = p.createCollisionShape(**PROGRAMMATIC_OBJECT_SPECS["collision"])
    visualid = p.createVisualShape(**PROGRAMMATIC_OBJECT_SPECS["visual"], 
        rgbaColor=CYLINDER_COLORS[object_name])
    body = p.createMultiBody(0.05, collisionid, visualid)
    p.resetBasePositionAndOrientation(body, object_position, object_quat)
    p.changeDynamics(body, -1, lateralFriction=1.0)
    return body


# TODO(avi) Maybe move this to a different file
BULLET_OBJECT_SPECS = dict(
    duck=dict(
        fileName='duck_vhacd.urdf',
        basePosition=(.65, 0.3, -.3),
        baseOrientation=(0, 0, 0.707107, 0.707107),
        globalScaling=0.8,
    ),
    bowl_small=dict(
        fileName=os.path.join(BASE_ASSET_PATH, 'bowl/bowl.urdf'),
        basePosition=(.72, 0.23, -.35),
        baseOrientation=(0, 0, 0.707107, 0.707107),
        globalScaling=0.07,
    ),
    pan=dict(
        fileName=os.path.join(BASE_ASSET_PATH, 'pan/pan.urdf'),
        basePosition=(.65, 0.3, -.3),
        baseOrientation=(0, 0, 0.707107, 0.707107),
        globalScaling=0.8,
    ),
    drawer=dict(
        fileName=os.path.join(
            BASE_ASSET_PATH, 'drawer/drawer_with_tray_inside.urdf'),
        basePosition=(.7, 0.2, -.35),
        baseOrientation=(0, 0, 0.707107, 0.707107),
        globalScaling=0.1,
    ),
    drawer_no_handle=dict(
        fileName=os.path.join(
            BASE_ASSET_PATH, 'drawer/drawer_no_handle.urdf'),
        basePosition=(.7, 0.2, -.35),
        baseOrientation=(0, 0, 0.707107, 0.707107),
        globalScaling=0.1,
    ),
    tray=dict(
        fileName='tray/tray.urdf',
        basePosition=(.7, 0.2, -.35),
        baseOrientation=(0, 0, 0.707107, 0.707107),
        globalScaling=0.25,
    ),
    open_box=dict(
        fileName=os.path.join(
            BASE_ASSET_PATH, 'box_open_top/box_open_top.urdf'),
        basePosition=(.7, 0.2, -.35),
        baseOrientation=(0, 0, 0.707107, 0.707107),
        globalScaling=0.25,
    ),
    cube=dict(
        fileName=os.path.join(
            BASE_ASSET_PATH, 'cube/cube.urdf'),
        basePosition=(.7, 0.2, -.35),
        baseOrientation=(0, 0, 0.707107, 0.707107),
        globalScaling=0.05,
    ),
    spam=dict(
        fileName=os.path.join(
            BASE_ASSET_PATH, 'spam/spam.urdf'),
        basePosition=(.7, 0.2, -.35),
        baseOrientation=(0, 0, 0.707107, 0.707107),
        globalScaling=0.25,
    ),
    pan_tefal=dict(
        fileName=os.path.join(
            BULLET3_ASSET_PATH, 'dinnerware/pan_tefal.urdf'),
        basePosition=(.65, 0.3, -.3),
        baseOrientation=(0, 0, 0.707107, 0.707107),
        globalScaling=0.8,
        useFixedBase=1,
    ),
    table_top=dict(
        fileName=os.path.join(
            BULLET3_ASSET_PATH, 'table/table2.urdf'),
        basePosition=(.65, 0.3, -.3),
        baseOrientation=(0, 0, 0.707107, 0.707107),
        globalScaling=0.8,
        useFixedBase=1,
    ),
    checkerboard_table=dict(
        fileName=os.path.join(
            BULLET3_ASSET_PATH, 'table_square/table_square.urdf'),
        basePosition=(.65, 0.3, -.3),
        baseOrientation=(0, 0, 0.707107, 0.707107),
        globalScaling=0.8,
        useFixedBase=1,
    ),
    torus=dict(
        fileName=os.path.join(
            BULLET3_ASSET_PATH, 'torus/torus.urdf'),
        basePosition=(.65, 0.3, -.3),
        baseOrientation=(0, 0, 0.707107, 0.707107),
        globalScaling=0.8,
        useFixedBase=1,
    ),
    cube_concave=dict(
        fileName=os.path.join(
            BULLET3_ASSET_PATH, 'cube_concave.urdf'),
        basePosition=(.65, 0.3, -.3),
        baseOrientation=(0, 0, 0.707107, 0.707107),
        globalScaling=0.8,
        useFixedBase=1,
    ),
    plate=dict(
        fileName=os.path.join(
            BULLET3_ASSET_PATH, 'dinnerware/plate.urdf'),
        basePosition=(.65, 0.3, -.3),
        baseOrientation=(0, 0, 0.707107, 0.707107),
        globalScaling=0.8,
        useFixedBase=1,
    ),
    husky=dict(
        fileName=os.path.join(
            BULLET3_ASSET_PATH, 'husky/husky.urdf'),
        basePosition=(.65, 0.3, -.3),
        baseOrientation=(0, 0, 0.707107, 0.707107),
        globalScaling=0.8,
        useFixedBase=1,
    ),
    marble_cube=dict(
        fileName=os.path.join(
            BULLET3_ASSET_PATH, 'marble_cube.urdf'),
        basePosition=(.65, 0.3, -.3),
        baseOrientation=(0, 0, 0.707107, 0.707107),
        globalScaling=0.8,
        useFixedBase=1,
    ),
    basket=dict(
        fileName=os.path.join(
            BULLET3_ASSET_PATH, 'dinnerware/cup/cup_small.urdf'),
        basePosition=(.65, 0.3, -.3),
        baseOrientation=(0, 0, 0.707107, 0.707107),
        globalScaling=0.8,
        useFixedBase=1,
    ),
    button=dict(
        fileName=os.path.join(
            BASE_ASSET_PATH, 'button/button.urdf'),
        basePosition=(.7, 0.2, -.35),
        baseOrientation=(0, 0, 0.707107, 0.707107),
        globalScaling=0.1,
    ),
    spoon=dict(
        fileName=os.path.join(
            BASE_ASSET_PATH, 'spoon/spoon.urdf'),
        basePosition=(.6, 0.25, -.3),
        baseOrientation=(0, 0, 0.707107, 0.707107),
        globalScaling=0.25,
    ),
    ## Cuboid
    white_cuboid=dict(
        fileName=os.path.join(
            BASE_ASSET_PATH, 'cuboid/white_cuboid.urdf'),
        basePosition=(.65, 0.3, -.3),
        baseOrientation=(0, 0, 0.707107, 0.707107),
        globalScaling=0.8,
        useFixedBase=0,
    ),
    black_cuboid=dict(
        fileName=os.path.join(
            BASE_ASSET_PATH, 'cuboid/black_cuboid.urdf'),
        basePosition=(.65, 0.3, -.3),
        baseOrientation=(0, 0, 0.707107, 0.707107),
        globalScaling=0.8,
        useFixedBase=0,
    ),
    red_cuboid=dict(
        fileName=os.path.join(
            BASE_ASSET_PATH, 'cuboid/red_cuboid.urdf'),
        basePosition=(.65, 0.3, -.3),
        baseOrientation=(0, 0, 0.707107, 0.707107),
        globalScaling=0.8,
        useFixedBase=0,
    ),
    blue_cuboid=dict(
        fileName=os.path.join(
            BASE_ASSET_PATH, 'cuboid/blue_cuboid.urdf'),
        basePosition=(.65, 0.3, -.3),
        baseOrientation=(0, 0, 0.707107, 0.707107),
        globalScaling=0.8,
        useFixedBase=0,
    ),
    green_cuboid=dict(
        fileName=os.path.join(
            BASE_ASSET_PATH, 'cuboid/green_cuboid.urdf'),
        basePosition=(.65, 0.3, -.3),
        baseOrientation=(0, 0, 0.707107, 0.707107),
        globalScaling=0.8,
        useFixedBase=0,
    ),
    yellow_cuboid=dict(
        fileName=os.path.join(
            BASE_ASSET_PATH, 'cuboid/yellow_cuboid.urdf'),
        basePosition=(.65, 0.3, -.3),
        baseOrientation=(0, 0, 0.707107, 0.707107),
        globalScaling=0.8,
        useFixedBase=0,
    ),
    cyan_cuboid=dict(
        fileName=os.path.join(
            BASE_ASSET_PATH, 'cuboid/cyan_cuboid.urdf'),
        basePosition=(.65, 0.3, -.3),
        baseOrientation=(0, 0, 0.707107, 0.707107),
        globalScaling=0.8,
        useFixedBase=0,
    ),
    magenta_cuboid=dict(
        fileName=os.path.join(
            BASE_ASSET_PATH, 'cuboid/magenta_cuboid.urdf'),
        basePosition=(.65, 0.3, -.3),
        baseOrientation=(0, 0, 0.707107, 0.707107),
        globalScaling=0.8,
        useFixedBase=0,
    ),
    ## Cans
    tomato_can_pushable=dict(
        fileName=os.path.join(
            BASE_ASSET_PATH, 'can/tomato_can/tomato_can.urdf'),
        basePosition=(.6, 0.25, -.3),
        baseOrientation=(0, 0, 0.707107, 0.707107),
        globalScaling=0.25, 
    ),
    tuna_fish_can_pushable=dict(
        fileName=os.path.join(
            BASE_ASSET_PATH, 'can/tuna_fish_can/tuna_fish_can.urdf'),
        basePosition=(.6, 0.25, -.3),
        baseOrientation=(0, 0, 0.707107, 0.707107),
        globalScaling=0.25, 
    ),
    tuna_fish_can_tall_pushable=dict(
        fileName=os.path.join(
            BASE_ASSET_PATH, 'can/tuna_fish_can/tuna_fish_can_tall.urdf'),
        basePosition=(.6, 0.25, -.3),
        baseOrientation=(0, 0, 0.707107, 0.707107),
        globalScaling=0.25, 
    ),
    pepsi_can_pushable=dict(
        fileName=os.path.join(
            BASE_ASSET_PATH, 'can/pepsi_can/pepsi_can.urdf'),
        basePosition=(.6, 0.25, -.3),
        baseOrientation=(0, 0, 0.707107, 0.707107),
        globalScaling=0.25, 
    ),
    mountain_dew_can_pushable=dict(
        fileName=os.path.join(
            BASE_ASSET_PATH, 'can/mountain_dew_can/mountain_dew_can.urdf'),
        basePosition=(.6, 0.25, -.3),
        baseOrientation=(0, 0, 0.707107, 0.707107),
        globalScaling=0.25, 
    ),
    xylitol_can_pushable=dict(
        fileName=os.path.join(
            BASE_ASSET_PATH, 'can/xylitol_can/xylitol_can.urdf'),
        basePosition=(.6, 0.25, -.3),
        baseOrientation=(0, 0, 0.707107, 0.707107),
        globalScaling=0.25, 
    ),
    decaf_can_pushable=dict(
        fileName=os.path.join(
            BASE_ASSET_PATH, 'can/decaf_can/decaf_can.urdf'),
        basePosition=(.6, 0.25, -.3),
        baseOrientation=(0, 0, 0.707107, 0.707107),
        globalScaling=0.25, 
    ),
    white_can_pushable=dict(
        fileName=os.path.join(
            BASE_ASSET_PATH, 'can/can/white_can.urdf'),
        basePosition=(.6, 0.25, -.3),
        baseOrientation=(0, 0, 0.707107, 0.707107),
        globalScaling=0.25,  
    ),
    black_can_pushable=dict(
        fileName=os.path.join(
            BASE_ASSET_PATH, 'can/can/black_can.urdf'),
        basePosition=(.6, 0.25, -.3),
        baseOrientation=(0, 0, 0.707107, 0.707107),
        globalScaling=0.25,  
    ),
    red_can_pushable=dict(
        fileName=os.path.join(
            BASE_ASSET_PATH, 'can/can/red_can.urdf'),
        basePosition=(.6, 0.25, -.3),
        baseOrientation=(0, 0, 0.707107, 0.707107),
        globalScaling=0.25,  
    ),
    blue_can_pushable=dict(
        fileName=os.path.join(
            BASE_ASSET_PATH, 'can/can/blue_can.urdf'),
        basePosition=(.6, 0.25, -.3),
        baseOrientation=(0, 0, 0.707107, 0.707107),
        globalScaling=0.25,  
    ),
    green_can_pushable=dict(
        fileName=os.path.join(
            BASE_ASSET_PATH, 'can/can/green_can.urdf'),
        basePosition=(.6, 0.25, -.3),
        baseOrientation=(0, 0, 0.707107, 0.707107),
        globalScaling=0.25,  
    ),
    yellow_can_pushable=dict(
        fileName=os.path.join(
            BASE_ASSET_PATH, 'can/can/yellow_can.urdf'),
        basePosition=(.6, 0.25, -.3),
        baseOrientation=(0, 0, 0.707107, 0.707107),
        globalScaling=0.25,  
    ),
    cyan_can_pushable=dict(
        fileName=os.path.join(
            BASE_ASSET_PATH, 'can/can/cyan_can.urdf'),
        basePosition=(.6, 0.25, -.3),
        baseOrientation=(0, 0, 0.707107, 0.707107),
        globalScaling=0.25,  
    ),
    magenta_can_pushable=dict(
        fileName=os.path.join(
            BASE_ASSET_PATH, 'can/can/magenta_can.urdf'),
        basePosition=(.6, 0.25, -.3),
        baseOrientation=(0, 0, 0.707107, 0.707107),
        globalScaling=0.25,  
    ),
    ## Containers
    steel_pot_pushable=dict(
        fileName=os.path.join(
            BASE_ASSET_PATH, 'pot/steel_pot.urdf'),
        basePosition=(.65, 0.3, -.3),
        baseOrientation=(0, 0, 0.707107, 0.707107),
        globalScaling=0.8,
        useFixedBase=0,
    ),
    purple_pot_pushable=dict(
        fileName=os.path.join(
            BASE_ASSET_PATH, 'pot/purple_pot.urdf'),
        basePosition=(.65, 0.3, -.3),
        baseOrientation=(0, 0, 0.707107, 0.707107),
        globalScaling=0.8,
        useFixedBase=0,
    ),
    lime_pot_pushable=dict(
        fileName=os.path.join(
            BASE_ASSET_PATH, 'pot/lime_pot.urdf'),
        basePosition=(.65, 0.3, -.3),
        baseOrientation=(0, 0, 0.707107, 0.707107),
        globalScaling=0.8,
        useFixedBase=0,
    ),
    maroon_pot_pushable=dict(
        fileName=os.path.join(
            BASE_ASSET_PATH, 'pot/maroon_pot.urdf'),
        basePosition=(.65, 0.3, -.3),
        baseOrientation=(0, 0, 0.707107, 0.707107),
        globalScaling=0.8,
        useFixedBase=0,
    ),
    orange_pot_pushable=dict(
        fileName=os.path.join(
            BASE_ASSET_PATH, 'pot/orange_pot.urdf'),
        basePosition=(.65, 0.3, -.3),
        baseOrientation=(0, 0, 0.707107, 0.707107),
        globalScaling=0.8,
        useFixedBase=0,
    ),
    yellowbowl_pushable=dict(
        fileName=os.path.join(
            BASE_ASSET_PATH, 'dishware/yellowbowl/yellowbowl.urdf'),
        basePosition=(.65, 0.3, -.3),
        baseOrientation=(0, 0, 0.707107, 0.707107),
        globalScaling=0.8,
        useFixedBase=0,
    ),
    turqoisebowl_pushable=dict(
        fileName=os.path.join(
            BASE_ASSET_PATH, 'dishware/turqoisebowl/turqoisebowl.urdf'),
        basePosition=(.65, 0.3, -.3),
        baseOrientation=(0, 0, 0.707107, 0.707107),
        globalScaling=0.8,
        useFixedBase=0,
    ),
    ramekinbowl_pushable=dict(
        fileName=os.path.join(
            BASE_ASSET_PATH, 'dishware/ramekinbowl/ramekinbowl.urdf'),
        basePosition=(.65, 0.3, -.3),
        baseOrientation=(0, 0, 0.707107, 0.707107),
        globalScaling=0.8,
        useFixedBase=0,
    ),
    white_pot_pushable=dict(
        fileName=os.path.join(
            BASE_ASSET_PATH, 'pot/white_pot.urdf'),
        basePosition=(.65, 0.3, -.3),
        baseOrientation=(0, 0, 0.707107, 0.707107),
        globalScaling=0.8,
        useFixedBase=0,
    ),
    black_pot_pushable=dict(
        fileName=os.path.join(
            BASE_ASSET_PATH, 'pot/black_pot.urdf'),
        basePosition=(.65, 0.3, -.3),
        baseOrientation=(0, 0, 0.707107, 0.707107),
        globalScaling=0.8,
        useFixedBase=0,
    ),
    red_pot_pushable=dict(
        fileName=os.path.join(
            BASE_ASSET_PATH, 'pot/red_pot.urdf'),
        basePosition=(.65, 0.3, -.3),
        baseOrientation=(0, 0, 0.707107, 0.707107),
        globalScaling=0.8,
        useFixedBase=0,
    ),
    green_pot_pushable=dict(
        fileName=os.path.join(
            BASE_ASSET_PATH, 'pot/green_pot.urdf'),
        basePosition=(.65, 0.3, -.3),
        baseOrientation=(0, 0, 0.707107, 0.707107),
        globalScaling=0.8,
        useFixedBase=0,
    ),
    blue_pot_pushable=dict(
        fileName=os.path.join(
            BASE_ASSET_PATH, 'pot/blue_pot.urdf'),
        basePosition=(.65, 0.3, -.3),
        baseOrientation=(0, 0, 0.707107, 0.707107),
        globalScaling=0.8,
        useFixedBase=0,
    ),
    cyan_pot_pushable=dict(
        fileName=os.path.join(
            BASE_ASSET_PATH, 'pot/cyan_pot.urdf'),
        basePosition=(.65, 0.3, -.3),
        baseOrientation=(0, 0, 0.707107, 0.707107),
        globalScaling=0.8,
        useFixedBase=0,
    ),
    magenta_pot_pushable=dict(
        fileName=os.path.join(
            BASE_ASSET_PATH, 'pot/magenta_pot.urdf'),
        basePosition=(.65, 0.3, -.3),
        baseOrientation=(0, 0, 0.707107, 0.707107),
        globalScaling=0.8,
        useFixedBase=0,
    ),
    yellow_pot_pushable=dict(
        fileName=os.path.join(
            BASE_ASSET_PATH, 'pot/yellow_pot.urdf'),
        basePosition=(.65, 0.3, -.3),
        baseOrientation=(0, 0, 0.707107, 0.707107),
        globalScaling=0.8,
        useFixedBase=0,
    ),
)

PROGRAMMATIC_OBJECT_SPECS = dict(
    visual=dict(
        shapeType=p.GEOM_CYLINDER,
        length=0.05,
        radius=0.03,
    ),
    collision=dict(
        shapeType=p.GEOM_CYLINDER,
        height=0.05,
        radius=0.03,
    )
)

CYLINDER_COLORS = dict(
    red_cylinder=(1, 0, 0, 1),
    blue_cylinder=(0, 0, 1, 1),
    green_cylinder=(0, 1, 0, 1),
    yellow_cylinder=(1, 1, 0, 1),
    cyan_cylinder=(0, 1, 1, 1),
    magenta_cylinder=(1, 0, 1, 1),
    gray_cylinder=(0.5, 0.5, 0.5, 1),
    maroon_cylinder=(0.5, 0, 0, 1),
)