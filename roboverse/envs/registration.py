import gym
from roboverse.assets.shapenet_object_lists \
import GRASP_TRAIN_OBJECTS, GRASP_TEST_OBJECTS, OBJECT_SCALINGS, PICK_PLACE_TRAIN_OBJECTS, \
PICK_PLACE_TEST_OBJECTS, PUSH_TRAIN_OBJECTS, TRAIN_CONTAINERS, TEST_CONTAINERS, TRAIN_OBJECTS, \
CUBOID_OBJECTS, DIVERSE_PUSH_TRAIN_OBJECTS, TRAIN_CONTAINER_OBJECTS

ENVIRONMENT_SPECS = (
    {
        'id': 'Widow250Grasp-v0',
        'entry_point': 'roboverse.envs.widow250:Widow250Env',
        'kwargs': {'reward_type': 'grasping',
                   'control_mode': 'discrete_gripper',
                   'target_object': 'beer_bottle',
                   'load_tray': True,
                   'xyz_action_scale': 0.2,
                   }
    },
    {
        'id': 'Widow250GraspEasy-v0',
        'entry_point': 'roboverse.envs.widow250:Widow250Env',
        'kwargs': {'reward_type': 'grasping',
                   'control_mode': 'discrete_gripper',
                   'target_object': 'shed',
                   'object_names': ('shed',),
                   'object_scales': (0.7,),
                   'load_tray': False,
                   'xyz_action_scale': 0.2,
                   'object_position_high': (.6, .2, -.30),
                   'object_position_low': (.6, .2, -.30),
                   }
    },
    {
        'id': 'Widow250MultiTaskGrasp-v0',
        'entry_point': 'roboverse.envs.widow250:Widow250Env',
        'kwargs': {'reward_type': 'grasping',
                   'control_mode': 'discrete_gripper',
                   'object_names': ('shed', 'two_handled_vase'),
                   'object_scales': (0.7, 0.6),
                   'target_object': 'shed',
                   'load_tray': False,
                   'object_position_high': (.68, .25, -.30),
                   'object_position_low': (.53, .15, -.30),
                   'xyz_action_scale': 0.2,
                   }
    },
    {
        'id': 'Widow250MultiObjectGraspTrain-v0',
        'entry_point': 'roboverse.envs.widow250:Widow250MultiObjectEnv',
        'kwargs': {'reward_type': 'grasping',
                   'control_mode': 'discrete_gripper',
                   'possible_objects': GRASP_TRAIN_OBJECTS,
                   'num_objects': 2,

                   'load_tray': False,
                   'object_position_high': (.68, .25, -.30),
                   'object_position_low': (.53, .15, -.30),
                   'xyz_action_scale': 0.2,

                   }
    },
    {
        'id': 'Widow250MultiObjectGraspTest-v0',
        'entry_point': 'roboverse.envs.widow250:Widow250MultiObjectEnv',
        'kwargs': {'reward_type': 'grasping',
                   'control_mode': 'discrete_gripper',

                   'possible_objects': GRASP_TEST_OBJECTS,
                   'num_objects': 2,

                   'load_tray': False,
                   'object_position_high': (.68, .25, -.30),
                   'object_position_low': (.53, .15, -.30),
                   'xyz_action_scale': 0.2,

                   }
    },
    {
        'id': 'Widow250MultiThreeObjectGraspTrain-v0',
        'entry_point': 'roboverse.envs.widow250:Widow250MultiObjectEnv',
        'kwargs': {'reward_type': 'grasping',
                   'control_mode': 'discrete_gripper',
                   'possible_objects': GRASP_TRAIN_OBJECTS,
                   'num_objects': 3,

                   'load_tray': False,
                   'object_position_high': (.7, .25, -.30),
                   'object_position_low': (.5, .15, -.30),
                   'xyz_action_scale': 0.2,


                   # Next three entries are ignored
                   'object_names': ('beer_bottle', 'gatorade', 'shed'),
                   'object_scales': (0.7, 0.6, 0.8),
                   'object_orientations': (
                       (0, 0, 1, 0), (0, 0, 1, 0), (0, 0, 1, 0)),
                   }
    },
    {
        'id': 'Widow250MultiThreeObjectGraspTest-v0',
        'entry_point': 'roboverse.envs.widow250:Widow250MultiObjectEnv',
        'kwargs': {'reward_type': 'grasping',
                   'control_mode': 'discrete_gripper',

                   'possible_objects': GRASP_TEST_OBJECTS,
                   'num_objects': 3,

                   'load_tray': False,
                   'object_position_high': (.7, .25, -.30),
                   'object_position_low': (.5, .15, -.30),
                   'xyz_action_scale': 0.2,


                   # Next three entries are ignored
                   'object_names': ('beer_bottle', 'gatorade', 'shed'),
                   'object_scales': (0.7, 0.6, 0.8),
                   'object_orientations': (
                        (0, 0, 1, 0), (0, 0, 1, 0), (0, 0, 1, 0)),
                   }
    },
    {
        'id': 'Widow250SingleObjGrasp-v0',
        'entry_point': 'roboverse.envs.widow250:Widow250Env',
        'kwargs': {'reward_type': 'grasping',
                   'control_mode': 'discrete_gripper',
                   'object_names': ('shed',),
                   'object_scales': (0.7,),
                   'target_object': 'shed',
                   'load_tray': False,
                   'object_position_high': (.68, .25, -.30),
                   'object_position_low': (.53, .15, -.30),
                   'xyz_action_scale': 0.2,
                   }
    },
    # Pick and place environments
    {
        'id': 'Widow250PickPlaceCube-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceEnv',
        'kwargs': {
            'reward_type': 'pick_place',
            'control_mode': 'discrete_gripper',
            'observation_img_dim': 128,
            'object_names': ('cube',),
            'object_scales': (0.04,),
            'target_object': 'cube',
            'load_tray': False,
            'object_position_high': (.72, 0.28, -.3),
            'object_position_low': (.48, 0.12, -.3),
            'container_name': 'tray',
            'fixed_container_position': True,
            'transpose_image': False,
        }
    },
    {
        'id': 'Widow250PickPlacePosition-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlacePositionEnv',
        'kwargs': {
            'reward_type': 'pick_place',
            'control_mode': 'discrete_gripper',
            'observation_img_dim': 128,
            'object_names': ('shed', 'fountain_vase'),
            'object_scales': [OBJECT_SCALINGS['shed'], OBJECT_SCALINGS['fountain_vase']],
            'target_object': 'shed',
            'load_tray': True,
            'object_position_high': (.72, 0.33, -.3),
            'object_position_low': (.48, 0.17, -.3),
            'container_position_low': (.72, 0.33, -.3),
            'container_position_high': (.48, 0.17, -.3),
            'transpose_image': False,
            'camera_distance': 0.37,
            # 'show_place_target': True
        }
    },
    {
        'id': 'Widow250PickPlacePositionMultiObject-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlacePositionMultiObjectEnv',
        'kwargs': {
            'reward_type': 'pick_place',
            'control_mode': 'discrete_gripper',
            'observation_img_dim': 128,
            'possible_objects': PICK_PLACE_TRAIN_OBJECTS[:8],
            'num_objects': 2,
            'load_tray': True,
            'object_position_high': (.72, 0.33, -.3),
            'object_position_low': (.48, 0.17, -.3),
            'container_position_low': (.72, 0.33, -.3),
            'container_position_high': (.48, 0.17, -.3),
            'transpose_image': False,
            'camera_distance': 0.37,
            # 'show_place_target': True
        }
    },
    {
        'id': 'Widow250PickPlacePushPositionMultiObject-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlacePositionMultiObjectEnv',
        'kwargs': {
            'reward_type': 'pick_place',
            'control_mode': 'discrete_gripper',
            'observation_img_dim': 48,
            'possible_objects': PICK_PLACE_TRAIN_OBJECTS[:8] + PUSH_TRAIN_OBJECTS[:8],
            'num_objects': 3,
            'load_tray': True,
            'object_position_high': (.7, 0.31, -.3),
            'object_position_low': (.5, 0.19, -.3),
            'container_position_low': (.7, 0.31, -.3),
            'container_position_high': (.5, 0.19, -.3),
            'transpose_image': False,
            'camera_distance': 0.37,
            # 'show_place_target': True
        }
    },
    {
        'id': 'Widow250DiversePickPlacePositionMultiObjectCluttered-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlacePositionMultiObjectEnv',
        'kwargs': {
            'reward_type': 'pick_place',
            'control_mode': 'discrete_gripper',
            'observation_img_dim': 128,
            'possible_objects': list(set(TRAIN_OBJECTS + GRASP_TRAIN_OBJECTS + PICK_PLACE_TRAIN_OBJECTS)),
            'num_objects': 4,
            'vary_num_objects_range': (1, 4),
            'load_tray': True,
            'object_position_high': (.72, 0.33, -.3),
            'object_position_low': (.48, 0.17, -.3),
            'container_position_low': (.72, 0.33, -.3),
            'container_position_high': (.48, 0.17, -.3),
            'min_distance_between_objects': 0.0,
            # 'min_distance_from_object': 0.0,
            'random_object_pose': True,
            'transpose_image': False,
            'camera_distance': 0.37,
            'ee_pos_high': (.77, 0.37, -.17),
            'ee_pos_low': (.43, 0.12, -.34),
            'random_ee_pose': True,
            # 'show_place_target': True
        }
    },
    {
        'id': 'Widow250DiversePickPlacePositionMultiObject-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlacePositionMultiObjectEnv',
        'kwargs': {
            'reward_type': 'pick_place',
            'control_mode': 'discrete_gripper',
            'observation_img_dim': 128,
            'possible_objects': list(set(TRAIN_OBJECTS + GRASP_TRAIN_OBJECTS + PICK_PLACE_TRAIN_OBJECTS)),
            'num_objects': 4,
            'vary_num_objects_range': (1, 4),
            'load_tray': True,
            'object_position_high': (.72, 0.33, -.3),
            'object_position_low': (.48, 0.17, -.3),
            'container_position_low': (.72, 0.33, -.3),
            'container_position_high': (.48, 0.17, -.3),
            # 'min_distance_between_objects': 0.0,
            # 'min_distance_from_object': 0.0,
            'random_object_pose': True,
            'transpose_image': False,
            'camera_distance': 0.37,
            'ee_pos_high': (.77, 0.37, -.17),
            'ee_pos_low': (.43, 0.12, -.34),
            'random_ee_pose': True,
            # 'show_place_target': True
        }
    },
    {
        'id': 'Widow250DiversePickPlacePositionMultiObjectFixGripper-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlacePositionMultiObjectEnv',
        'kwargs': {
            'reward_type': 'pick_place',
            'control_mode': 'discrete_gripper',
            'observation_img_dim': 128,
            'possible_objects': list(set(TRAIN_OBJECTS + GRASP_TRAIN_OBJECTS + PICK_PLACE_TRAIN_OBJECTS)),
            'num_objects': 4,
            'vary_num_objects_range': (1, 4),
            'load_tray': True,
            'object_position_high': (.72, 0.33, -.3),
            'object_position_low': (.48, 0.17, -.3),
            'container_position_low': (.72, 0.33, -.3),
            'container_position_high': (.48, 0.17, -.3),
            # 'min_distance_between_objects': 0.0,
            # 'min_distance_from_object': 0.0,
            'random_object_pose': True,
            'transpose_image': False,
            'camera_distance': 0.37,
            'ee_pos_high': (.77, 0.37, -.17),
            'ee_pos_low': (.43, 0.12, -.34),
            'random_ee_pose': False,
            # 'show_place_target': True
        }
    },
    {
        'id': 'Widow250DiversePickPlacePositionMultiObjectFixGripperFixObjQuat-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlacePositionMultiObjectEnv',
        'kwargs': {
            'reward_type': 'pick_place',
            'control_mode': 'discrete_gripper',
            'observation_img_dim': 128,
            'possible_objects': list(set(TRAIN_OBJECTS + GRASP_TRAIN_OBJECTS + PICK_PLACE_TRAIN_OBJECTS)),
            'num_objects': 4,
            'vary_num_objects_range': (1, 4),
            'load_tray': True,
            'object_position_high': (.72, 0.33, -.3),
            'object_position_low': (.48, 0.17, -.3),
            'container_position_low': (.72, 0.33, -.3),
            'container_position_high': (.48, 0.17, -.3),
            # 'min_distance_between_objects': 0.0,
            # 'min_distance_from_object': 0.0,
            'random_object_pose': False,
            'transpose_image': False,
            'camera_distance': 0.37,
            'ee_pos_high': (.77, 0.37, -.17),
            'ee_pos_low': (.43, 0.12, -.34),
            'random_ee_pose': False,
            # 'show_place_target': True
        }
    },
    {
        'id': 'Widow250TableArrangement-v0',
        'entry_point': 'roboverse.envs.widow250_tablearrange:Widow250TableArrangementEnv',
        'kwargs': {
            'reward_type': 'pick_place',
            'control_mode': 'discrete_gripper',
            'observation_img_dim': 128,
            'possible_container_objects': ['steel_pot_pushable'],
            'possible_utensil_objects': ['spoon'],
            'possible_pickplace_objects': ['spam'],
            'possible_push_objects': ['tomato_can_pushable'],
            'num_objects': 4,
            'load_tray': True,
            'object_position_high': (.72, 0.32, -.3),
            'object_position_low': (.48, 0.18, -.3),
            'container_position_low': (.72, 0.33, -.3),
            'container_position_high': (.48, 0.17, -.3),
            'min_distance_between_objects': 0.05,
            'min_distance_from_object': 0.05,
            'place_success_height_threshold': -0.3,
            'place_success_radius_threshold': 0.07,
            'transpose_image': False,
            'camera_distance': .37,
            'ee_pos_high': (.77, 0.37, -.17),
            'ee_pos_low': (.43, 0.12, -.34),
            'random_ee_pose': False,
            'num_sim_steps_reset': 200,
            # 'show_place_target': True
        }
    },
    {
        'id': 'Widow250DiverseTableArrangement-v0',
        'entry_point': 'roboverse.envs.widow250_tablearrange:Widow250TableArrangementEnv',
        'kwargs': {
            'reward_type': 'pick_place',
            'control_mode': 'discrete_gripper',
            'observation_img_dim': 128,
            'possible_container_objects': list(TRAIN_CONTAINER_OBJECTS),
            'possible_utensil_objects': list(set(TRAIN_OBJECTS + GRASP_TRAIN_OBJECTS + PICK_PLACE_TRAIN_OBJECTS)), 
            'possible_pickplace_objects': list(set(TRAIN_OBJECTS + GRASP_TRAIN_OBJECTS + PICK_PLACE_TRAIN_OBJECTS + CUBOID_OBJECTS)),
            'possible_push_objects': list(DIVERSE_PUSH_TRAIN_OBJECTS),
            'num_objects': 4,
            'load_tray': True,
            'object_position_high': (.72, 0.32, -.3),
            'object_position_low': (.48, 0.18, -.3),
            'container_position_low': (.72, 0.33, -.3),
            'container_position_high': (.48, 0.17, -.3),
            'min_distance_between_objects': 0.05,
            'min_distance_from_object': 0.05,
            'place_success_height_threshold': -0.3,
            'place_success_radius_threshold': 0.07,
            'transpose_image': False,
            'camera_distance': .37,
            'ee_pos_high': (.77, 0.37, -.17),
            'ee_pos_low': (.43, 0.12, -.34),
            'random_ee_pose': False,
            'num_sim_steps_reset': 200,
            # 'show_place_target': True
        }
    },
    {
        'id': 'Widow250PickPlacePositionMultiObjectTest-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlacePositionMultiObjectEnv',
        'kwargs': {
            'reward_type': 'pick_place',
            'control_mode': 'discrete_gripper',
            'observation_img_dim': 128,
            'possible_objects': PICK_PLACE_TEST_OBJECTS[:4],
            'num_objects': 2,
            'load_tray': False,
            'object_position_high': (.72, 0.28, -.3),
            'object_position_low': (.48, 0.12, -.3),
            'container_position_low': (.72, 0.28, -.3),
            'container_position_high': (.48, 0.12, -.3),
            'transpose_image': False,
            # 'camera_yaw': 20,
            # 'show_place_target': True
        }
    },
    {
        'id': 'Widow250PickPlaceEasy-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('shed',),
                   'object_scales': (0.7,),
                   'target_object': 'shed',
                   'load_tray': False,
                   'object_position_low': (.6, .2, -.3),
                   'object_position_high': (.6, .2, -.3),

                   'container_name': 'bowl_small',
                   'fixed_container_position': True,

                   }
    },
    {
        'id': 'Widow250PickPlace-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('shed',),
                   'object_scales': (0.7,),
                   'target_object': 'shed',
                   'load_tray': False,
                   'object_position_low': (.49, .18, -.30),
                   'object_position_high': (.59, .27, -.30),

                   'container_name': 'bowl_small',


                   }
    },
    {
        'id': 'Widow250PickPlaceMultiObjectMultiContainerTrain-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace'
                       ':Widow250PickPlaceMultiObjectMultiContainerEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'load_tray': False,
                   'num_objects': 2,

                   'possible_objects': PICK_PLACE_TRAIN_OBJECTS,
                   'possible_containers': TRAIN_CONTAINERS,

                   }
    },
    {
        'id': 'Widow250PickPlaceMultiObjectMultiContainerTest-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace'
                       ':Widow250PickPlaceMultiObjectMultiContainerEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'load_tray': False,
                   'num_objects': 2,

                   'possible_objects': PICK_PLACE_TEST_OBJECTS,
                   'possible_containers': TEST_CONTAINERS,
                   }
    },
    {
        'id': 'Widow250PickTray-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('shed',),
                   'object_scales': (0.7,),
                   'target_object': 'shed',
                   'load_tray': False,
                   'object_position_low': (.49, .18, -.30),
                   'object_position_high': (.75, .27, -.30),

                   'container_name': 'tray',
                   'fixed_container_position': True,

                   'use_neutral_action': True,
                   'neutral_gripper_open': False,
                   }
    },
    {
        'id': 'Widow250PlaceTray-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('shed',),
                   'object_scales': (0.7,),
                   'target_object': 'shed',
                   'load_tray': False,
                   'object_position_low': (.49, .18, -.30),
                   'object_position_high': (.75, .27, -.30),

                   'container_name': 'tray',
                   'fixed_container_position': True,
                   'start_object_in_gripper': True,

                   'use_neutral_action': True,
                   'neutral_gripper_open': False,
                   }
    },
    {
        'id': 'Widow250SinglePutInBowl-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('shed',),
                   'object_scales': (0.7,),
                   'target_object': 'shed',
                   'load_tray': False,
                   'object_position_low': (.49, .18, -.30),
                   'object_position_high': (.59, .27, -.30),

                   'container_name': 'bowl_small',


                   }
    },
    {
        'id': 'Widow250SinglePutInBowlRandomBowlPosition-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('shed',),
                   'object_scales': (0.7,),
                   'target_object': 'shed',
                   'load_tray': False,
                   'object_position_low': (.5, .18, -.30),
                   'object_position_high': (.7, .27, -.30),

                   'container_name': 'bowl_small',


                   }
    },
    {
        'id': 'Widow250PutInBowlRandomBowlPosition-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('shed', 'two_handled_vase'),
                   'object_scales': (0.7, 0.6),
                   'target_object': 'shed',
                   'load_tray': False,
                   'object_position_low': (.5, .18, -.30),
                   'object_position_high': (.7, .27, -.30),

                   'container_name': 'bowl_small',


                   }
    },
    {
        'id': 'Widow250MultiObjectPutInBowlRandomBowlPositionTrain-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceMultiObjectEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'possible_objects': PICK_PLACE_TRAIN_OBJECTS,
                   'num_objects': 2,
                   'load_tray': False,
                   'object_position_low': (.5, .18, -.30),
                   'object_position_high': (.7, .27, -.30),

                   'container_name': 'bowl_small',


                   }
    },
    {
        'id': 'Widow250MultiObjectPutInBowlRandomBowlPositionTest-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceMultiObjectEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'possible_objects': PICK_PLACE_TEST_OBJECTS,
                   'num_objects': 2,
                   'load_tray': False,
                   'object_position_low': (.5, .18, -.30),
                   'object_position_high': (.7, .27, -.30),

                   'container_name': 'bowl_small',


                   }
    },
    {
        'id': 'Widow250PutInTray-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('shed', 'two_handled_vase'),
                   'object_scales': (0.7, 0.6),
                   'target_object': 'shed',
                   'load_tray': False,
                   'object_position_low': (.49, .18, -.30),
                   'object_position_high': (.59, .27, -.30),

                   'container_name': 'tray',

                   }
    },
    {
        'id': 'Widow250PutInBox-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('shed', 'two_handled_vase'),
                   'object_scales': (0.7, 0.6),
                   'target_object': 'shed',
                   'load_tray': False,
                   'object_position_low': (.49, .18, -.30),
                   'object_position_high': (.59, .27, -.30),

                   'container_name': 'open_box',

                   }
    },
    {
        'id': 'Widow250PlaceOnCube-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('shed', 'two_handled_vase'),
                   'object_scales': (0.7, 0.6),
                   'target_object': 'shed',
                   'load_tray': False,
                   'object_position_low': (.49, .18, -.30),
                   'object_position_high': (.59, .27, -.30),

                   'container_name': 'cube',

                   }
    },
    {
        'id': 'Widow250PutInPanTefal-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('shed', 'two_handled_vase'),
                   'object_scales': (0.7, 0.6),
                   'target_object': 'shed',
                   'load_tray': False,
                   'object_position_low': (.49, .18, -.30),
                   'object_position_high': (.69, .27, -.30),

                   'container_name': 'pan_tefal',

                   }
    },
    {
        'id': 'Widow250PutInPanTefalTestRL1-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('square_rod_embellishment',
                                    'grill_trash_can'),
                   'object_scales': (0.6, 0.5),
                   'object_orientations': ((0, 0, 1, 0), (0, 0.707, 0.707, 0)),
                   'target_object': 'square_rod_embellishment',

                   'load_tray': False,
                   'container_name': 'pan_tefal',

                   }
    },
    {
        'id': 'Widow250PutInPanTefalFixedTestRL1-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('square_rod_embellishment',
                                    'grill_trash_can'),
                   'object_scales': (0.6, 0.5),
                   'object_orientations': ((0, 0, 1, 0), (0, 0.707, 0.707, 0)),
                   'target_object': 'square_rod_embellishment',

                   'load_tray': False,
                   'container_name': 'pan_tefal',
                   'fixed_container_position': True,
                   }
    },
    {
        'id': 'Widow250PutInPanTefalTestRL2-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('shed', 'sack_vase'),
                   'object_scales': (0.6, 0.6),
                   'object_orientations': ((0, 0, 1, 0), (0, 0.707, 0.707, 0)),
                   'target_object': 'shed',

                   'load_tray': False,
                   'container_name': 'pan_tefal',
                   }
    },
    {
        'id': 'Widow250PutInPanTefalTestRL3-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('two_handled_vase',
                                    'thick_wood_chair',),
                   'object_scales': (0.45, 0.4),
                   'object_orientations': ((0, 0, 1, 0), (0, 0, 1, 0)),
                   'target_object': 'two_handled_vase',

                   'load_tray': False,
                   'container_name': 'pan_tefal',

                   }
    },
    {
        'id': 'Widow250PutInPanTefalRL4-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('curved_handle_cup',
                                    'baseball_cap',),
                   'object_scales': (0.5, 0.5),
                   'object_orientations': ((0, 0.707, 0.707, 0),
                                           (0, -0.707, 0.707, 0)),
                   'target_object': 'curved_handle_cup',

                   'load_tray': False,
                   'container_name': 'pan_tefal',
                   }
    },
    {
        'id': 'Widow250PutInTableTop-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('shed', 'two_handled_vase'),
                   'object_scales': (0.7, 0.6),
                   'target_object': 'shed',
                   'load_tray': False,
                   'object_position_low': (.49, .18, -.30),
                   'object_position_high': (.69, .27, -.30),

                   'container_name': 'table_top',

                   }
    },
    {
        'id': 'Widow250PutOnTorus-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('shed', 'two_handled_vase'),
                   'object_scales': (0.7, 0.6),
                   'target_object': 'shed',
                   'load_tray': False,
                   'object_position_low': (.49, .18, -.30),
                   'object_position_high': (.69, .27, -.30),

                   'container_name': 'torus',
                   }
    },
    {
        'id': 'Widow250PutOnCubeConcave-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('shed', 'two_handled_vase'),
                   'object_scales': (0.7, 0.6),
                   'target_object': 'shed',
                   'load_tray': False,
                   'object_position_low': (.49, .18, -.30),
                   'object_position_high': (.69, .27, -.30),

                   'container_name': 'cube_concave',

                   }
    },
    {
        'id': 'Widow250PutOnPlate-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('shed', 'two_handled_vase'),
                   'object_scales': (0.7, 0.6),
                   'target_object': 'shed',
                   'load_tray': False,
                   'object_position_low': (.49, .18, -.30),
                   'object_position_high': (.69, .27, -.30),

                   'container_name': 'plate',

                   }
    },
    {
        'id': 'Widow250PutOnHusky-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('shed', 'two_handled_vase'),
                   'object_scales': (0.7, 0.6),
                   'target_object': 'shed',
                   'load_tray': False,
                   'object_position_low': (.49, .18, -.30),
                   'object_position_high': (.69, .27, -.30),

                   'container_name': 'husky',

                   }
    },
    {
        'id': 'Widow250PutOnMarbleCube-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('shed', 'two_handled_vase'),
                   'object_scales': (0.7, 0.6),
                   'target_object': 'shed',
                   'load_tray': False,
                   'object_position_low': (.49, .18, -.30),
                   'object_position_high': (.69, .27, -.30),

                   'container_name': 'marble_cube',

                   }
    },
    {
        'id': 'Widow250PutOnMarbleCubeTestRL1-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('square_rod_embellishment',
                                    'grill_trash_can'),
                   'object_scales': (0.6, 0.5),
                   'object_orientations': ((0, 0, 1, 0), (0, 0.707, 0.707, 0)),
                   'target_object': 'square_rod_embellishment',

                   'load_tray': False,

                   'container_name': 'marble_cube',

                   }
    },
    {
        'id': 'Widow250PutOnMarbleCubeTestRL2-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('shed', 'sack_vase'),
                   'object_scales': (0.6, 0.6),
                   'object_orientations': ((0, 0, 1, 0), (0, 0.707, 0.707, 0)),
                   'target_object': 'shed',

                   'load_tray': False,
                   'container_name': 'marble_cube',

                   }
    },
    {
        'id': 'Widow250PutOnMarbleCubeTestRL3-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('two_handled_vase',
                                    'thick_wood_chair',),
                   'object_scales': (0.45, 0.4),
                   'object_orientations': ((0, 0, 1, 0), (0, 0, 1, 0)),
                   'target_object': 'two_handled_vase',

                   'load_tray': False,
                   'container_name': 'marble_cube',

                   }
    },
    {
        'id': 'Widow250PutOnMarbleCubeTestRL4-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('curved_handle_cup',
                                    'baseball_cap',),
                   'object_scales': (0.5, 0.5),
                   'object_orientations': ((0, 0.707, 0.707, 0),
                                           (0, -0.707, 0.707, 0)),
                   'target_object': 'curved_handle_cup',

                   'load_tray': False,
                   'container_name': 'marble_cube',

                   }
    },
    {
        'id': 'Widow250PutOnMarbleCubeFixedTestRL4-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('curved_handle_cup',
                                    'baseball_cap',),
                   'object_scales': (0.5, 0.5),
                   'object_orientations': ((0, 0.707, 0.707, 0),
                                           (0, -0.707, 0.707, 0)),
                   'target_object': 'curved_handle_cup',

                   'load_tray': False,
                   'container_name': 'marble_cube',
                   'fixed_container_position': True,

                   }
    },
    {
        'id': 'Widow250PutInBasket-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('shed', 'two_handled_vase'),
                   'object_scales': (0.7, 0.6),
                   'target_object': 'shed',

                   'load_tray': False,
                   'container_name': 'basket',

                   }
    },
    {
        'id': 'Widow250PutInBasketTestRL1-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('square_rod_embellishment',
                                    'grill_trash_can'),
                   'object_scales': (0.6, 0.5),
                   'object_orientations': ((0, 0, 1, 0), (0, 0.707, 0.707, 0)),
                   'target_object': 'square_rod_embellishment',

                   'load_tray': False,
                   'container_name': 'basket',

                   }
    },
    {
        'id': 'Widow250PutInBasketTestRL2-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('shed', 'sack_vase'),
                   'object_scales': (0.6, 0.6),
                   'object_orientations': ((0, 0, 1, 0), (0, 0.707, 0.707, 0)),
                   'target_object': 'shed',

                   'load_tray': False,
                   'container_name': 'basket',

                   }
    },
    {
        'id': 'Widow250PutInBasketFixedTestRL2-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('shed', 'sack_vase'),
                   'object_scales': (0.6, 0.6),
                   'object_orientations': ((0, 0, 1, 0), (0, 0.707, 0.707, 0)),
                   'target_object': 'shed',

                   'load_tray': False,
                   'container_name': 'basket',
                   'fixed_container_position': True,

                   }
    },
    {
        'id': 'Widow250PutOnCheckerboardTable-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('shed', 'two_handled_vase'),
                   'object_scales': (0.7, 0.6),
                   'target_object': 'shed',
                   'load_tray': False,
                   'object_position_low': (.49, .18, -.30),
                   'object_position_high': (.69, .27, -.30),

                   'container_name': 'checkerboard_table',

                   }
    },
    {
        'id': 'Widow250PutOnCheckerboardTableTestRL3-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('two_handled_vase',
                                    'thick_wood_chair',),
                   'object_scales': (0.45, 0.4),
                   'object_orientations': ((0, 0, 1, 0), (0, 0, 1, 0)),
                   'target_object': 'two_handled_vase',

                   'load_tray': False,
                   'container_name': 'checkerboard_table',

                   }
    },
    {
        'id': 'Widow250PutOnCheckerboardTableFixedTestRL3-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('two_handled_vase',
                                    'thick_wood_chair',),
                   'object_scales': (0.45, 0.4),
                   'object_orientations': ((0, 0, 1, 0), (0, 0, 1, 0)),
                   'target_object': 'two_handled_vase',

                   'load_tray': False,
                   'container_name': 'checkerboard_table',
                   'fixed_container_position': True,

                   }
    },
    {
        'id': 'Widow250PutOnCheckerboardTableTestRL4-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('curved_handle_cup',
                                    'baseball_cap',),
                   'object_scales': (0.5, 0.5),
                   'object_orientations': ((0, 0.707, 0.707, 0),
                                           (0, -0.707, 0.707, 0)),
                   'target_object': 'curved_handle_cup',

                   'load_tray': False,
                   'container_name': 'checkerboard_table',

                   }
    },
    # Drawer environments
    {
        'id': 'Widow250DrawerOpen-v0',
        'entry_point': 'roboverse.envs.widow250_drawer:Widow250DrawerEnv',
        'kwargs': {'reward_type': 'opening',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('ball',),
                   'object_scales': (0.75,),
                   'target_object': 'ball',
                   'load_tray': False,
                   }
    },
    {
        'id': 'Widow250DrawerOpenNeutral-v0',
        'entry_point': 'roboverse.envs.widow250_drawer:Widow250DrawerEnv',
        'kwargs': {'reward_type': 'opening',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('ball',),
                   'object_scales': (0.75,),
                   'target_object': 'ball',
                   'load_tray': False,
                   'use_neutral_action': True
                   }
    },
    {
        'id': 'Widow250DrawerGrasp-v0',
        'entry_point': 'roboverse.envs.widow250_drawer:Widow250DrawerEnv',
        'kwargs': {'reward_type': 'grasping',
                   'control_mode': 'discrete_gripper',
                   'start_opened': True,
                   'object_names': ('ball',),
                   'object_scales': (0.75,),
                   'target_object': 'ball',
                   'load_tray': False,
                   }
    },
    {
        'id': 'Widow250DrawerGraspNeutral-v0',
        'entry_point': 'roboverse.envs.widow250_drawer:Widow250DrawerEnv',
        'kwargs': {'reward_type': 'grasping',
                   'control_mode': 'discrete_gripper',
                   'start_opened': True,
                   'object_names': ('ball',),
                   'object_scales': (0.75,),
                   'target_object': 'ball',
                   'load_tray': False,
                   'use_neutral_action': True
                   }
    },
    {
        'id': 'Widow250DrawerOpenGrasp-v0',
        'entry_point': 'roboverse.envs.widow250_drawer:Widow250DrawerEnv',
        'kwargs': {'reward_type': 'grasping',
                   'control_mode': 'discrete_gripper',
                   'object_names': ('ball',),
                   'object_scales': (0.75,),
                   'target_object': 'ball',
                   'load_tray': False,
                   'use_neutral_action': True
                   }
    },
    {
        'id': 'Widow250DrawerRandomizedOpen-v0',
        'entry_point': 'roboverse.envs.widow250_drawer:Widow250DrawerRandomizedEnv',
        'kwargs': {'reward_type': 'opening',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('ball',),
                   'object_scales': (0.75,),
                   'target_object': 'ball',
                   'load_tray': False,
                   }
    },
    {
        'id': 'Widow250DoubleDrawerOpenNeutral-v0',
        'entry_point': 'roboverse.envs.widow250_drawer:Widow250DoubleDrawerEnv',
        'kwargs': {'drawer_pos': (0.47, 0.2, -.35),
                   'reward_type': 'opening',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('ball',),
                   'object_scales': (0.75,),
                   'target_object': 'ball',
                   'load_tray': False,
                   'use_neutral_action': True
                   }
    },
    {
        'id': 'Widow250DoubleDrawerOpenGraspNeutral-v0',
        'entry_point': 'roboverse.envs.widow250_drawer:Widow250DoubleDrawerEnv',
        'kwargs': {'drawer_pos': (0.47, 0.2, -.35),
                   'reward_type': 'grasping',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('ball',),
                   'object_scales': (0.75,),
                   'target_object': 'ball',
                   'load_tray': False,
                   'use_neutral_action': True
                   }
    },
    {
        'id': 'Widow250DoubleDrawerGraspNeutral-v0',
        'entry_point': 'roboverse.envs.widow250_drawer:Widow250DoubleDrawerEnv',
        'kwargs': {'drawer_pos': (0.47, 0.2, -.35),
                   'reward_type': 'grasping',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('ball',),
                   'object_scales': (0.75,),
                   'target_object': 'ball',
                   'load_tray': False,
                   'start_opened': True,
                   'use_neutral_action': True
                   }
    },
    {
        'id': 'Widow250DoubleDrawerPickPlaceOpenGraspNeutral-v0',
        'entry_point': 'roboverse.envs.widow250_drawer:Widow250DoubleDrawerEnv',
        'kwargs': {'drawer_pos': (0.47, 0.2, -.35),
                   'reward_type': 'grasping',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('ball',),
                   'object_scales': (0.75,),
                   'target_object': 'ball',
                   'load_tray': False,
                   'start_opened': False,
                   'use_neutral_action': True,
                   'blocking_object_in_tray': False,
                   }
    },
    {
        'id': 'Widow250DoubleDrawerCloseOpenNeutral-v0',
        'entry_point': 'roboverse.envs.widow250_drawer:Widow250DoubleDrawerEnv',
        'kwargs': {'drawer_pos': (0.47, 0.2, -.35),
                   'reward_type': 'opening',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('ball',),
                   'object_scales': (0.75,),
                   'target_object': 'ball',
                   'load_tray': False,
                   'start_opened': False,
                   'start_top_opened': True,
                   'use_neutral_action': True,
                   }
    },
    {
        'id': 'Widow250DoubleDrawerCloseOpenGraspNeutral-v0',
        'entry_point': 'roboverse.envs.widow250_drawer:Widow250DoubleDrawerEnv',
        'kwargs': {'drawer_pos': (0.47, 0.2, -.35),
                   'reward_type': 'grasping',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('ball',),
                   'object_scales': (0.75,),
                   'target_object': 'ball',
                   'load_tray': False,
                   'start_opened': False,
                   'start_top_opened': True,
                   'use_neutral_action': True,
                   }
    },
    {

        'id': 'Widow250DrawerRandomizedOpenTwoObjGrasp-v0',
        'entry_point': 'roboverse.envs.widow250_drawer:Widow250DrawerRandomizedEnv',
        'kwargs': {'reward_type': 'opening',
                   'control_mode': 'discrete_gripper',

                   'object_names': ("shed", "sack_vase"),
                   'object_scales': (0.6, 0.6),
                   'object_orientations': ((0, 0, 1, 0), (0, 0.707, 0.707, 0)),
                   'target_object': "shed",
                   'load_tray': False,
                   }
    },
    # Button environments
    {
        'id': 'Widow250ButtonPress-v0',
        'entry_point': 'roboverse.envs.widow250_button:Widow250ButtonEnv',
        'kwargs': {'control_mode': 'discrete_gripper',
                   'load_tray': False,
                   }
    },
    {
        'id': 'Widow250ButtonPressTwoObjGrasp-v0',
        'entry_point': 'roboverse.envs.widow250_button:Widow250ButtonEnv',
        'kwargs': {'control_mode': 'discrete_gripper',

                   'object_names': ("shed", "sack_vase"),
                   'object_scales': (0.6, 0.6),
                   'object_orientations': ((0, 0, 1, 0), (0, 0.707, 0.707, 0)),
                   'object_position_high': (.75, .25, -.30),
                   'object_position_low': (.6, .1, -.30),
                   'target_object': "shed",
                   'load_tray': False,
                   }
    },
    {
        'id': 'Widow250RandPosButtonPressTwoObjGrasp-v0',
        'entry_point': 'roboverse.envs.widow250_button:Widow250ButtonEnv',
        'kwargs': {'control_mode': 'discrete_gripper',
                   'button_pos_low': (0.5, 0.25, -.34),
                   'button_pos_high': (0.55, 0.15, -.34),

                   'object_names': ("shed", "sack_vase"),
                   'object_scales': (0.6, 0.6),
                   'object_orientations': ((0, 0, 1, 0), (0, 0.707, 0.707, 0)),
                   'object_position_high': (.75, .25, -.30),
                   'object_position_low': (.65, .1, -.30),
                   'target_object': "shed",
                   'load_tray': False,
                   }
    },
)


def register_environments():
    for env in ENVIRONMENT_SPECS:
        gym.register(**env)

    gym_ids = tuple(
        environment_spec['id']
        for environment_spec in ENVIRONMENT_SPECS)

    return gym_ids


def make(env_name, *args, **kwargs):
    env = gym.make(env_name, *args, **kwargs)
    return env
