from abc import abstractmethod
import numpy as np
import pybullet as p

from roboverse.bullet.control import deg_to_quat, quat_to_deg	
from roboverse.bullet import object_utils
from roboverse.bullet.drawer_utils import open_drawer, close_drawer

MAX_ATTEMPTS_TO_GENERATE_OBJECT_POSITIONS = 200

class SawyerObjectUtil():
    def __init__(
        self,
        gripper_pos_low,
        gripper_pos_high,
    ):
        self.gripper_pos_low = gripper_pos_low
        self.gripper_pos_high = gripper_pos_high

    @abstractmethod
    def generate_obj_config(self):
        pass

    def spawn_obj(self):
        config = self.generate_obj_config()
        obj = object_utils.load_object(**config)

        return obj

class SawyerDrawerWithTrayObjectUtil(SawyerObjectUtil):
    def __init__(
        self,
        name='multicolor_drawer',
        quadrants = [
            [.525, .1675],
            [.525, -.1675],
            [.775, -.1675],
            [.775, .1675],
        ],
        z = -.34,
        scale = .11,
        drawer_close_coeff = 0.15134,
        drawer_open_coeff = 0.2695,
        tray_name='multicolor_tray',
        tray_offset = [0, 0, .059],
        tray_scale = 0.165,
        **kwargs,
    ):
        self.name = name
        self.quadrants = quadrants
        self.z = z
        self.scale = scale
        self.drawer_close_coeff = drawer_close_coeff
        self.drawer_open_coeff = drawer_open_coeff
        self.tray_name = tray_name
        self.tray_offset = tray_offset
        self.tray_scale = tray_scale
        super().__init__(**kwargs)

    def generate_obj_config(self):
        do_close_drawer = np.random.uniform() < .5

        drawer_quadrant_i = np.random.choice([0, 1])
        drawer_quadrant = self.quadrants[drawer_quadrant_i]
        drawer_frame_pos = np.array([
            drawer_quadrant[0], 
            drawer_quadrant[1], 
            self.z,
        ])

        if drawer_quadrant_i == 0:
            drawer_yaw = np.random.uniform(0, 90)
        else:
            drawer_yaw = np.random.uniform(90, 180)
        drawer_quat = deg_to_quat([0, 0, drawer_yaw])
        
        config = {
            'object_name': self.name,
            'object_position': drawer_frame_pos,
            'object_quat': drawer_quat,
            'scale': self.scale,

            'do_close_drawer': do_close_drawer,
        }
        return config
    
    def spawn_obj(self):
        config = self.generate_obj_config()
        do_close_drawer = config.pop('do_close_drawer')
        drawer = object_utils.load_object(**config)

        if do_close_drawer:
            close_drawer(drawer)
        else:
            open_drawer(drawer)

        tray_config = {
            'object_name': self.tray_name,
            'object_position': config['object_position'] + self.tray_offset,
            'object_quat': config['object_quat'],
            'scale': self.tray_scale
        }
        tray = object_utils.load_object(**tray_config)

        return drawer, tray
        
    def _get_drawer_handle_future_pos(self, drawer_frame_pos=None, drawer_yaw=None, coeff=0):
        return drawer_frame_pos + coeff * np.array([
            np.sin(drawer_yaw * np.pi / 180), 
            -np.cos(drawer_yaw * np.pi / 180), 
            0
        ])

class SawyerPushObjectUtil(SawyerObjectUtil):
    def __init__(
        self,
        name='cylinder',
        quadrants = [
            [0.61, 0.09],
            [0.61, -0.09],
            [0.73, -0.09],
            [0.73, 0.09]
        ],
        z = -.3525,
        scale = 1.4,
        **kwargs,
    ):
        self.name = name
        self.quadrants = quadrants
        self.z = z
        self.scale = scale
        super().__init__(**kwargs)
    
    def generate_obj_config(self):
        quadrant_i = np.random.choice([0, 1, 2, 3])
        quadrant = self.quadrants[quadrant_i]
        push_obj_pos = np.array([
            quadrant[0], 
            quadrant[1], 
            self.z,
        ])

        config = {
            'object_name': self.name,
            'object_position': push_obj_pos,
            'object_quat': deg_to_quat([0, 0, 0]),
            'scale': self.scale,
        }
        return config

class SawyerPickPlaceObjectUtil(SawyerObjectUtil):
    def __init__(
        self,
        name = 'lego',
        z = -0.1,
        scale = 2.0,
        **kwargs,
    ):
        self.name = name
        self.z = z
        self.scale = scale
        super().__init__(**kwargs)
    
    def generate_obj_config(self):
        pos = np.array([
            np.random.uniform(self.gripper_pos_low[0], self.gripper_pos_high[0]),
            np.random.uniform(self.gripper_pos_low[1], self.gripper_pos_high[1]),
            self.z
        ])
        yaw = np.random.uniform(0, 360)
        quat = deg_to_quat([0, 0, yaw])

        config = {
            'object_name': self.name,
            'object_position': pos,
            'object_quat': quat,
            'scale': self.scale,
        }
        return config