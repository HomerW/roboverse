import pybullet_data
import pybullet as p
import os
import roboverse.bullet as bullet
import random

CUR_PATH = os.path.dirname(os.path.realpath(__file__))
ASSET_PATH = os.path.join(CUR_PATH, '../assets')
SHAPENET_ASSET_PATH = os.path.join(ASSET_PATH, 'bullet-objects/ShapeNetCore')

"""
NOTE: Use this file only for core objects, add others to bullet/object_utils.py
This file will likely be deprecated in the future.
"""

TABLE_COLORS = [(1, 0, 0, 1),(0, 0, 1, 1),(0, 1, 0, 1),(1, 1, 0, 1),(0, 1, 1, 1),(1, 0, 1, 1),(0.5, 0.5, 0.5, 1),(0.5, 0, 0, 1)]

# def load_color_table():
#     collisionid = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=[1, 1, 1])
#     visualid = p.createVisualShape(
#         shapeType=p.GEOM_BOX,
#         rgbaColor=random.choice(TABLE_COLORS))
#     body = p.createMultiBody(0.05, collisionid, visualid)
#     # p.resetBasePositionAndOrientation(body, [.75, -.2, -1], [0, 0, 0.707107, 0.707107])
#     # p.changeDynamics(body, -1, lateralFriction=1.0)
#     return body

# def load_back_wall():
#     p.setAdditionalSearchPath(pybullet_data.getDataPath())
#     table_id = p.loadURDF('table/table.urdf',
#                           basePosition=[.75, -.2, -1],
#                           baseOrientation=[0, 0, 0.707107, 0.707107],
#                           globalScaling=1.0)
#     return table_id

def table():
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    table_id = p.loadURDF('table/table.urdf',
                          basePosition=[.75, -.2, -1],
                          baseOrientation=[0, 0, 0.707107, 0.707107],
                          globalScaling=1.0)
    return table_id

def table_centered():
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    table_id = p.loadURDF('table/table.urdf',
                          basePosition=[.625, -.2, -1],
                          baseOrientation=[0, 0, 0.707107, 0.707107],
                          globalScaling=1.0)
    return table_id

def tray(base_position=(.60, 0.3, -.37), scale=0.5):
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    tray_id = p.loadURDF('tray/tray.urdf',
                         basePosition=base_position,
                         baseOrientation=[0, 0, 0.707107, 0.707107],
                         globalScaling=scale)
    return tray_id

def tray_no_divider(base_position=(.60, 0.25, -.37), scale=0.6):
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    BASE_ASSET_PATH = os.path.join(ASSET_PATH, 'bullet-objects')
    tray_id = p.loadURDF(os.path.join(BASE_ASSET_PATH, 'tray/tray_no_divider.urdf'),
                         basePosition=base_position,
                         baseOrientation=[0, 0, 0.707107, 0.707107],
                         globalScaling=scale)
    return tray_id

def tray_no_divider_scaled(base_position=(.60, 0.25, -.37), scale=0.6):
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    BASE_ASSET_PATH = os.path.join(ASSET_PATH, 'bullet-objects')
    tray_id = p.loadURDF(os.path.join(BASE_ASSET_PATH, 'tray/tray_no_divider_scaled.urdf'),
                         basePosition=base_position,
                         baseOrientation=[0, 0, 0.707107, 0.707107],
                         globalScaling=scale)
    return tray_id


def widow250():
    widow250_path = os.path.join(ASSET_PATH,
                                 'interbotix_descriptions/urdf/wx250s.urdf')
    widow250_id = p.loadURDF(widow250_path,
                             basePosition=[0.6, 0, -0.4],
                             baseOrientation=bullet.deg_to_quat([0., 0., 0])
                             )
    return widow250_id


# def back_wall():
#     wall_path = os.path.join(ASSET_PATH,
#                             'bullet-objects/wall/back_wall.urdf')
#     wall_id = p.loadURDF(wall_path,
#                         basePosition=(.6, 0.25, -.3),
#                         baseOrientation=(0, 0, 0.707107, 0.707107),
#                         globalScaling=0.25)
#     return wall_id