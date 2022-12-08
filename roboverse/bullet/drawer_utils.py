import pybullet as p
import roboverse.bullet as bullet
import roboverse.bullet.control as control
import numpy as np


def open_drawer(drawer, half_open=False):
    return slide_drawer(drawer, -1, half_slide=half_open)


def close_drawer(drawer):
    return slide_drawer(drawer, 1)


def get_drawer_base_joint(drawer):
    joint_names = [control.get_joint_info(drawer, j, 'jointName')
                   for j in range(p.getNumJoints(drawer))]
    drawer_frame_joint_idx = joint_names.index('base_frame_joint')
    return drawer_frame_joint_idx


def get_drawer_handle_link(drawer):
    link_names = [bullet.get_joint_info(drawer, j, 'linkName')
                  for j in range(bullet.p.getNumJoints(drawer))]
    handle_link_idx = link_names.index('handle_r')
    return handle_link_idx

def get_drawer_frame_link(drawer):
    link_names = [bullet.get_joint_info(drawer, j, 'linkName')
                  for j in range(bullet.p.getNumJoints(drawer))]
    frame_link_idx = link_names.index('frame')
    return frame_link_idx

def get_drawer_pos(drawer):
    drawer_pos, _ = bullet.get_link_state(
        drawer, get_drawer_base_joint(drawer))
    return np.array(drawer_pos)

def get_drawer_handle_pos(drawer):
    handle_pos, _ = bullet.get_link_state(
        drawer, get_drawer_handle_link(drawer))
    return np.array(handle_pos)

def get_drawer_frame_pos(drawer):
    frame_pos, _ = bullet.get_link_state(
        drawer, get_drawer_frame_link(drawer))
    return np.array(frame_pos)

def get_drawer_opened_percentage(
        left_opening, min_x_pos, max_x_pos, drawer_x_pos):
    if left_opening:
        return (drawer_x_pos - min_x_pos) / (max_x_pos - min_x_pos)
    else:
        return (max_x_pos - drawer_x_pos) / (max_x_pos - min_x_pos)


def slide_drawer(drawer, direction, half_slide=False):
    assert direction in [-1, 1]
    # -1 = open; 1 = close
    drawer_frame_joint_idx = get_drawer_base_joint(drawer)

    num_ts = np.random.randint(low=57, high=61)

    if half_slide:
        num_ts = int(num_ts / 2)

    command = 0.5*direction

    # Wait a little before closing
    wait_ts = 30  # 0 if direction == -1 else 30
    control.step_simulation(wait_ts)

    p.setJointMotorControl2(
        drawer,
        drawer_frame_joint_idx,
        controlMode=p.VELOCITY_CONTROL,
        targetVelocity=command,
        force=5
    )

    drawer_pos = get_drawer_pos(drawer)

    control.step_simulation(num_ts)

    p.setJointMotorControl2(
        drawer,
        drawer_frame_joint_idx,
        controlMode=p.VELOCITY_CONTROL,
        targetVelocity=0,
        force=5
    )

    control.step_simulation(num_ts)
    return drawer_pos
