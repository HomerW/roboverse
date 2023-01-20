import numpy as np
import time
import os
import os.path as osp
import roboverse
from roboverse.policies import policies
import argparse
from tqdm import tqdm
import h5py

from roboverse.utils import get_timestamp

EPSILON = 0.1

# TODO(avi): Clean this up
NFS_PATH = "/nfs/kun1/users/avi/imitation_datasets/"


def add_transition(
    traj, observation, action, reward, info, agent_info, done, next_observation, img_dim
):
    observation["image"] = np.reshape(
        np.uint8(observation["image"] * 255.0), (img_dim, img_dim, 3)
    )
    traj["observations"].append(observation)
    next_observation["image"] = np.reshape(
        np.uint8(next_observation["image"] * 255.0), (img_dim, img_dim, 3)
    )
    traj["next_observations"].append(next_observation)
    traj["actions"].append(action)
    traj["rewards"].append(reward)
    traj["terminals"].append(done)
    traj["agent_infos"].append(agent_info)
    traj["env_infos"].append(info)
    return traj


def collect_one_traj(env, policy, num_timesteps, noise, accept_trajectory_key, reset_kwargs):
    num_steps = -1
    rewards = []
    success = False
    img_dim = env.observation_img_dim
    env.reset(**reset_kwargs)
    policy.reset()
    time.sleep(1)
    traj = dict(
        observations=[],
        actions=[],
        rewards=[],
        next_observations=[],
        terminals=[],
        agent_infos=[],
        env_infos=[],
    )
    for j in range(num_timesteps):

        action, agent_info = policy.get_action()

        # In case we need to pad actions by 1 for easier realNVP modelling
        env_action_dim = env.action_space.shape[0]
        if env_action_dim - action.shape[0] == 1:
            action = np.append(action, 0)
        action += np.random.normal(scale=noise, size=(env_action_dim,))
        action = np.clip(action, -1 + EPSILON, 1 - EPSILON)
        observation = env.get_observation()
        next_observation, reward, done, info = env.step(action)
        add_transition(
            traj,
            observation,
            action,
            reward,
            info,
            agent_info,
            done,
            next_observation,
            img_dim,
        )

        if info[accept_trajectory_key] and num_steps < 0:
            num_steps = j

        rewards.append(reward)
        # if done or agent_info["done"]:
        #     break

    if info[accept_trajectory_key]:
        success = True

    return traj, success, num_steps


def main(args):

    timestamp = get_timestamp()
    if osp.exists(NFS_PATH):
        data_save_path = osp.join(NFS_PATH, args.save_directory)
    else:
        data_save_path = osp.join(__file__, "../..", "data", args.save_directory)
    data_save_path = osp.abspath(data_save_path)
    if not osp.exists(data_save_path):
        os.makedirs(data_save_path)

    env = roboverse.make(args.env_name, gui=args.gui, transpose_image=False)

    data = []
    assert (
        args.policy_name in policies.keys()
    ), f"The policy name must be one of: {policies.keys()}"
    assert (
        args.accept_trajectory_key in env.get_info().keys()
    ), f"""The accept trajectory key must be one of: {env.get_info().keys()}"""
    policy_class = policies[args.policy_name]
    policy = policy_class(env)
    num_success = 0
    num_saved = 0
    num_attempts = 0
    accept_trajectory_key = args.accept_trajectory_key

    reset_infos = h5py.File(args.data_path, 'r')['infos']
    original_object_positions = reset_infos['initial_positions'][args.index]
    target_position = reset_infos['target_position'][args.index]
    object_names = [
        reset_infos['push_object_name'][args.index].decode('UTF-8'),
        reset_infos['container_object_name'][args.index].decode('UTF-8'),
        reset_infos['utensil_object_name'][args.index].decode('UTF-8'),
        reset_infos['pickplace_object_name'][args.index].decode('UTF-8'),
    ]
    target_object = reset_infos['target_object'][args.index].decode('UTF-8')

    ## Add offsets to object
    # original_object_positions[0][0] += 0.15
    # original_object_positions[1][0] += 0.05
    # original_object_positions[3][0] += 0.05

    reset_kwargs = {
        'original_object_positions': original_object_positions,
        'target_position': target_position,
        'object_names': object_names,
        'target_object': target_object,
    }

    progress_bar = tqdm(total=args.num_trajectories)

    while num_saved < args.num_trajectories:
        num_attempts += 1
        traj, success, num_steps = collect_one_traj(
            env, policy, args.num_timesteps, args.noise, accept_trajectory_key, reset_kwargs
        )

        if success:
            if args.gui:
                print("num_timesteps: ", num_steps)
            data.append(traj)
            num_success += 1
            num_saved += 1
            progress_bar.update(1)
        elif args.save_all:
            data.append(traj)
            num_saved += 1
            progress_bar.update(1)

        if args.gui:
            print("success rate: {}".format(num_success / (num_attempts)))

    progress_bar.close()
    print("success rate: {}".format(num_success / (num_attempts)))

    path = osp.join(
        data_save_path, "scripted_{}_{}.hdf5".format(args.env_name, timestamp)
    )
    print(path)
    with h5py.File(path, "w") as f:
        f["observations/images0"] = np.array([o["image"] for t in data for o in t["observations"]])
        f["next_observations/images0"] = np.array([o["image"] for t in data for o in t["next_observations"]])
        f["observations/state"] = np.array([o["state"] for t in data for o in t["observations"]])
        f["next_observations/state"] = np.array([o["state"] for t in data for o in t["observations"]])
        f["actions"] = np.array([a for t in data for a in t["actions"]], dtype=np.float32)
        f["terminals"] = np.zeros(f["actions"].shape[0], dtype=np.bool_)
        f["truncates"] = np.zeros(f["actions"].shape[0], dtype=np.bool_)
        for key in data[0]['env_infos'][0]:
            f[f"infos/{key}"] = [i[key] for t in data for i in t['env_infos']]
        f["steps_remaining"] = np.zeros(f["actions"].shape[0], dtype=np.uint32)
        end = 0
        for traj in data:
            start = end
            end += len(traj["actions"])
            f["truncates"][end - 1] = True
            f["steps_remaining"][start:end] = np.arange(end - start)[::-1]


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env-name", type=str, required=True)
    parser.add_argument("-pl", "--policy-name", type=str, required=True)
    parser.add_argument("-a", "--accept-trajectory-key", type=str, required=True)
    parser.add_argument("-n", "--num-trajectories", type=int, required=True)
    parser.add_argument("-t", "--num-timesteps", type=int, required=True)
    parser.add_argument("--save-all", action="store_true", default=False)
    parser.add_argument("--gui", action="store_true", default=False)
    parser.add_argument("-o", "--target-object", type=str)
    parser.add_argument("-d", "--save-directory", type=str, default="")
    parser.add_argument("--noise", type=float, default=0.1)

    # For env initialization
    parser.add_argument("-dp", "--data_path", type=str, required=True)
    parser.add_argument("-i", "--index", type=int, required=True)
    args = parser.parse_args()

    main(args)
