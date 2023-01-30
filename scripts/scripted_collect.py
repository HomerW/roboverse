import numpy as np
import time
import tensorflow as tf
import roboverse
from roboverse.policies import policies
import argparse
from tqdm import tqdm
from multiprocessing import Pool

from roboverse.utils import get_timestamp

EPSILON = 0.1


def get_data_save_directory(args):
    data_save_directory = args.save_directory

    data_save_directory += "/{}".format(args.env_name)

    if args.num_trajectories > 1000:
        data_save_directory += "_{}K".format(int(args.num_trajectories / 1000))
    else:
        data_save_directory += "_{}".format(args.num_trajectories)

    if args.save_all:
        data_save_directory += "_save_all"

    data_save_directory += "_noise_{}".format(args.noise)
    data_save_directory += "_{}".format(get_timestamp())

    return data_save_directory


def tensor_feature(value):
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(value).numpy()])
    )


def add_transition(
    traj, observation, action, reward, info, agent_info, done, next_observation, img_dim
):
    if "image" in observation:
        observation["image"] = np.reshape(
            np.uint8(observation["image"] * 255.0), (img_dim, img_dim, 3)
        )
        next_observation["image"] = np.reshape(
            np.uint8(next_observation["image"] * 255.0), (img_dim, img_dim, 3)
        )
    traj["observations"].append(observation)
    traj["next_observations"].append(next_observation)
    traj["actions"].append(action)
    traj["rewards"].append(reward)
    traj["terminals"].append(done)
    traj["agent_infos"].append(agent_info)
    traj["env_infos"].append(info)
    return traj


def collect_one_traj(env, policy, num_timesteps, noise, accept_trajectory_key):
    num_steps = -1
    rewards = []
    success = False
    img_dim = env.observation_img_dim
    env.reset()
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


def collect(args, save_path, num_traj):
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

    progress_bar = tqdm(total=num_traj)

    while num_saved < num_traj:
        num_attempts += 1
        traj, success, num_steps = collect_one_traj(
            env, policy, args.num_timesteps, args.noise, accept_trajectory_key
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

    with tf.io.TFRecordWriter(save_path) as writer:
        for traj in data:
            truncates = np.zeros(len(traj["actions"]), dtype=np.bool_)
            truncates[-1] = True
            steps_existing = np.arange(len(traj["actions"]), dtype=np.int32)
            steps_remaining = steps_existing[::-1]

            infos = {}
            for key in data[0]["env_infos"][0]:
                infos[f"infos/{key}"] = tensor_feature(
                    np.array([i[key] for i in traj["env_infos"]])
                )

            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "observations/images0": tensor_feature(
                            np.array(
                                [o["image"] for o in traj["observations"]],
                                dtype=np.uint8,
                            ),
                        ),
                        "observations/state": tensor_feature(
                            np.array(
                                [o["state"] for o in traj["observations"]],
                                dtype=np.float32,
                            )
                        ),
                        "next_observations/images0": tensor_feature(
                            np.array(
                                [o["image"] for o in traj["next_observations"]],
                                dtype=np.uint8,
                            ),
                        ),
                        "next_observations/state": tensor_feature(
                            np.array(
                                [o["state"] for o in traj["next_observations"]],
                                dtype=np.float32,
                            )
                        ),
                        "actions": tensor_feature(
                            np.array(traj["actions"], dtype=np.float32)
                        ),
                        "terminals": tensor_feature(
                            np.zeros(len(traj["actions"]), dtype=np.bool_)
                        ),
                        "truncates": tensor_feature(truncates),
                        "steps_existing": tensor_feature(steps_existing),
                        "steps_remaining": tensor_feature(steps_remaining),
                        **infos,
                    }
                )
            )
            writer.write(example.SerializeToString())


def main(args):
    save_directory = get_data_save_directory(args)
    if not tf.io.gfile.exists(save_directory):
        tf.io.gfile.makedirs(save_directory)

    worker_args = []
    for i in range(args.num_parallel_threads):
        worker_args.append((
            args,
            tf.io.gfile.join(save_directory, f"{i}.tfrecord"),
            args.num_trajectories // args.num_parallel_threads
            + int(i < args.num_trajectories % args.num_parallel_threads),
        ))

    with Pool(args.num_parallel_threads) as p:
        p.starmap(collect, worker_args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env-name", type=str, required=True)
    parser.add_argument("-pl", "--policy-name", type=str, required=True)
    parser.add_argument("-a", "--accept-trajectory-key", type=str, required=True)
    parser.add_argument("-n", "--num-trajectories", type=int, required=True)
    parser.add_argument("-t", "--num-timesteps", type=int, required=True)
    parser.add_argument("--save-all", action="store_true", default=False)
    parser.add_argument("--gui", action="store_true", default=False)
    parser.add_argument("-d", "--save-directory", type=str, default="")
    parser.add_argument("-p", "--num-parallel-threads", type=int, default=10)
    parser.add_argument("--noise", type=float, default=0.1)
    args = parser.parse_args()

    main(args)
