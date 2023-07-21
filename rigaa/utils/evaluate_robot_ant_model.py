import numpy as np
import pickle
import gzip
import h5py
import argparse
from d4rl.locomotion import maze_env, ant, swimmer
from d4rl.locomotion.wrappers import NormalizedBoxEnv
import torch
from PIL import Image
import os




RESET = R = 'r'  # Reset position.
GOAL = G = 'g'

def reset_data():
    return {'observations': [],
            'actions': [],
            'terminals': [],
            'timeouts': [],
            'rewards': [],
            'infos/goal': [],
            'infos/qpos': [],
            'infos/qvel': [],
            }

def append_data(data, s, a, r, tgt, done, timeout, env_data):
    data['observations'].append(s)
    data['actions'].append(a)
    data['rewards'].append(r)
    data['terminals'].append(done)
    data['timeouts'].append(timeout)
    data['infos/goal'].append(tgt)
    data['infos/qpos'].append(env_data.qpos.ravel().copy())
    data['infos/qvel'].append(env_data.qvel.ravel().copy())

def npify(data):
    for k in data:
        if k in ['terminals', 'timeouts']:
            dtype = np.bool_
        else:
            dtype = np.float32

        data[k] = np.array(data[k], dtype=dtype)

def load_policy(policy_file):
    data = torch.load(policy_file)
    policy = data['exploration/policy'].to('cpu')
    env = data['evaluation/env']
    #print("Policy loaded")
    return policy, env

def save_video(save_dir, file_name, frames, episode_id=0, tr=0):
    filename = os.path.join(save_dir, file_name+ 'episode{}_goal_point{}'.format(episode_id, tr))
    if not os.path.exists(filename):
        os.makedirs(filename)
    num_frames = frames.shape[0]
    for i in range(num_frames):
        img = Image.fromarray(np.flipud(frames[i]), 'RGB')
        img.save(os.path.join(filename, 'frame_{}.png'.format(i)))

def evaluate_robot_ant_model(maze, waypoints, video=True):
    noisy = False
    env = "Ant"
    env_name = env
    max_episode_steps = 1000
    multi_start = False
    policy_file = "d4rl/ant_hierarch_pol.pkl"
    maze[1][1] = "r"
    


    
    if env == 'Ant':
        env = NormalizedBoxEnv(ant.AntMazeEnv(maze_map=maze, maze_size_scaling=4.0, non_zero_reset=multi_start))
    elif env == 'Swimmer':
        env = NormalizedBoxEnv(swimmer.SwimmerMazeEnv(mmaze_map=maze, maze_size_scaling=4.0, non_zero_reset=multi_start))
    else:
        raise NotImplementedError

    #target_list = [[5*4,3*4], [11*4, 7*4], [19*4, 12*4]]
    target_list_ = waypoints#[[1, 1], [6, 5], [15, 5], [16, 6], [16, 9], [18, 11], [22, 11], [24, 13], [19, 15], [21, 17], [25, 20], [32, 20], [33, 21], [33, 26], [36, 29], [31, 32], [37, 37]]
    target_list = [[4*i[0], 4*i[1]] for i in target_list_]
    targets_reached = 0
    
    env.set_target(target_location=target_list[0])
    s = env.reset()
    act = env.action_space.sample()
    done = False

    # Load the policy
    policy, train_env = load_policy(policy_file)

    # Define goal reaching policy fn
    def _goal_reaching_policy_fn(obs, goal):
        goal_x, goal_y = goal
        obs_new = obs[2:-2]
        goal_tuple = np.array([goal_x, goal_y])

        # normalize the norm of the relative goals to in-distribution values
        goal_tuple = goal_tuple / np.linalg.norm(goal_tuple) * 10.0

        new_obs = np.concatenate([obs_new, goal_tuple], -1)
        return policy.get_action(new_obs)[0], (goal_tuple[0] + obs[0], goal_tuple[1] + obs[1])      

    data = reset_data()

    # create waypoint generating policy integrated with high level controller
    data_collection_policy = env.create_navigation_policy(
        _goal_reaching_policy_fn,
    )

    if video:
        frames = []
    
    ts = 0
    num_episodes = 0
    episode = 1
    #for _ in range(args.num_samples):
    #for _ in range(len(target_list)):
    last_target_reached = 0
    tries_num = 3
    current_try = 0
    target_reached_list = []
    old_xy = np.array([0,0])
    final_rewards = []
    current_run_rewards = []
    while targets_reached < len(target_list) and current_try < tries_num:
        act, waypoint_goal = data_collection_policy(s)

        if noisy:
            act = act + np.random.randn(*act.shape)*0.2
            act = np.clip(act, -1.0, 1.0)

        ns, r, done, info = env.step(act)
        current_run_rewards.append(r)
        #print("Reward", r)
        #print("Done", done)
        timeout = False
        if ts >= max_episode_steps:
            timeout = True
            #done = True
        
        append_data(data, s[:-2], act, r, env.target_goal, done, timeout, env.physics.data)

        if len(data['observations']) % 10000 == 0:
            print(len(data['observations']))

        ts += 1
        #dist_from_prev = np.linalg.norm(env.get_xy() - old_xy)
        #timeout = dist_from_prev < 0.005 and ts > 100
        done = np.linalg.norm(env.get_xy() - target_list[targets_reached]) <= 6
        #old_xy = env.get_xy().copy()

        if done or timeout:
            done = False
            #print("Number of time steps", ts)
            ts = 0
            #s = env.reset()
            
            if timeout==True:

                last_target_reached = targets_reached
                target_reached_list.append(last_target_reached/len(target_list))
                final_rewards.append(sum(current_run_rewards)/len(current_run_rewards))
                current_run_rewards = []
                
                s = env.reset()
                #print("Reached ", targets_reached, " targets so far")
                targets_reached = 0
                env.set_target_goal(goal_input=target_list[targets_reached])
                #print("Doing reset")
                current_try += 1
            else:
                #print("Reached ", targets_reached, " targets so far")
                targets_reached += 1
                last_target_reached = targets_reached
                if targets_reached < len(target_list):
                    env.set_target_goal(goal_input=target_list[targets_reached])
                else:
                    target_reached_list.append(last_target_reached/len(target_list))
                    final_rewards.append(sum(current_run_rewards)/len(current_run_rewards))
                    current_run_rewards = []
                    s = env.reset()
                    #print("Reached ", targets_reached, " targets so far")
                    targets_reached = 0
                    env.set_target_goal(goal_input=target_list[targets_reached])
                    #print("Doing reset")
                    current_try += 1

            if video:
                frames = np.array(frames)
                #save_video('./videos/', env_name + '_navigation', frames, num_episodes, last_target_reached)
            
            num_episodes += 1
            frames = []
        else:
            s = ns

        if video:
            curr_frame = env.physics.render(width=400, height=400, mode="window",depth=False)
            frames.append(curr_frame)

    
    #glfw.terminate()
    env.close()
    
    return sum(target_reached_list)/len(target_reached_list), sum(final_rewards)/len(final_rewards)
    
    


if __name__ == '__main__':
    evaluate_robot_ant_model()
