import os
import argparse
from typing import Tuple

import gym
import numpy as np
import torch

from gym_uav.env.gym_uav import GymUav

from ddpg_pytorch.ddpg import DDPG
from ddpg_pytorch.utils.noise import OrnsteinUhlenbeckActionNoise
from ddpg_pytorch.utils.replay_memory import ReplayMemory, Transition
from ddpg_pytorch.wrappers.normalized_actions import NormalizedActions


# noinspection DuplicatedCode
def train_one(param: Tuple[float, float, float, float], time_steps: int, render: bool, device: torch.device):
    if 'set_float32_matmul_precision' in dir(torch):
        torch.set_float32_matmul_precision('high')
    checkpoints_dir = f'{os.path.dirname(os.path.abspath(__file__))}/saved_models'
    env = GymUav(
        size_len=30, obs_percentage=0.6,
        obs_interval=6, obs_radius=2,
        minimum_dist_to_destination=1,
        sensor_max_dist=12, reward_params=param
    )
    env = NormalizedActions(env)

    hidden_size = (400, 300)
    agent = DDPG(0.99,
                 0.001,
                 hidden_size,
                 env.observation_space.shape[0],
                 env.action_space,
                 device=device,
                 checkpoint_dir=checkpoints_dir
                 )
    nb_actions = env.action_space.shape[-1]
    ou_noise = OrnsteinUhlenbeckActionNoise(
        mu=np.zeros(nb_actions),
        sigma=0.2 * np.ones(nb_actions)
    )
    memory = ReplayMemory(int(1e5))
    timestep = 0
    while timestep <= time_steps:
        print('timestep:', timestep)
        ou_noise.reset()
        epoch_return = 0
        state = torch.Tensor([env.reset()]).to(device)
        while True:
            if render:
                env.render()
            
            action = agent.calc_action(state, ou_noise)
            next_state, reward, done, _ = env.step(action.cpu().numpy()[0])
            timestep += 1
            epoch_return += reward

            mask = torch.Tensor([done]).to(device)
            reward = torch.Tensor([reward]).to(device)
            next_state = torch.Tensor([next_state]).to(device)

            memory.push(state, action, mask, next_state, reward)

            state = next_state

            epoch_value_loss = 0
            epoch_policy_loss = 0

            if len(memory) > 64:
                transitions = memory.sample(64)
                batch = Transition(*zip(*transitions))
                value_loss, policy_loss = agent.update_params(batch)

                epoch_value_loss += value_loss
                epoch_policy_loss += policy_loss
            if done:
                break
    
    env.close()
    
    print(f'Finished training for {param} with {timestep} timesteps')
    eid = 0
    postfix_base = f'_{param[0]:.2f}_{param[1]:.2f}_{param[2]:.2f}_{param[3]:.2f}'
    postfix = f'{postfix_base}'
    while not agent.save_checkpoint(timestep, memory, postfix, override=False):
        postfix = f'{postfix_base}_{eid}'
        eid += 1
    
    return agent


def test_agent(agent: DDPG, param: Tuple[float, float, float, float], num_episodes: int, render: bool, device: torch.device):
    env = GymUav(
        size_len=30, obs_percentage=0.3,
        obs_interval=6, obs_radius=2,
        minimum_dist_to_destination=1,
        sensor_max_dist=12, reward_params=param
    )
    env = NormalizedActions(env)
    nb_actions = env.action_space.shape[-1]
    ou_noise = OrnsteinUhlenbeckActionNoise(
        mu=np.zeros(nb_actions),
        sigma=0.2 * np.ones(nb_actions)
    )
    
    successes, collides, timeout = 0, 0, 0
    for _ in range(num_episodes):
        ou_noise.reset()
        state = torch.Tensor([env.reset()]).to(device)
        while True:
            if render:
                env.render()
            
            action = agent.calc_action(state, ou_noise)
            next_state, reward, done, info = env.step(action.cpu().numpy()[0])

            if done:
                if info.done.value == 1:  # Check if the episode was successful
                    successes += 1
                elif info.done.value == 2 or info.done.value == 3:  # Check if the episode ended due to a collision
                    collides += 1
                elif info.done.value == 4: # Check if the episode ended due to a timeout
                    timeout += 1
                break
            
            state = torch.Tensor([next_state]).to(device)
    
    print(f'Success rate for parameter {param}: {successes / num_episodes}')
    print(f'Collision rate for parameter {param}: {collides / num_episodes}')
    print(f'Timeout rate for parameter {param}: {timeout / num_episodes}')


def main():
    arg = argparse.ArgumentParser()
    arg.add_argument('--gpu', type=int, default=0)
    arg.add_argument('--param', type=float, nargs=5, default=(1, 1, 1, 1, -0.5))
    arg.add_argument('--render', type=int, default=0)
    arg.add_argument('--train_time_steps', type=int, default=200000)
    arg.add_argument('--test_episodes', type=int, default=1000)
    args = arg.parse_args()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print('Start training...')
    agent = train_one(args.param, args.train_time_steps, args.render, device)
    print('Finished training...\n')
    print('Start testing...')
    test_agent(agent, args.param, args.test_episodes, args.render, device)
    print('Finished testing...\n')


if __name__ == '__main__':
    main()
