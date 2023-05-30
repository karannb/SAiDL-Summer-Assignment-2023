import gym
import numpy as np
import torch
import pickle
import random
import matplotlib.pyplot as plt

from evaluate_ import evaluate_episode_rtg
from model_ import decisionRNN
from trainer_ import Trainer


def weighted_avg(x, gamma):
    weighted_avg = np.zeros_like(x)
    weighted_avg[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        weighted_avg[t] = x[t] + gamma * weighted_avg[t+1]
    return weighted_avg

class Expt:

    def __init__(self, ax1, ax2, cell_type='LSTM', mode='normal', context_length=50, batch_size=256, 
                 num_eval_episodes=100, lr=1e-3, dataset='medium', num_layers=2, activation_fn="Tanh",
                 hidden_dim=64, proj_dim=128, dropout=0.1, iterations=1000, epochs=10):
        
        self.ax1 = ax1
        self.ax2 = ax2
        self.cell_type = cell_type
        self.mode = mode
        self.context_length = context_length 
        self.batch_size = batch_size 
        self.num_eval_episodes = num_eval_episodes 
        self.lr = lr
        self.dataset = dataset
        self.num_layers = num_layers
        self.activation_fn = activation_fn
        self.hidden_dim = hidden_dim
        self.proj_dim = proj_dim
        self.p = dropout
        self.iterations = iterations
        self.epochs = epochs
        self.path = f"__pycache__/c{cell_type}-d{dataset}-m{mode}.pth"

    def experiment(self):

        cell_type = self.cell_type

        env = gym.make('Hopper-v3')
        max_ep_len = 1000
        env_targets = [3600, 1800]  # evaluation conditioning targets
        scale = 1000.  # normalization for rewards/returns

        state_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]

        # load dataset
        dataset_path = f'data/hopper-{self.dataset}-v2.pkl'
        with open(dataset_path, 'rb') as f:
            trajectories = pickle.load(f)

        # save all path information into separate lists
        mode = self.mode
        states, traj_lens, returns = [], [], []
        for path in trajectories:
            if mode == 'delayed':  # delayed: all rewards moved to end of trajectory
                path['rewards'][-1] = path['rewards'].sum()
                path['rewards'][:-1] = 0.
            states.append(path['observations'])
            traj_lens.append(len(path['observations']))
            returns.append(path['rewards'].sum())
        traj_lens, returns = np.array(traj_lens), np.array(returns)

        # used for input normalization
        states = np.concatenate(states, axis=0)
        state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

        num_timesteps = sum(traj_lens)

        print('=' * 50)
        print(f'Starting new experiment: {self.dataset}')
        print(f'{len(traj_lens)} trajectories, {num_timesteps} timesteps found')
        print(f'Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}')
        print(f'Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}')
        print('=' * 50)

        context_length = self.context_length
        batch_size = self.batch_size
        num_eval_episodes = self.num_eval_episodes

        sorted_inds = np.argsort(returns)  # lowest to highest
        num_trajectories = 1
        timesteps = traj_lens[sorted_inds[-1]]
        ind = len(trajectories) - 2
        while ind >= 0 and timesteps + traj_lens[sorted_inds[ind]] <= num_timesteps:
            timesteps += traj_lens[sorted_inds[ind]]
            num_trajectories += 1
            ind -= 1
        sorted_inds = sorted_inds[-num_trajectories:]

        # used to reweight sampling so we sample according to timesteps instead of trajectories
        p_sample = traj_lens[sorted_inds] / sum(traj_lens[sorted_inds])

        def get_batch(batch_size=256, max_len=context_length):

            batch_inds = np.random.choice(
                np.arange(num_trajectories),
                size=batch_size,
                replace=True,
                p=p_sample,  # reweights so we sample according to timesteps
            )

            s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
            for i in range(batch_size):
                traj = trajectories[int(sorted_inds[batch_inds[i]])]
                si = random.randint(0, traj['rewards'].shape[0] - 1)

                # get sequences from dataset
                s.append(traj['observations'][si:si + max_len].reshape(1, -1, state_dim))
                a.append(traj['actions'][si:si + max_len].reshape(1, -1, act_dim))
                r.append(traj['rewards'][si:si + max_len].reshape(1, -1, 1))
                if 'terminals' in traj:
                    d.append(traj['terminals'][si:si + max_len].reshape(1, -1))
                else:
                    d.append(traj['dones'][si:si + max_len].reshape(1, -1))
                timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
                timesteps[-1][timesteps[-1] >= max_ep_len] = max_ep_len-1  # padding cutoff
                rtg.append(weighted_avg(traj['rewards'][si:], gamma=1.)[:s[-1].shape[1]].reshape(1, -1, 1))
                #if rtg[-1].shape[1] <= s[-1].shape[1]:
                #    rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

                # padding and state + reward normalization
                tlen = s[-1].shape[1]
                s[-1] = np.concatenate([np.zeros((1, max_len - tlen, state_dim)), s[-1]], axis=1)
                s[-1] = (s[-1] - state_mean) / state_std
                a[-1] = np.concatenate([np.ones((1, max_len - tlen, act_dim)) * -10., a[-1]], axis=1)
                r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
                d[-1] = np.concatenate([np.ones((1, max_len - tlen)) * 2, d[-1]], axis=1)
                rtg[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1) / scale
                timesteps[-1] = np.concatenate([np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1)

            s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32)
            a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32)
            r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32)
            d = torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.long)
            rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32)
            timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long)

            return s, a, r, d, rtg, timesteps

        def eval_episodes(target_rew):
            def fn(model):
                returns, lengths = [], []
                for _ in range(num_eval_episodes):
                    with torch.no_grad():
                        ret, length = evaluate_episode_rtg(
                            env,
                            state_dim,
                            act_dim,
                            model,
                            max_ep_len=max_ep_len,
                            scale=scale,
                            target_return=target_rew/scale,
                            mode=mode,
                            state_mean=state_mean,
                            state_std=state_std,
                        )
                    returns.append(ret)
                    lengths.append(length)
                return {
                    f'target_{target_rew}_return_mean': np.mean(returns),
                    f'target_{target_rew}_return_std': np.std(returns),
                    f'target_{target_rew}_length_mean': np.mean(lengths),
                    f'target_{target_rew}_length_std': np.std(lengths),
                }, returns, lengths, num_eval_episodes
            return fn

        model = decisionRNN(
            cell_type=self.cell_type,
            context_length=context_length,
            max_ep_len=max_ep_len,
            state_dim=state_dim,
            act_dim=act_dim,
            hidden_dim=self.hidden_dim,
            proj_dim=self.proj_dim,
            num_layers=self.num_layers,
            activation_fn=self.activation_fn,
            dropout=self.p
        )

        #warmup_steps = variant['warmup_steps']
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.lr
        )
        #scheduler = torch.optim.lr_scheduler.LambdaLR(
        #    optimizer,
        #    lambda steps: min((steps+1)/warmup_steps, 1)
        #)

        trainer = Trainer(
            model=model,
            get_batch=get_batch,
            eval_fns=[eval_episodes(tar) for tar in env_targets],
            optimizer=optimizer,
            ax1 = self.ax1,
            ax2 = self.ax2,
            batch_size=batch_size
        )

        for iter in range(self.epochs):
            outputs = trainer.train_iteration(num_steps=self.iterations, epoch=iter+1, print_logs=True)
        
        torch.save(model.state_dict(), self.path)