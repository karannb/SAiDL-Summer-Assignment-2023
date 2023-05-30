import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

import time

from evaluate_ import evaluate_episode_rtg

class Trainer:

    def __init__(self, model, get_batch, eval_fns, optimizer, ax1, ax2, batch_size=256): #, scheduler=None
        
        self.model = model
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.get_batch = get_batch
        self.loss_fn = torch.nn.MSELoss()
        #self.scheduler = scheduler
        self.eval_fns = eval_fns
        #self.diagnostics = dict()
        self.ax1 = ax1
        self.ax2 = ax2

        self.start_time = time.time()

    def train_iteration(self, num_steps, epoch=0, print_logs=True):

        train_losses = {}
        logs = dict()

        train_start = time.time()

        self.model.train()

        for i in range(num_steps):

            train_loss = self.train_step()
            train_losses[epoch*num_steps + i] = train_loss

            #if self.scheduler is not None:
            #    self.scheduler.step()

        logs['time/training'] = time.time() - train_start

        eval_start = time.time()

        self.model.eval()
        
        returns, ep_len = [], []

        for eval_fn in self.eval_fns:

            outputs = eval_fn(self.model)
            for k, v in outputs[0].items():
                logs[f'evaluation/{k}'] = v
            returns = outputs[1]
            ep_len = outputs[2]
            total_ep = outputs[3]
            
            self.ax1.plot(np.arange(epoch*total_ep, epoch*total_ep + total_ep), returns)
            self.ax2.plot(np.arange(epoch*total_ep, epoch*total_ep + total_ep), ep_len)

        logs['time/total'] = time.time() - self.start_time
        logs['time/evaluation'] = time.time() - eval_start
        logs['training/train_loss_mean'] = np.mean(list(train_losses.values()))
        logs['training/train_loss_std'] = np.std(list(train_losses.values()))

        if print_logs:
            print('=' * 80)
            print(f'Epoch {epoch}')
            for k, v in logs.items():
                print(f'{k}: {v}')

        return logs

    def train_step(self):

        states, actions, _, _, returns_to_go, timesteps = self.get_batch(self.batch_size)

        action_target = torch.clone(actions)

        action_preds = self.model.forward(states, actions, returns_to_go, timesteps)

        loss = self.loss_fn(action_preds, action_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().item()