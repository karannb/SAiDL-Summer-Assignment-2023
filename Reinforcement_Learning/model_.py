import numpy as np
import torch
import torch.nn as nn

class decisionRNN(nn.Module):

    """
    This model uses GPT to model (Return_1, state_1, action_1, Return_2, state_2, ...)
    """

    def __init__(self, cell_type='LSTM', context_length=50, max_ep_len=1000, state_dim=11, act_dim=3, 
                 hidden_dim=64, proj_dim=128, num_layers=2, activation_fn="Tanh", dropout=0.1):

        super(decisionRNN, self).__init__()

        self.proj_dim = proj_dim
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.activation_fn = {"Tanh" : torch.nn.Tanh(), "ReLU" : torch.nn.ReLU()}[activation_fn]
        self.context_length = context_length

        self.embed_timestep = nn.Embedding(max_ep_len, proj_dim)
        self.embed_return = torch.nn.Linear(1, proj_dim)
        self.embed_state = torch.nn.Linear(self.state_dim, proj_dim)
        self.embed_action = torch.nn.Linear(self.act_dim, proj_dim)

        self.embed_ln = nn.LayerNorm(proj_dim)

        if cell_type=='RNN':
            self.RNN = nn.RNN(input_size=self.proj_dim, hidden_size=self.hidden_dim,
                              num_layers=num_layers, batch_first=True, dropout=dropout)
        elif cell_type=='GRU':
            self.RNN = nn.GRU(input_size=self.proj_dim, hidden_size=self.hidden_dim,
                              num_layers=num_layers, batch_first=True, dropout=dropout)
        else:
            self.RNN = nn.LSTM(input_size=self.proj_dim, hidden_size=self.hidden_dim,
                               num_layers=num_layers, batch_first=True, dropout=dropout)

        self.predict_action = nn.Sequential(
            nn.Linear(hidden_dim, self.act_dim),
            self.activation_fn
        )

    def forward(self, states, actions, returns_to_go, timesteps):

        batch_size, seq_length = states.shape[0], states.shape[1]

        assert seq_length == self.context_length

        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps)

        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings

        stacked_inputs = torch.stack(
            (returns_embeddings, state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 3*seq_length, self.proj_dim)

        stacked_inputs = self.embed_ln(stacked_inputs)

        # we feed in the input embeddings (not word indices as in NLP) to the model
        outputs = self.RNN(stacked_inputs)

        x = outputs[0]

        x = x.reshape(batch_size, seq_length, 3, self.hidden_dim).permute(0, 2, 1, 3)

        action_preds = self.predict_action(x[:,1]) 

        return action_preds

    def get_action(self, states, actions, returns_to_go, timesteps):

        states = states.reshape(1, -1, self.state_dim)
        actions = actions.reshape(1, -1, self.act_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        states = states[:,-self.context_length:]
        actions = actions[:,-self.context_length:]
        returns_to_go = returns_to_go[:,-self.context_length:]
        timesteps = timesteps[:,-self.context_length:]

        states = torch.cat(
            [torch.zeros((states.shape[0], self.context_length-states.shape[1], self.state_dim), device=states.device), states],
            dim=1).to(dtype=torch.float32)
        actions = torch.cat(
            [torch.zeros((actions.shape[0], self.context_length - actions.shape[1], self.act_dim),
                            device=actions.device), actions],
            dim=1).to(dtype=torch.float32)
        returns_to_go = torch.cat(
            [torch.zeros((returns_to_go.shape[0], self.context_length-returns_to_go.shape[1], 1), device=returns_to_go.device), returns_to_go],
            dim=1).to(dtype=torch.float32)
        timesteps = torch.cat(
            [torch.zeros((timesteps.shape[0], self.context_length-timesteps.shape[1]), device=timesteps.device), timesteps],
            dim=1
        ).to(dtype=torch.long)

        action_preds = self.forward(states, actions, returns_to_go, timesteps)

        return action_preds[0,-1] #because at test time, sampling sequentially