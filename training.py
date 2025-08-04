"""
Copyright (C) 2024 Irene Perez-Salesa <i.perez at unizar dot es> (University of Zaragoza)
For more information see <https://github.com/ireneperezsalesa/NN-ETM/blob/master/README.md>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""


# Training loop for the NN-ETM

import torch
from torch import nn
from model.model import Simple
from utils.generate_data import load_graph
from algorithms.consensus_lin import lin_consensus_train, full_comm_err


def loss_fn(gt_states, estim_states, events, full_comm_mse, L):  # trade-off between error and communication

    len_t, nodes = estim_states.shape

    comm = sum(sum(events)) / (nodes * len_t)

    err = 0
    for node in range(0, nodes):
        for time in range(0, len_t):
            err = err + (torch.linalg.norm(gt_states[time] - estim_states[time, node])
                         * torch.linalg.norm(gt_states[time] - estim_states[time, node]))
    mserr = err / (N * len_t)
    relu = nn.ReLU()
    relative_mserr = relu(mserr - full_comm_mse) / full_comm_mse # use relu function to only penalize error if it is higher than in the full communication case

    output = relative_mserr + L * comm  # L assigns relative weights to both terms

    return output



print('Running training with L = 0.001')

# Checkpoint folder
path_to_checkpoint = './checkpoints/'

# Load NN model
model = torch.load('./pretrain/example_pretrain.pth')  # load pretrained model as initialization for the weights
model.train()

# Optimizer and training configuration
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
epochs = 1000
L = 0.001

# Load pre-generated batch of training sequences
h = 1e-3
T = 10
N = 3
Adj = load_graph(N)  # adjacency matrix for the graph
seq_batch = torch.load('data/random_5_seqs.pt')
batch_size = seq_batch['batch_size']
times_batch = seq_batch['t']
u_batch = seq_batch['u']
u_avg_batch = seq_batch['u_avg']
# or generate them as: times_batch, u_batch, u_avg_batch = generate_signals(N, h, T, batch_size)

# Initial conditions for auxiliary variables. Must fulfill sum(p0)=0. We use the same initialization for consensus
# with NN-ETM and the full communication case in the cost function
p0 = torch.randn(N)
p0[0] = -torch.sum(p0[1:])
z0 = torch.randn(N)  # choose fixed values instead of random if training several models for comparison

# Trigger parameters
eps = 0.001
sigma = 1

# Consensus gain
kappa = 50

# Compute MSE of the consensus protocol at full communication over the training sequences to use in cost function
full_comm = full_comm_err(batch_size, times_batch, u_batch, u_avg_batch, Adj, kappa, z0, p0)

print('Start training')
for epoch in range(0, epochs + 1):

    optimizer.zero_grad()

    loss = torch.tensor(0.0)
    for seq in range(0, batch_size):
        # Select one simulation from the batch
        times, u, u_avg = times_batch[seq], u_batch[seq], u_avg_batch[seq]

        # Run estimation
        z_estim, evs, etas = lin_consensus_train(times, u, Adj, kappa, sigma, eps, model, z0, p0)

        # Add loss
        loss = loss + loss_fn(u_avg, z_estim, evs, full_comm[seq], L)

    print('Epoch: ', epoch, ', loss: ', loss)

    # Backprop
    loss.backward()
    optimizer.step()

    # Save checkpoints
    with torch.no_grad():
        if epoch % 10 == 0:
            print('Saving')
            cp_file_name = path_to_checkpoint + 'cp_' + str(epoch) + '.tar'
            model_name = path_to_checkpoint + 'm_' + str(epoch) + '.pth'
            torch.save({
                'last_epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
            }, cp_file_name)
            torch.save(model, model_name)
