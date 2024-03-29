"""
Copyright (C) 2024 Irene Perez-Salesa <i.perez at unizar dot es> (University of Zaragoza)
For more information see <https://github.com/ireneperezsalesa/NN-ETM/blob/main/README.md>

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
from model.model import *
from utils.generate_data import *
from algorithms.consensus_alg import *


def loss_fn(gt_states, estim_states, events, full_comm_mse, L):  # trade-off between error and communication

    len_t, nodes = estim_states.shape

    comm = sum(sum(events)) / (nodes * len_t)

    err = 0
    for node in range(0, nodes):
        for time in range(0, len_t):
            err = err + (torch.linalg.norm(gt_states[time] - estim_states[time, node])
                         * torch.linalg.norm(gt_states[time] - estim_states[time, node]))
    mserr = err / (N * len_t)

    relative_mserr = abs(mserr - full_comm_mse) / full_comm_mse

    output = relative_mserr + L * comm  # L assigns relative weights to both terms

    return output


# Checkpoint folder
path_to_checkpoint = './checkpoints/'  # the trained model will be stored here

# Load NN model
model = torch.load('./pretrain/m_500.pth')  # load pretrained model as initialization for the weights
model.train()

# Optimizer and training configuration
optimizer = torch.optim.Adam(model.parameters(), lr=5e-2)
epochs = 500  # training iterations
batch_size = 10  # number of pre-generated sequences for the reference signals
L = 1  # assigns weights to the error vs. communication trade-off in the cost function loss_fn

# Generate groundtruth training data from system
h = 1e-3  # simulation step
T = 10  # simulation time
N = 2  # number of agents
Adj = load_graph(N)  # adjacency matrix for the graph
times_batch, u_batch, u_avg_batch = generate_signals(N, h, T, batch_size)  # generate reference signals for consensus

# Trigger parameters
eps = 0.001
sigma = 0.1

# Consensus gain
kappa = 5

full_comm = full_comm_err(batch_size, times_batch, u_batch, u_avg_batch, Adj, kappa)  # compute MSE of consensus at full communication (used in loss_fn)

print('Start training')
with open("losses.txt", "a") as file1:
    for epoch in range(0, epochs + 1):

        optimizer.zero_grad()

        loss = torch.tensor(0.0)
        for i in range(0, batch_size):
            # Select one simulation from the batch
            times = times_batch[i]
            u = u_batch[i]
            u_avg = u_avg_batch[i]

            # Run estimation
            z_estim, evs, etas = lin_consensus_train(times, u, Adj, kappa, sigma, eps, model)

            # Add loss
            loss = loss + loss_fn(u_avg, z_estim, evs, full_comm[i], L)

        print('Epoch: ', epoch, ', loss: ', loss)
        string = 'Epoch: ' + str(epoch) + ', loss: ' + str(loss) + ' \n'
        file1.write(string)

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
