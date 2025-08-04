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


# Training loop for the pre-training stage
# Force NN to learn a fixed output for eta(t), the learned weights will be used to initialize the NN in training.py

import torch
from model.model import Simple
from utils.generate_data import *
from algorithms.consensus_lin import lin_consensus_train


def loss_fn(target_eta, etas):
    len_t, n = etas.shape
    output = torch.norm(torch.ones(len_t, n) * target_eta - etas, 'fro')  # force network to learn a fixed eta(t)
    return output


# Checkpoint folder
path_to_checkpoint = './pretrain/'

# Load NN model
model = Simple()
model.train()

# Optimizer and training configuration
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
epochs = 50
batch_size = 1

# Generate sequences of reference signals
h = 1e-3
T = 5
N = 3
Adj = load_graph(N)
times_batch, u_batch, u_avg_batch = generate_signals(N, h, T, batch_size)

# Trigger design parameters
eps = 0.001
sigma = 1

# Consensus gain
kappa = 50

target_eta = 0.5  # target output value for eta(t)

# Initialization for the auxiliary variables. Must fulfill sum(p0)=0
p0 = 10 * torch.randn(N)
p0[0] = -torch.sum(p0[1:])
z0 = 10 * torch.randn(N)


print('Start training')
with open("./pretrain/losses_pretrain.txt", "a") as file1:
    for epoch in range(0, epochs + 1):

        optimizer.zero_grad()

        loss = torch.tensor(0.0)
        for i in range(0, batch_size):
            # Select one simulation from the batch
            times = times_batch[i]
            u = u_batch[i]
            u_avg = u_avg_batch[i]

            # Run estimation
            z_estim, evs, etas = lin_consensus_train(times, u, Adj, kappa, sigma, eps, model, z0, p0)

            # Add loss
            loss = loss + loss_fn(target_eta, etas)

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
