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


# Test consensus with trained NN-ETM over a batch of sequences of reference signals

import torch
from utils.generate_data import *
from algorithms.consensus_lin import *
from algorithms.consensus_george import *

# Load signals
h = 1e-3
T = 5
N = 3
Adj = load_graph(N)

seq_batch = torch.load('data/random_20_seqs.pt')
batch_size = seq_batch['batch_size']
times_batch = seq_batch['t']
u_batch = seq_batch['u']
u_avg_batch = seq_batch['u_avg']

# Checkpoint folder and NN-ETM configuration
path_to_model = './checkpoints/example_model.pth'  # path to trained NN model
model = torch.load(path_to_model)
model.eval()
eps = 1e-3
sigma = 1

# Consensus gain
kappa = 50

# Initialization for auxiliary consensus variables, fulfilling sum(p0)=0
p0 = 10 * torch.randn(N)
p0[0] = -torch.sum(p0[1:])
z0 = 10 * torch.randn(N)  # fix a given initialization when comparing different models for a fair comparison


mserr = torch.zeros(batch_size)
comm = torch.zeros(batch_size)
numevs = torch.zeros(batch_size)

for i in range(0, batch_size):
    # Select one simulation from the batch
    times = times_batch[i]
    u = u_batch[i]
    u_avg = u_avg_batch[i]

    # Run estimation
    zn, evs, deltas, ei, disag = lin_consensus_test(times, u, Adj, kappa, sigma, eps, model, z0, p0)

    # Compute communication rate, events and MSE
    comm[i] = sum(sum(evs))/(N*times.shape[0])
    gt_states = u_avg
    estim_states = zn
    err = 0
    for node in range(0, N):
        for time in range(0, times.shape[0]):
            err = err + (torch.linalg.norm(gt_states[time] - estim_states[time, node])
                         * torch.linalg.norm(gt_states[time] - estim_states[time, node]))
    mserr[i] = err/(N*times.shape[0])
    numevs[i] = sum(sum(evs))/N


avg_mse = sum(mserr)/batch_size
avg_comm = sum(comm)/batch_size
avg_evs = sum(numevs)/batch_size

print('Avg MSE: ', avg_mse)
print('Avg communication rate: ', avg_comm)
print('Avg events per node: ', avg_evs)


