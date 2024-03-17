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


import torch
import numpy as np
from utils.generate_data import *
import matplotlib.pyplot as plt
from algorithms.consensus_alg import *


Nsim = 1000  # number of runs for the histogram experiment

# Tensors to save error and communication rate for each run and for each NN model
# case1
error1 = torch.zeros(Nsim)
comm1 = torch.zeros(Nsim)
# case2
error2 = torch.zeros(Nsim)
comm2 = torch.zeros(Nsim)
# case3
error3 = torch.zeros(Nsim)
comm3 = torch.zeros(Nsim)

# Generate groundtruth data from system
h = 1e-3
T = 10
N = 3
Adj = load_graph(N)

# Trigger parameters
eps = 0.001
sigma = 0.1

# Consensus gain
kappa = 5

# Load NN for the NN-ETM
model1 = torch.load('./checkpoints/L0001/m_60.pth')
model1.eval()

model2 = torch.load('./checkpoints/L01/m_230.pth')
model2.eval()

model3 = torch.load('./checkpoints/L1/m_70.pth')
model3.eval()


for i in range(0, Nsim):
    print('Simulation ', i)
    # generate groundtruth
    batch_size = 1  # generate sequences one at a time, run all models on it
    times_batch, u_batch, u_avg_batch = generate_signals(N, h, T, batch_size)
    times = times_batch[0]
    u = u_batch[0]
    u_avg = u_avg_batch[0]

    full_comm = full_comm_err(batch_size, times_batch, u_batch, u_avg_batch, Adj, kappa)  # MSE at full communication

    # run case 1
    zn, evs, deltas = lin_consensus_test(times, u, Adj, kappa, sigma, eps, model1)

    err = 0
    for node in range(0, N):
        for time in range(0, times.shape[0]):
            err = err + (torch.linalg.norm(u_avg[time] - zn[time, node])
                         * torch.linalg.norm(u_avg[time] - zn[time, node]))
    mserr = err / (N * times.shape[0])
    comm = torch.sum(torch.sum(evs)) / (N * times.shape[0])

    error1[i] = (mserr - full_comm[0])/full_comm[0]
    comm1[i] = comm

    # run case 2
    zn, evs, deltas = lin_consensus_test(times, u, Adj, kappa, sigma, eps, model2)

    err = 0
    for node in range(0, N):
        for time in range(0, times.shape[0]):
            err = err + (torch.linalg.norm(u_avg[time] - zn[time, node])
                         * torch.linalg.norm(u_avg[time] - zn[time, node]))
    mserr = err / (N * times.shape[0])
    comm = torch.sum(torch.sum(evs)) / (N * times.shape[0])

    error2[i] = (mserr - full_comm[0])/full_comm[0]
    comm2[i] = comm


    # run case 3
    zn, evs, deltas = lin_consensus_test(times, u, Adj, kappa, sigma, eps, model3)

    err = 0
    for node in range(0, N):
        for time in range(0, times.shape[0]):
            err = err + (torch.linalg.norm(u_avg[time] - zn[time, node])
                         * torch.linalg.norm(u_avg[time] - zn[time, node]))
    mserr = err / (N * times.shape[0])
    comm = torch.sum(torch.sum(evs)) / (N * times.shape[0])

    error3[i] = (mserr - full_comm[0])/full_comm[0]
    comm3[i] = comm

# Save results to file for plotting (with plot_histograms.py)
data = torch.cat((error1.unsqueeze(dim=0).T, error2.unsqueeze(dim=0).T, error3.unsqueeze(dim=0).T,
                  comm1.unsqueeze(dim=0).T, comm2.unsqueeze(dim=0).T, comm3.unsqueeze(dim=0).T), dim=1)
with open('./figs/histogram_N5_1000sims.txt', 'w') as f:
    for line in data:
        string = ''
        for i in range(6):
            string = string + str(line[i].item()) + ' '
        string = string + '\n'
        f.write(string)
