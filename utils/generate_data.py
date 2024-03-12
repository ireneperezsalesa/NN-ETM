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

def load_graph(N):  # TODO automate graph generation
    # Graph
    if N == 2:
        Adj = torch.tensor(np.matrix('0 1; 1 0')).float()
        D = torch.tensor(np.matrix('1 -1; -1 1')).float()
        Inc = torch.tensor(np.matrix('1; -1')).float()

    if N == 5:  # graph with 5 nodes and connectivity = 0.5
        Adj = torch.tensor(np.matrix('0 1 0 1 1; 1 0 0 1 1; 0 0 0 1 0; 1 1 1 0 0; 1 1 0 0 0'))

    return Adj


def generate_signals(N, h, T, batch_size):
    times_batch = []
    u_batch = []
    u_avg_batch = []
    for i in range(0, batch_size):
        times = torch.arange(0, T+h, h)
        times_batch.append(times)

        u_nodes = torch.zeros(times.shape[0], N)
        u_avg = torch.zeros(times.shape[0])
        for n in range(0, N):
            a = torch.randint(3, 6, (1, 1)).item()
            b = torch.rand(1).item()
            u_nodes[:, n] = a * torch.sin(b*times)
            u_avg = u_avg + (1/N) * u_nodes[:, n]
        u_batch.append(u_nodes)
        u_avg_batch.append(u_avg)

    return times_batch, u_batch, u_avg_batch