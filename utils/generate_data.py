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


import torch
import numpy as np

def load_graph(N):  # TODO automate graph generation
    # Communication graph

    if N == 3: # tree graph
        Adj = torch.tensor(np.matrix('0 1 0; 1 0 1; 0 1 0'))

    if N == 10:
        Adj = torch.tensor(np.matrix('0 0 0 1 1 1 1 1 0 0; 0 0 1 0 0 1 1 0 0 0; 0 1 0 1 1 0 0 0 1 0; 1 0 1 0 0 1 0 0 0 0; 1 0 1 0 0 0 0 0 1 0; 1 1 0 1 0 0 0 1 0 0; 1 1 0 0 0 0 0 0 1 0; 1 0 0 0 0 1 0 0 0 1; 0 0 1 0 1 0 1 0 0 1; 0 0 0 0 0 0 0 1 1 0'))

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
            if n <= 5:
                a = 10 * (n + 1) * torch.rand(1).item()
                b = (n + 1) * torch.rand(1).item()
                c = torch.rand(1).item() * torch.pi
                u_nodes[:, n] = a * torch.sin(b * times + c)
                u_avg = u_avg + (1 / N) * u_nodes[:, n]
            else:
                a = 10 * (n + 1) / 2 * torch.rand(1).item()
                b = (n + 1) / 2 * torch.rand(1).item()
                c = torch.rand(1).item() * torch.pi
                u_nodes[:, n] = a * torch.sin(b * times + c)
                u_avg = u_avg + (1 / N) * u_nodes[:, n]
        u_batch.append(u_nodes)
        u_avg_batch.append(u_avg)

    return times_batch, u_batch, u_avg_batch


def generate_signals_break(N, h, T, t_break, batch_size):
    # similar as generate_signals, but randomly changes reference signals at t = t_break
    times_batch = []
    u_batch = []
    u_avg_batch = []

    step_break = int(t_break / h)
    len_t = int(T/h)

    for i in range(0, batch_size):
        times = torch.arange(0, T+h, h)
        times_batch.append(times)

        u_nodes = torch.zeros(times.shape[0], N)
        u_avg = torch.zeros(times.shape[0])
        for n in range(0, N):
            if n <= 5:
                a = 10 * (n + 1)
                b = (n + 1)
                c = torch.pi
                u_nodes[0:step_break, n] = a * torch.rand(1).item() * torch.sin(b * torch.rand(1).item() * times[0:step_break] + c * torch.rand(1).item())
                u_nodes[step_break:, n] = a * torch.rand(1).item() * torch.sin(b * torch.rand(1).item() * times[step_break:] + c * torch.rand(1).item()) + 2
                u_avg = u_avg + (1 / N) * u_nodes[:, n]
            else:
                a = 10 * (n + 1) / 2
                b = (n + 1) / 2
                c = torch.pi
                u_nodes[0:step_break, n] = a * torch.rand(1).item() * torch.sin(b * torch.rand(1).item() * times[0:step_break] + c * torch.rand(1).item())
                u_nodes[step_break:, n] = a * torch.rand(1).item() * torch.sin(b * torch.rand(1).item() * times[step_break:] + c * torch.rand(1).item()) + 2
                u_avg = u_avg + (1 / N) * u_nodes[:, n]
        u_batch.append(u_nodes)
        u_avg_batch.append(u_avg)

    return times_batch, u_batch, u_avg_batch

