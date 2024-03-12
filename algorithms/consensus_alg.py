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


# Consensus algorithms (test and train versions)
import torch

def lin_consensus_test(t, u, Adj, kappa, sigma, eps, nn):
    h = t[1] - t[0]
    N = Adj.shape[0]

    # initialize variables
    p = torch.zeros(t.shape[0], N)
    z = torch.zeros(t.shape[0], N)
    zk = torch.zeros(t.shape[0], N)

    p0 = 2*torch.randn(N)
    p0[0] = -torch.sum(p0[1:])  # ensure orthogonality of initial conditions
    p[0, :] = p0

    # events
    event = torch.zeros(t.shape[0], N)
    event[0, :] = torch.ones(N)
    t_since_event = h * torch.ones(N)
    etas = torch.zeros(t.shape[0], N)

    # run event-triggered consensus algorithm
    for i in range(1, t.shape[0]):
        for n in range(0, N):
            sum_neighbors = torch.zeros(1)
            for j in range(0, N):
                sum_neighbors = sum_neighbors + Adj[n, j]*(z[i-1, n] - zk[i-1, j])
            p[i, n] = p[i-1, n] + h * (kappa * sum_neighbors)
            z[i, n] = u[i, n] - p[i, n]

            # dynamic variable
            nn_input = torch.cat((sum_neighbors, t_since_event[n].unsqueeze(dim=0)))
            eta = nn(nn_input)
            etas[i, n] = eta

            if torch.norm(z[i, n] - zk[i-1, n]) >= sigma * eta + eps:
                zk[i, n] = z[i, n]
                event[i, n] = 1
                t_since_event[n] = h
            else:
                zk[i, n] = zk[i-1, n]  # keep last value as default
                t_since_event[n] = t_since_event[n] + h

    return z, event, etas


def lin_consensus_train(t, u, Adj, kappa, sigma, eps, nn):
    h = t[1] - t[0]
    N = Adj.shape[0]

    # initialize variables
    p = torch.zeros(t.shape[0], N)
    z = torch.zeros(t.shape[0], N)
    zk = torch.zeros(t.shape[0], N)
    z_estim = torch.zeros(t.shape[0], N)
    etas = torch.zeros(t.shape[0], N)

    p0 = torch.randn(N)
    p0[0] = -torch.sum(p0[1:])  # ensure orthogonality of initial conditions
    p[0, :] = p0

    # events
    event = torch.zeros(t.shape[0], N)
    t_since_event = h * torch.ones(N)

    # run event-triggered consensus algorithm
    for i in range(1, t.shape[0]):
        for n in range(0, N):
            sum_neighbors = torch.zeros(1)
            for j in range(0, N):
                sum_neighbors = sum_neighbors + Adj[n, j] * (z[i-1, n] - zk[i-1, j])
            p[i, n] = p[i-1, n] + h * (kappa * sum_neighbors)
            z[i, n] = u[i, n] - p[i, n]

            # dynamic variable
            nn_input = torch.cat((sum_neighbors, t_since_event[n].unsqueeze(dim=0)))
            eta = nn(nn_input)
            etas[i, n] = eta

            # evolve if event / no event
            sig = torch.sigmoid(10 * (torch.norm(z[i, n] - zk[i-1, n]) - sigma * eta - eps))  # to fuzzify the ETM
            zk[i, n] = sig * z[i, n].clone() + (1 - sig) * zk[i-1, n].clone()
            event[i, n] = sig * 1 + (1 - sig) * 0
            next_time = t_since_event[n] + h
            t_since_event[n] = sig * h + (1 - sig) * next_time
            z_estim[i, n] = z[i, n]

    return z_estim, event, etas

def full_comm_err(batch_size, t_batch, u_batch, u_avg_batch, Adj, kappa):
    # computes MSE of the consensus algorithm run at full communication

    mserrs_batch = torch.zeros(batch_size)
    N = Adj.shape[0]

    for b in range(0, batch_size):
        t = t_batch[b]
        u = u_batch[b]
        u_avg = u_avg_batch[b]

        h = t[1] - t[0]

        # initialize variables
        p = torch.zeros(t.shape[0], N)
        z = torch.zeros(t.shape[0], N)

        p0 = torch.randn(N)
        p0[0] = -torch.sum(p0[1:])  # ensure orthogonality
        p[0, :] = p0


        # run consensus algorithm
        for i in range(1, t.shape[0]):
            for n in range(0, N):
                sum_neighbors = torch.zeros(1)
                for j in range(0, N):
                    sum_neighbors = sum_neighbors + Adj[n, j]*(z[i-1, n] - z[i-1, j])
                p[i, n] = p[i-1, n] + h * (kappa * sum_neighbors)
                z[i, n] = u[i, n] - p[i, n]

        # compute errors with full comm
        gt_states = u_avg
        estim_states = z
        err = 0
        for node in range(0, N):
            for time in range(0, t.shape[0]):
                err = err + (torch.linalg.norm(gt_states[time] - estim_states[time, node])
                             * torch.linalg.norm(gt_states[time] - estim_states[time, node]))
        mserr = err / (N * t.shape[0])
        mserrs_batch[b] = mserr

    return mserrs_batch


def full_comm_consensus(batch_size, t_batch, u_batch, u_avg_batch, Adj, kappa):
    # estimates for the consensus algorithm at full communication
    estims_batch = []
    N = Adj.shape[0]

    for b in range(0, batch_size):
        t = t_batch[b]
        u = u_batch[b]
        u_avg = u_avg_batch[b]

        h = t[1] - t[0]

        # initialize variables
        p = torch.zeros(t.shape[0], N)
        z = torch.zeros(t.shape[0], N)

        p0 = torch.randn(N)
        p0[0] = -torch.sum(p0[1:])  # ensure orthogonality
        p[0, :] = p0


        # run consensus algorithm
        for i in range(1, t.shape[0]):
            for n in range(0, N):
                sum_neighbors = torch.zeros(1)
                for j in range(0, N):
                    sum_neighbors = sum_neighbors + Adj[n, j]*(z[i-1, n] - z[i-1, j])
                p[i, n] = p[i-1, n] + h * (kappa * sum_neighbors)
                z[i, n] = u[i, n] - p[i, n]

        estims_batch.append(z)

    return estims_batch