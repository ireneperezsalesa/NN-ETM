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

from utils.generate_data import load_graph
from utils.generate_data import generate_signals

# Generate a batch of sequences of reference signals, to be used for training or test

h = 1e-3 # simulation step
T = 5 # length of each sequence
N = 3 # number of agents in the network
Adj = load_graph(N) # adjacency matrix for communication graph
batch_size = 20 # number of sequences in the batch
times_batch, u_batch, u_avg_batch = generate_signals(N, h, T, batch_size)

data_to_save = {'batch_size': batch_size, 't': times_batch, 'u': u_batch, 'u_avg': u_avg_batch}
torch.save(data_to_save, 'data/random_20_seqs.pt')
