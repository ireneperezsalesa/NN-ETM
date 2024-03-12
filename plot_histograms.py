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


# Plot histograms generated from histograms.py

import torch
import numpy as np
from utils.generate_data import *
import matplotlib.pyplot as plt
import scienceplots

Nsim = 1000  # number of runs for the histogram experiment

# Read values from txt file containing results
with open('./figs/histogram_N5_1000sims.txt', 'r') as f:
    error1 = torch.zeros(Nsim)
    error2 = torch.zeros(Nsim)
    error3 = torch.zeros(Nsim)
    comm1 = torch.zeros(Nsim)
    comm2 = torch.zeros(Nsim)
    comm3 = torch.zeros(Nsim)  # as many as models used for the histogram experiment

    i = 0
    for line in f:

        e1, e2, e3, c1, c2, c3, n = line.split(' ')
        e1, e2, e3, c1, c2, c3 = float(e1), float(e2), float(e3), float(c1), float(c2), float(c3)
        error1[i], error2[i], error3[i] = e1, e2, e3
        comm1[i], comm2[i], comm3[i] = c1, c2, c3

        i = i + 1

# Plot histograms
plt.style.use(['science', 'ieee'])
plt.rcParams['text.usetex'] = True

bins_comm = np.arange(0, 1, 0.01)
bins_err = np.arange(0, 4, 0.05)

fig, axs = plt.subplots(2, 1)
fig.tight_layout(pad=1)
fig.set_figheight(3.6)

axs[0].hist(error3, bins=bins_err, color=[1, 0, 0, 0.2], edgecolor='r', linewidth=0.5, label='$\lambda = 1$')
axs[0].hist(error2, bins=bins_err, color=[0, 1, 0, 0.2], edgecolor='g', linewidth=0.5, label='$\lambda = 0.1$')
axs[0].hist(error1, bins=bins_err, color=[0, 0, 1, 0.2], edgecolor='b', linewidth=0.5, label='$\lambda = 0.001$')
axs[0].set(xlabel='Error $\mathcal{E}_r$', ylabel='Frequency')
axs[0].set_yscale('log')
legend = axs[0].legend(frameon='true', facecolor='white', framealpha=1)
frame = legend.get_frame()
frame.set_linewidth(0)

axs[1].hist(comm1, bins=bins_comm, color=[0, 0, 1, 0.2], edgecolor='b', linewidth=0.5, label='$\lambda = 0.001$')
axs[1].hist(comm2, bins=bins_comm, color=[0, 1, 0, 0.2], edgecolor='g', linewidth=0.5, label='$\lambda = 0.1$')
axs[1].hist(comm3, bins=bins_comm, color=[1, 0, 0, 0.2], edgecolor='r', linewidth=0.5, label='$\lambda = 1$')
axs[1].set(xlabel='Communication rate $\mathcal{C}$', ylabel='Frequency')

fig.savefig('./figs/hist.pdf')
