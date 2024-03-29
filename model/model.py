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


# Neural network architecture for the ETM

from torch import nn


class Simple(nn.Module):  # simple network
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        out = self.seq(x.float())
        return out


class RNN(nn.Module):  # recurrent neural network
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(input_size=4, hidden_size=1)
        self.sig = nn.Sigmoid()

    def forward(self, x, prevstate):
        temp, state = self.gru(x, prevstate)
        out = self.sig(temp)
        return out, state


