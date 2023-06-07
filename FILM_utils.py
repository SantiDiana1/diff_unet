import torch
import torch.nn as nn

class FiLM_Simple_Layer(nn.Module):
    def __init__(self):
        super(FiLM_Simple_Layer, self).__init__()

    def forward(self, x, gamma, beta):
        s = list(x.shape)
        s[0] = 1
        s[2] = 1

        g = gamma.transpose(1, 2).unsqueeze(-1).unsqueeze(-1).expand(s)
        b = beta.transpose(1, 2).unsqueeze(-1).unsqueeze(-1).expand(s)

        return b + x * g
        