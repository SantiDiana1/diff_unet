import torch 
import torch.nn as nn


class CNNBlock(nn.Module):
    def __init__(self, n_filters, kernel_size, padding, initializer, activation='relu'):
        super(CNNBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i, (f, p) in enumerate(zip(n_filters, padding)):
            extra = i != 0
            layer = nn.Conv1d(in_channels=f, out_channels=f, kernel_size=kernel_size, padding=p)
            initializer(layer.weight)
            self.layers.append(layer)
            if extra:
                self.layers.append(nn.Dropout(0.5))
                self.layers.append(nn.BatchNorm1d(f, momentum=0.9, affine=True, track_running_stats=True))
                self.layers.append(nn.ReLU(inplace=True))
            
    def forward(self, x):
        for layer in self.layers:
            print(layer)
            x = layer(x)
        return x


n_filters = [16, 32, 128]
kernel_size = 4 ## 4 filters because there are 4 sources. 
padding = [1, 1, 1]
initializer = nn.init.xavier_uniform_
input_tensor = torch.randn(16, 16, 100)

cnn_block = CNNBlock(n_filters, kernel_size, padding, initializer)
output = cnn_block(input_tensor)
