import torch 
import torch.nn as nn
import os

class CNNBlock(nn.Module):
    def __init__(self, n_filters, kernel_size, padding, initializer, activation='relu'):
        super(CNNBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i, (f, p) in enumerate(zip(n_filters, padding)):
            extra = i != 0
            if i != 2:
                layer = nn.Conv1d(in_channels=n_filters[i], out_channels=n_filters[i+1], kernel_size=kernel_size, padding=p)
            else:
                layer = nn.Conv1d(in_channels=f, out_channels=f, kernel_size=kernel_size, padding=p)
                
            initializer(layer.weight)
            self.layers.append(layer)
            if extra:
                self.layers.append(nn.Dropout(0.5))
                if i != 2:
                    self.layers.append(nn.BatchNorm1d(n_filters[i+1], momentum=0.9, affine=True, track_running_stats=True))
                else:
                    self.layers.append(nn.BatchNorm1d(f, momentum=0.9, affine=True, track_running_stats=True))
                self.layers.append(nn.ReLU(inplace=True))
            
    def forward(self, x):
        for layer in self.layers:
            print(layer)
            x = layer(x)
        return x
    
class CNN_CONTROL(nn.Module): ## Hay que revisar esto bien. 
    def __init__(self,n_conditions,n_filters):
        super(CNN_CONTROL, self).__init__()

        self.Z_DIM = 4
        self.padding = [1,1,1]
        """
        For simple dense control:
            - n_conditions: 6
            - n_filters: [16,32,128]

        """
        # Define the layers
        self.input_conditions = nn.Linear(self.Z_DIM, 1)
        self.cnn = CNNBlock(n_filters, self.Z_DIM, self.padding)
        self.gammas = nn.Linear(n_filters[-1], n_conditions)
        self.betas = nn.Linear(n_filters[-1], n_conditions)
        
        # Initialize the weights
        self._init_weights()
        
    def _init_weights(self):
        # Initialize the weights with random_normal_initializer
        nn.init.normal_(self.input_conditions.weight, std=0.02)
        nn.init.normal_(self.cnn.weight, std=0.02)
        nn.init.normal_(self.gammas.weight, std=0.02)
        nn.init.normal_(self.betas.weight, std=0.02)
        
    def forward(self, input_conditions):
        cnn_output = self.cnn(input_conditions)
        gammas = self.gammas(cnn_output)
        betas = self.betas(cnn_output)
        
        return input_conditions, gammas, betas



# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# n_filters = [16, 32, 128]
# kernel_size = 4 ## 4 filters because there are 4 sources. 
# padding = [1, 1, 1] ## "same", "same", "valid"
# initializer = nn.init.xavier_uniform_
# input_tensor = torch.randn(16, 16, 128)

# cnn_block = CNNBlock(n_filters, kernel_size, padding, initializer)
# output = cnn_block(input_tensor)


        
