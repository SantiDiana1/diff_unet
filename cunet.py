import torch
import torch.nn as nn

from control_model import CNN_CONTROL
from FILM_utils import FiLM_Simple_Layer


class RCB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RCB, self).__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(in_channels, out_channels, kernel_size=3,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(out_channels, out_channels, kernel_size=3,padding=1)
        )
        
    def forward(self,x):
        
        return self.block(x)
    
class RCB2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RCB2, self).__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(in_channels, in_channels, kernel_size=3,padding=1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(in_channels, out_channels, kernel_size=3,padding=1)
        )

    def forward(self,x):
        
        return self.block(x)

class UNET(nn.Module):
    def __init__(
            self,in_channels=1,out_channels=1,features = [16,32,64,128,256,512], n_rcb_per_encoder=4, n_rcb_per_decoder=4, n_icbs = 4
    ):
        super(UNET,self).__init__()


        self.features = features
        self.n_rcb_per_encoder = n_rcb_per_encoder
        self.n_rcb_per_decoder = n_rcb_per_decoder
        self.downs_total = nn.ModuleList()
        self.downs_level = nn.ModuleList()
        self.ups_total = nn.ModuleList()
        self.ups_level = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.out = nn.ModuleList()
        self.pool = nn.AvgPool2d(kernel_size=2)
        self.bottleneck = nn.ModuleList()

        ## Down part of the Unet.
        for feature in features: ## feature is out_channels. 
            for _ in range(n_rcb_per_encoder):  
                self.downs_level.append(RCB(in_channels,feature)) 
                in_channels=feature
            
         ## Bottleneck

        for _ in range(n_icbs-1):
            self.bottleneck.append(RCB(features[-1],features[-1]))

        self.bottleneck.append(RCB(features[-1],features[-1]*2))
            
        
        ## Up part of the Unet
        for feature in reversed(features): ## Aquí en cada iteración se añaden 2 elementos a self.ups.
            self.ups_level.append(nn.ConvTranspose2d(feature*2,feature,kernel_size=2,stride=2))
            
            for _ in range(n_rcb_per_decoder-1):
                self.ups_level.append(RCB(feature*2,feature*2))

            self.ups_level.append(RCB2(feature*2,feature)) 

        ## Out part. 
        for _ in range(n_rcb_per_decoder-1):
            self.out.append(RCB(features[0],features[0]))

        self.out.append(RCB(features[0],out_channels))
        
        ## Sigmoid. 
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        skip_connections = []
        x_input = torch.clone(x.unsqueeze(1))
        
        # x = self.preprocess(x)
        x = x.unsqueeze(1)

        
        for i in range(0,len(self.downs_level),4):
            for levels in range(self.n_rcb_per_encoder):
                down = self.downs_level[i+levels] ### Aquí seleccionamos cual es el siguiente bloque
                x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        ## Bottleneck.
        for bottle in self.bottleneck:
            x = bottle(x)
        
        skip_connections = skip_connections[::-1] ## Reverse the list of skip connections.

        
        for idx in range(0,len(self.ups_level),5):
            
            x = self.ups_level[idx](x)
            skip_connection = skip_connections[idx//5]
            concat_skip = torch.cat((skip_connection,x),dim=1)
            
            for levels in range(self.n_rcb_per_decoder):
                up = self.ups_level[idx+levels+1]
                x = up(concat_skip) ## Se aplica todo en la primera. 

        
        
        for out in self.out:
            x = out(x)

        x = self.sigmoid(x)
        result = x * x_input
        return result
    

