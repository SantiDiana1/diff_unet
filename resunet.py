import torch
import torch.nn as nn


class RCB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RCB, self).__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(in_channels, out_channels, kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(out_channels, out_channels, kernel_size=3)
        )

    def forward(self,x):
        return self.block(x)

class UNET(nn.Module):
    def __init__(
            self,in_channels=1,out_channels=1,features = [16,32,64,128,256,512], n_rcb_per_encoder=4, n_rcb_per_decoder=4, n_icbs = 4
    ):
        super(UNET,self).__init__()

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
            for _ in range(n_rcb_per_encoder):  ## Según esto, in_channels cambia a feature en el primer RCB y se queda así hasta que en el siguiente bloque de encoder cambia features. 
                self.downs_level.append(RCB(in_channels,feature))
                in_channels=feature
            self.downs_total.append(self.downs_level)
            self.downs_level = nn.ModuleList()  ## Lo que hacemos aquí es añadir en cada nivel del encoder un downs_level que contiene 4 RCBs, para 
            ## luego en el forward poder sacar bien la skip connection. 

         ## Bottleneck

        for _ in range(n_icbs-1):
            self.bottleneck.append(RCB(features[-1],features[-1]))

        self.bottleneck.append(RCB(features[-1],features[-1]*2))
            
        ## Up part 

        for feature in reversed(features): ## Aquí en cada iteración se añaden 2 elementos a self.ups.
            self.ups_total.append(nn.ConvTranspose2d(feature*2,feature,kernel_size=2,stride=2))
            self.ups_level.append(RCB(feature*2,feature)) ## Primer RCB del bloque donde se cambian los canales. 
            for _ in range(n_rcb_per_decoder-1):
                self.ups_level.append(RCB(feature,feature))
            self.ups_total.append(self.ups_level)
            self.ups_level = nn.ModuleList() ##Limpio para luego añadir a ups_total solo el nuevo nivel. 
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

        ## Down process. 
        for down in self.downs_total: ## Debería tener 6 bloques que a su vez contienen 4 RCB cada uno. 
            x = down(x)
            skip_connections.append(x) ## Esto es el final de 4 RCBs. 
            x = self.pool(x)

        ## Bottleneck.
        for bottle in range(self.bottleneck):
            x = bottle(x)

        skip_connections = skip_connections[::-1] ## Reverse the list of skip connections.

        for idx in range(0,len(self.ups_total),2):
            x = self.ups_total[idx](x)
            skip_connection = skip_connections[idx//2] ## Addition of the skip connection before the DoubleConv.
            concat_skip = torch.cat((skip_connection,x),dim=1)
            x = self.ups_total[idx+1](concat_skip) ## Se aplica todo en la primera. 

        for out in range(self.out):
            x = out(x)

        x = self.sigmoid(x)
        result = x * x_input
        return result
    

def test():
    x = torch.randn((1,1,512,128))
    model = UNET(in_channels=1,out_channels=1)
    preds = model(x)
    print(preds.shape)
    print(x.shape)
    assert preds.shape == x.shape 

if __name__ == "__main__":
    test()


### ESTE ERROR ME PUEDE ESTAR SALIENDO QUIZÁ PORQUE TENGO IR DE UNO EN UNO EN EL FORWARD Y NO METERLE UN MODULE LIST DE MODULE LISTS 
### DIRECTAMENTE??

### Por cierto, me falta el shortcut connection ?¿?¿