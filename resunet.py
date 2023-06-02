import torch
import torch.nn as nn


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
        # self.norm = nn.BatchNorm2d(in_channels)
        # self.relu = nn.LeakyReLU(negative_slope=0.01)
        # self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,padding = 1)
        # self.norm2 = nn.BatchNorm2d(out_channels)
        # self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3)

    def forward(self,x):
        # print(x.shape,'shape de x')
        # x = self.norm(x)
        # print(x.shape,'shape de x')
        # x = self.relu(x)
        # print(x.shape,'shape de x')
        # x = self.conv1(x)
        # print(x.shape,'shape de x')
        # x = self.norm2(x)
        # print(x.shape,'shape de x')
        # x = self.conv2(x)
        # print(x.shape,'shape de x')
        # quit()
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
            for _ in range(n_rcb_per_encoder):  ## Según esto, in_channels cambia a feature en el primer RCB y se queda así hasta que en el siguiente bloque de encoder cambia features. 
                self.downs_level.append(RCB(in_channels,feature)) ## downs_level es una lista de nn.Sequentials. 
                in_channels=feature
            #self.downs_total.append(self.downs_level)
            #self.downs_level = nn.ModuleList()  ## Lo que hacemos aquí es añadir en cada nivel del encoder un downs_level que contiene 4 RCBs, para 
            ## luego en el forward poder sacar bien la skip connection. 

         ## Bottleneck

        for _ in range(n_icbs-1):
            self.bottleneck.append(RCB(features[-1],features[-1]))

        self.bottleneck.append(RCB(features[-1],features[-1]*2))
            
        ## Up part 

        # for feature in reversed(features): ## Aquí en cada iteración se añaden 2 elementos a self.ups.
        #     self.ups_total.append(nn.ConvTranspose2d(feature*2,feature,kernel_size=2,stride=2))
        #     self.ups_level.append(RCB(feature*2,feature)) ## Primer RCB del bloque donde se cambian los canales. 
        #     for _ in range(n_rcb_per_decoder-1):
        #         self.ups_level.append(RCB(feature,feature))
        #     self.ups_total.append(self.ups_level)
        #     self.ups_level = nn.ModuleList() ##Limpio para luego añadir a ups_total solo el nuevo nivel. 

        for feature in reversed(features): ## Aquí en cada iteración se añaden 2 elementos a self.ups.
            self.ups_level.append(nn.ConvTranspose2d(feature*2,feature,kernel_size=2,stride=2))
            self.ups_level.append(RCB(feature*2,feature)) ## Primer RCB del bloque donde se cambian los canales. 
            for _ in range(n_rcb_per_decoder-1):
                self.ups_level.append(RCB(feature,feature))

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
                #print(down,'down')
                x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        ## Bottleneck.
        for bottle in self.bottleneck:
            x = bottle(x)
        
        skip_connections = skip_connections[::-1] ## Reverse the list of skip connections.

        print(self.ups_level)
        quit()
        
        for idx in range(0,len(self.ups_level),5):
            print(x.shape,'antes de trans')
            x = self.ups_level[idx](x)
            print(x.shape,'despues de trans')
            
            skip_connection = skip_connections[idx//5]
            concat_skip = torch.cat((skip_connection,x),dim=1)
            print(concat_skip.shape,'shape concat')
            for levels in range(self.n_rcb_per_decoder):
                up = self.ups_level[idx+levels+1]
                x = up(concat_skip) ## Se aplica todo en la primera. 

        print('Acabado')
        
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