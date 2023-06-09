import torch
import torch.nn as nn
import math


class RCB(nn.Module):
    def __init__(self, in_channels, out_channels,emb_channels):
        super(RCB, self).__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(in_channels, out_channels, kernel_size=3,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(out_channels, out_channels, kernel_size=3,padding=1)
        )

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels, out_channels)
        )
    def forward(self,x,emb):
        
        x = x.to("cuda")
        
        x = self.block(x)
        if emb is not None:
            emb_out = self.emb_layers(emb)

            while len(emb_out.shape) < len(x.shape): ## Cambio de shape del embedding. 
                emb_out = emb_out[..., None]

            x = x + emb_out

        return x
    
class RCB2(nn.Module):
    def __init__(self, in_channels, out_channels,emb_channels):
        super(RCB2, self).__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(in_channels, in_channels, kernel_size=3,padding=1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(in_channels, out_channels, kernel_size=3,padding=1)
        )

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels, out_channels)
        )
    def forward(self,x,emb):
        
        x = x.to("cuda")
        
        x = self.block(x)
        
        if emb is not None:
            emb_out = self.emb_layers(emb)
            while len(emb_out.shape) < len(x.shape): ## Cambio de shape del embedding. 
                emb_out = emb_out[..., None]   
            x = x + emb_out
        return x

class UNET(nn.Module):
    def __init__(
            self,in_channels=1,out_channels=1,features = [16,32,64,128,256,512], n_rcb_per_encoder=4, n_rcb_per_decoder=4, n_icbs = 4, model_channels = 16
    ):
        super(UNET,self).__init__()

        self.model_channels = model_channels
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

        emb_channels = model_channels * 4
        
        self.time_embedding = nn.Sequential(  ## Esto se inicializa aquí pero ya se usa más adelante en el forward. 
            nn.Linear(model_channels, emb_channels),
            nn.SiLU(),
            nn.Linear(emb_channels, emb_channels),
        )


        ## Down part of the Unet.
        for feature in features: ## feature is out_channels. 
            for _ in range(n_rcb_per_encoder):  ## Según esto, in_channels cambia a feature en el primer RCB y se queda así hasta que en el siguiente bloque de encoder cambia features. 
                self.downs_level.append(RCB(in_channels,feature,emb_channels)) ## downs_level es una lista de nn.Sequentials. 
                in_channels=feature
            #self.downs_total.append(self.downs_level)
            #self.downs_level = nn.ModuleList()  ## Lo que hacemos aquí es añadir en cada nivel del encoder un downs_level que contiene 4 RCBs, para 
            ## luego en el forward poder sacar bien la skip connection. 

         ## Bottleneck

        for _ in range(n_icbs-1):
            self.bottleneck.append(RCB(features[-1],features[-1],emb_channels))

        self.bottleneck.append(RCB(features[-1],features[-1]*2,emb_channels))
            
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
            
            for _ in range(n_rcb_per_decoder-1):
                self.ups_level.append(RCB(feature*2,feature*2,emb_channels))

            self.ups_level.append(RCB2(feature*2,feature,emb_channels)) 

        ## Out part. 
        for _ in range(n_rcb_per_decoder-1):
            self.out.append(RCB(features[0],features[0],emb_channels))

        self.out.append(RCB(features[0],out_channels,emb_channels))
        
        ## Sigmoid. 
        self.sigmoid = nn.Sigmoid()

    def forward(self,x,timesteps): ##Timesteps es un vector de batch_size valores de t random en el rango. 

        if timesteps is not None:
            emb = self.time_embedding(timestep_embedding(timesteps, self.model_channels))
        else:
            emb = None
        skip_connections = []
        x_input = torch.clone(x.unsqueeze(1))
        
        # x = self.preprocess(x)
        x = x.unsqueeze(1)

        
        for i in range(0,len(self.downs_level),4):
            for levels in range(self.n_rcb_per_encoder):
                down = self.downs_level[i+levels] ### Aquí seleccionamos cual es el siguiente bloque
                x = down(x,emb)
            skip_connections.append(x)
            x = self.pool(x)
        
        
        ## Bottleneck.
        for bottle in self.bottleneck:
            x = bottle(x,emb)
        
        
        skip_connections = skip_connections[::-1] ## Reverse the list of skip connections.

        
        for idx in range(0,len(self.ups_level),5):
            
            x = self.ups_level[idx](x)
            skip_connection = skip_connections[idx//5]
            concat_skip = torch.cat((skip_connection,x),dim=1)
            
            for levels in range(self.n_rcb_per_decoder):
                up = self.ups_level[idx+levels+1]
                x = up(concat_skip,emb) ## Se aplica todo en la primera. 

        
        
        for out in self.out:
            x = out(x,emb)
        
        
        x = self.sigmoid(x)
        result = x * x_input
        
        return result
    
def timestep_embedding(timesteps, dim, max_period=10000): ## Genera batch_size embeddings dependiendo del time step. El embedding del mismo timestep debe ser el mismo.
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding
    



### Por cierto, me falta el SHORTCUT CONNECTION 