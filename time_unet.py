import torch
import torch.nn as nn
import math

from params import params
import os


class DoubleConv(nn.Module):
    def __init__(self,in_channels,out_channels,emb_channels):
        super(DoubleConv,self).__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels, out_channels)
        )

    def forward(self,x,emb):
        
        x = x.to("cuda")
        x = self.conv(x)
        if emb is not None:
            emb_out = self.emb_layers(emb)

            while len(emb_out.shape) < len(x.shape): ## Cambio de shape del embedding. 
                emb_out = emb_out[..., None]

            x = x + emb_out

        return x
    
class UNET(nn.Module):
    def __init__(
            self,in_channels=1,out_channels=1,features = [16,32,64,128,256,512,1024],model_channels=16
    ):
        super(UNET,self).__init__()

        self.model_channels = model_channels
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2) ##El downsampling no tiene time embedding. 


        emb_channels = model_channels * 4
        
        self.time_embedding = nn.Sequential(  ## Esto se inicializa aquí pero ya se usa más adelante en el forward. 
            nn.Linear(model_channels, emb_channels),
            nn.SiLU(),
            nn.Linear(emb_channels, emb_channels),
        )

        # ## Preprocessing
        # self.preprocess = nn.Conv2d(in_channels,model_channels,kernel_size=3,stride=1,padding=1,bias=False)


        ## Down part of the Unet.
        # in_channels=16
        for feature in features:
            self.downs.append(DoubleConv(in_channels,feature,emb_channels))
            in_channels = feature
            
        ## Up part 

        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2,feature,kernel_size=2,stride=2   ## kernel_size = 2, stride = 2 will double the size of the image. El upsampling no tiene time embedding. 
                )
            )
            self.ups.append(DoubleConv(feature*2,feature,emb_channels))

        ## Bottleneck
        self.bottleneck = DoubleConv(features[-1],features[-1]*2,emb_channels)
        self.out = nn.Conv2d(features[0],out_channels,kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x,timesteps):

        if timesteps is not None:
            emb = self.time_embedding(timestep_embedding(timesteps, self.model_channels))
        else:
            emb = None

        skip_connections = []
        x_input = torch.clone(x.unsqueeze(1)).to("cuda")
        
        # x = self.preprocess(x)
        x = x.unsqueeze(1)
        for down in self.downs:
            x = down(x,emb)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x,emb)
        skip_connections = skip_connections[::-1] ## Reverse the list of skip connections.

        for idx in range(0,len(self.ups),2):
            x = self.ups[idx](x) ## ConvTransposed 2d.
            skip_connection = skip_connections[idx//2] ## Addition of the skip connection before the DoubleConv.
            concat_skip = torch.cat((skip_connection,x),dim=1)
            x = self.ups[idx+1](concat_skip,emb) ## DoubleConv. 

        x = self.out(x)
        x = self.sigmoid(x)
        result = x * x_input
        result = result.squeeze(1)
        return result
    


def timestep_embedding(timesteps, dim, max_period=10000):
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
    


# def test():
#     os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#     x = torch.randn((params.batch_size,512,128))
#     model = UNET(in_channels=1,out_channels=1).to("cuda")
#     N = params.batch_size
#     timesteps = torch.randint(1, params.iters + 1, [N], device="cuda")
#     preds = model(x,timesteps)
#     print(preds.shape)
#     print(x.shape)
#     assert preds.shape == x.shape 

# if __name__ == "__main__":
#     test()
