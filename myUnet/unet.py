import torch
import torch.nn as nn



class DoubleConv(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(DoubleConv,self).__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # self.conv1= nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1,bias=False)
        # self.norm= nn.BatchNorm2d(out_channels)
        # self.relu = nn.ReLU(inplace=True)
        # self.conv2 = nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1,bias=False)

    def forward(self,x):
    #     x = self.conv1(x)
    #     x = self.norm(x)
    #     x = self.relu(x)
    #     x = self.conv2(x)
    #     x = self.norm(x)
    #     x = self.relu(x)

        return self.conv(x)
    
class UNET(nn.Module):
    def __init__(
            self,in_channels=1,out_channels=1,features = [16,32,64,128,256,512,1024],model_channels=16
    ):
        super(UNET,self).__init__()

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)

        # ## Preprocessing
        # self.preprocess = nn.Conv2d(in_channels,model_channels,kernel_size=3,stride=1,padding=1,bias=False)


        ## Down part of the Unet.
        # in_channels=16
        for feature in features:
            self.downs.append(DoubleConv(in_channels,feature))
            in_channels = feature
            
        ## Up part 

        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2,feature,kernel_size=2,stride=2   ## kernel_size = 2, stride = 2 will double the size of the image. 
                )
            )
            self.ups.append(DoubleConv(feature*2,feature))

        ## Bottleneck
        self.bottleneck = DoubleConv(features[-1],features[-1]*2)
        self.out = nn.Conv2d(features[0],out_channels,kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        skip_connections = []
        x_input = torch.clone(x.unsqueeze(1))
        
        # x = self.preprocess(x)
        x = x.unsqueeze(1)
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1] ## Reverse the list of skip connections.

        for idx in range(0,len(self.ups),2):
            x = self.ups[idx](x) ## ConvTransposed 2d.
            skip_connection = skip_connections[idx//2] ## Addition of the skip connection before the DoubleConv.
            concat_skip = torch.cat((skip_connection,x),dim=1)
            x = self.ups [idx+1](concat_skip) ## DoubleConv. 

        x = self.out(x)
        x = self.sigmoid(x)
        result = x * x_input
        return result
    

# def test():
#     x = torch.randn((1,1,512,128))
#     model = UNET(in_channels=1,out_channels=1)
#     preds = model(x)
#     print(preds.shape)
#     print(x.shape)
#     assert preds.shape == x.shape 

# if __name__ == "__main__":
#     test()
