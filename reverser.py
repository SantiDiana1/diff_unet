import os
import torch
import numpy as np

from time_unet import UNET
from params import AttrDict, params


class ReverseProcess:
    def __init__(self, steps, device="cuda"):
        self.model = None
        self.steps = steps
        self.device = device

    def load_model(self, model_dir):
        device = torch.device('cuda')
        
        if os.path.exists(f'{model_dir}/weights.pt'):
            checkpoint = torch.load(f'{model_dir}/weights.pt')
        else:
            checkpoint = torch.load(model_dir)
        self.model = UNET().to(device)
        self.model.load_state_dict(checkpoint['model'])
        #self.model.eval()

    def predict(self, signal, device="cpu"):
        """Reverse the process of the model.
        Args:
            model: torch.nn.Module, model to reverse.
            schedule: list
        Returns:
            output signal
        """
        #signal=signal.to(self.device)
        if self.model is None:
            raise ValueError("Model not loaded")
        base = torch.ones(np.shape(signal)[0], dtype=torch.int32).to(self.device)
        for t in range(self.steps, 0, -1):
            # print(signal.shape,'shape signal')
            # print(signal.squeeze(0).shape, 'shape con squeeze 0')
            # print(signal.squeeze(1).shape, 'shape con squeeze 1')
            # quit()
            signal = self.model(signal.to(self.device), base * t).squeeze(0).cpu().detach()
            #signal = self.model(signal.to(self.device), base * t).cpu().detach()
            #print(signal.shape)
            #print(torch.cuda.memory_allocated(0))
        return signal.to(self.device)



