import os
import torch
import numpy as np
import librosa
import matplotlib.pyplot as plt


from time_unet import UNET
from params import AttrDict, params

def save_stft_image(mix_mag, output_file):
    # # Convert STFT tensor to numpy array and take magnitude
    # stft_mag = np.abs(stft_tensor.cpu().numpy())

    # # Select a single channel from the magnitude array
    # mag_channel = stft_mag[0,:,:]

    # # Normalize magnitude values to [0, 1] range
    # mag_channel -= mag_channel.min()
    # mag_channel /= mag_channel.max()
    mix_mag=mix_mag.cpu() 
    D = librosa.power_to_db(mix_mag.squeeze().numpy(), ref=np.max)

    # Convert magnitude to grayscale image and save to file
    plt.imsave(output_file, D, cmap='viridis')

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

    def predict(self, signal,device="cpu"):
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
        a=0
        # previous_signal = signal
        for t in range(self.steps, 0, -1):
            # print(signal.shape,'shape signal')
            # print(signal.squeeze(0).shape, 'shape con squeeze 0')
            # print(signal.squeeze(1).shape, 'shape con squeeze 1')
            # quit()
            signal = self.model(signal.to(self.device), base * t).squeeze(0).cpu().detach()

            
            a = a+1
            #save_stft_image(signal,f'audios_diff_resunet_8steps/results_diff/specs/spec{a}_{epoch}.png')
            # previous_signal = signal
            # print(np.mean(np.array(signal)),'C',a)
            # # print(signal.shape)
            # a = a+1
            # #save_stft_image(signal,f'specs_change/spec{a}.png')
            # if a == 8:
            #     quit()
            #signal = self.model(signal.to(self.device), base * t).cpu().detach()
            #print(signal.shape)
            #print(torch.cuda.memory_allocated(0))
        return signal.to(self.device)



