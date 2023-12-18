import torchaudio
from torchaudio import transforms
import torch
import numpy as np


def load_and_resample(track):
    
    mixture,sr=torchaudio.load(track)
    mixture=mixture.to("cpu")
    mixture=mixture.squeeze(0)
    print(mixture.shape, 'primera sr')
    if sr!=22050:
        transform = transforms.Resample(sr, 22050)
        mixture = transform(mixture)
    if mixture.shape[0]==2:
        mixture = torch.mean(mixture, dim=0, keepdim=False)
    print(mixture.shape, 'segunda sr')
    return mixture


def compute_stft(signal, params):
    #import pdb; pdb.set_trace()
    window=torch.hann_window(params.n_win)
    signal_stft= check_shape_3d(
        check_shape_3d(torch.stft(
            signal,
            n_fft=params.n_fft,
            hop_length=params.n_hop,
            window=window,
            center=True,
            return_complex=True),1),2)
    mag = torch.abs(signal_stft)
    phase = torch.angle(signal_stft)

    return mag, phase


def compute_signal_from_stft2(magnitude, phase, params):
    # Combine the magnitude and phase into a complex spectrogram
    #print(magnitude.shape, 'Shape de la magnitud que entra')
    complex_spec = torch.stack([magnitude * torch.cos(phase), magnitude * torch.sin(phase)], dim=-1) 
    #print(complex_spec.shape,'shape del espectro concatenado') 
    spec_phase = torch.transpose(complex_spec, 2, 0) 
     
    spec_phase = torch.transpose(spec_phase, 0, 1).contiguous() 
    spec_phase=torch.squeeze(spec_phase)
    #print(spec_phase.shape,'shape spec')
    spec_phase = torch.cat([spec_phase, torch.zeros(1, spec_phase.shape[1], 2).to("cuda")])
    #print(spec_phase.shape,'debería ser 513')
    # Reconstruct the audio signal from the complex spectrogram
    spec_phase=torch.view_as_complex(spec_phase)
    #print(spec_phase.shape,'shape despues de view as complex, esto entra a istft')
    audio_signal = torch.istft(spec_phase,n_fft=params.n_fft, hop_length=params.n_hop, window=torch.hann_window(params.n_win).to("cuda"))
    #print(audio_signal.shape,'shape del audio')    
    return audio_signal

def log2(x, base):
    return int(np.log(x) / np.log(base))

def prev_power_of_2(n):
    # decrement `n` (to handle the case when `n` itself is a power of 2)
    n = n - 1
    # calculate the position of the last set bit of `n`
    lg = log2(n, 2)
    # previous power of two will have a bit set at position `lg+1`.
    return 1 << lg #+ 1


def check_shape_3d(data, dim):
    n = data.shape[dim]
    if n % 2 != 0:
        n = data.shape[dim] - 1
    if dim==0:
        return data[:n, :, :]
    if dim==1:
        return data[:, :n, :]
    if dim==2:
        return data[:, :, :n]