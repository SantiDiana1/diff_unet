import glob
from utils import load_and_resample, compute_signal_from_stft2, compute_stft, prev_power_of_2
import math
import torch
import os
from resunet import UNET
import numpy as np

from params import params
import museval
import soundfile
import torch.nn as nn

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def load_model(model_dir,bestsdr):
    device = torch.device('cuda')
    
    if bestsdr:
        if os.path.exists(f'{model_dir}/weights_bestSDR.pt'):
            print('True best SDR')
            checkpoint = torch.load(f'{model_dir}/weights_bestSDR.pt')
        else:
            checkpoint = torch.load(model_dir)
    else:
        if os.path.exists(f'{model_dir}/weights.pt'):
            print('True')
            checkpoint = torch.load(f'{model_dir}/weights.pt')
        else:
            checkpoint = torch.load(model_dir)

    model = UNET().to(device)
    model.load_state_dict(checkpoint['model'])
    #model.eval()
    return model




def main (args=None):

    entire_med_sdr_voc = []
    entire_med_sdr_acc = []

    bestsdr = True

    model = load_model("./ckpt2/model",bestsdr)  
    mus = glob.glob("/home/santi/datasets/musdb_test/*/*/mixture.wav")
    c=0
    #a=0
    for count,filename in (enumerate(mus)):
        print(filename,count, 'Filename')
        mixture = load_and_resample(filename)
        vocals = load_and_resample(filename.replace("mixture.wav", "vocals.wav"))

        output_voice = []
        trim_low = 0
        duration = int(trim_low + (15*22050)) // params.n_hop * params.n_hop
        sec = math.ceil(mixture.shape[0] / 22050)

        for trim in (np.arange(math.ceil(sec / 15))):
            trim_high = trim_low + duration
            if trim_high > mixture.shape[0]:
                #print("last one")
                if (mixture.shape[0] // params.n_hop * params.n_hop) - trim_low > params.n_hop:
                    trim_high = mixture.shape[0] // params.n_hop * params.n_hop
                    if int(trim_high - trim_low) > 4096:
                        mixture_analyse = mixture[trim_low:trim_high]
                        #print(trim_low, trim_high)
                    else:
                        mixture_analyse = None
                else:
                    mixture_analyse = None
            else:
                mixture_analyse = mixture[trim_low:trim_high]


            if mixture_analyse is not None:
                # [1, T], iter x [1, T]
                mix_mag, mix_phase = compute_stft(mixture_analyse[None], params)
                #print(mix_mag.shape,'mix_magnitude')
                new_len = prev_power_of_2(mix_mag.shape[2])
                #mix_mag=torch.transpose(mix_mag,1,2)
                mix_mag = mix_mag[:, :, :new_len]
                mix_phase = mix_phase[:, :, :new_len]
                mix_mag=mix_mag.to("cuda")
                mix_phase=mix_phase.to("cuda")
                
                #mix_mag=mix_mag.unsqueeze(1) #TODO: he añadido un unsqueeze porque sino peta el predict. 
                diff_res = model(mix_mag)

                output_signal = diff_res[:, :mix_mag.shape[1], :]
                #print(output_signal.shape,'shape stft')
                #print(mix_phase.shape,'shape de la fase')
                #save_stft_image(mix_mag,f"stfts3/{c}.png")
                #save_stft_image(output_signal,f"stfts2/{c}.png")
                output_signal = compute_signal_from_stft2(output_signal, mix_phase, params).to("cpu").detach().numpy()
                #print(output_signal.shape,'shape output signal')
                mixture_aux = mixture_analyse[:output_signal.shape[0]].cpu().detach().numpy()
                del mix_mag, mix_phase
                #output_accomp = mixture_aux - output_signal
                #output_signal = output_signal.squeeze(0)
                #output_accomp = output_accomp.squeeze(0)
                if trim == 0:
                    output_voice = output_signal
                    #output_accompaniment = output_accomp
                else:
                    output_voice = np.concatenate([output_voice, output_signal])
                    #output_accompaniment = np.concatenate([output_accompaniment, output_accomp])

                ### SANTI: Getting here the exact point where we have cropped in order not to lose any music between chunks
                trim_low = output_voice.shape[0]
                torch.cuda.empty_cache()
    
        #print('Now we calculate estimates and scores')

        voc_ref = vocals[:output_voice.shape[0]].detach().numpy()
        #print(voc_ref.shape,'voc ref shape')
        # Getting array of estimates
        #output_voice= (output_voice * max(abs(voc_ref)) / max(abs(output_voice)))
        c=c+1
        soundfile.write(f"audios2/inference_bestmodel/audio{c}.wav", output_voice, 22050)
        estimates = np.array([output_voice])[..., None]

        scores = museval.evaluate(
            np.array([voc_ref])[..., None], estimates, win=22050, hop=22050)
        
        voc_sdr = scores[0][0]
        voc_sdr = np.round(np.median(voc_sdr[~np.isnan(voc_sdr)]), 3)

        print("VOCALS ==> SDR:", voc_sdr) #, " SIR:", voc_sir, " SAR:", voc_sar)
        entire_med_sdr_voc.append(voc_sdr)

    median_sdr_voc=np.median(entire_med_sdr_voc)
    print('All median SDR for vocals:',median_sdr_voc)
    

if __name__ == '__main__':
    main()