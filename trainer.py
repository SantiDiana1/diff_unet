import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim 
from resunet import UNET
import numpy as np
# from utils import (
#     load_checkpoint,
#     save_checkpoint,
#     get_loaders,
#     check_accuracy,
#     save_predictions_as_images
# )
from dataset import from_path
from utils import load_and_resample, compute_stft, compute_signal_from_stft2, prev_power_of_2

import glob
import random
import math
import museval
import soundfile
from torchsummary import summary

def _nested_map(struct, map_fn):
        if isinstance(struct, tuple):
            return tuple(_nested_map(x, map_fn) for x in struct)
        if isinstance(struct, list):
            return [_nested_map(x, map_fn) for x in struct]
        if isinstance(struct, dict):
            return { k: _nested_map(v, map_fn) for k, v in struct.items() }
        return map_fn(struct)


class Learner():
    def __init__(self,model_dir, model, trainset, testset, optimizer, params, *args, **kwargs):
        self.model_dir=model_dir
        self.model = model
        self.trainset = trainset
        self.testset = testset
        self.optimizer = optimizer
        self.params = params
        self.device = "cuda"
        self.loss_fn = nn.L1Loss()
        self.step=0
        self.scaler = torch.cuda.amp.GradScaler(enabled=kwargs.get("fp16", False))

        ## Validation
        self.vector_medians = [0]
        self.best_SDR = 0
        self.best_epoch = 0

    def state_dict(self): ## This returns a dictionary containing the current state of the model and optimizer, and some additional training parameters.
        # It is commonly used for save and load the state of a model during training or to transfer a model between different processes or machines. 
        # By storing the current state of the model and optimizer, it is possible to resume training from the same point later. 
        if hasattr(self.model, "module") and isinstance(self.model.module, nn.Module):
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()
        return {
            "step": self.step,
            "model": { k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in model_state.items() },
            "optimizer": { k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in self.optimizer.state_dict().items() },
            "params": dict(self.params),
            "scaler": self.scaler.state_dict(),
        }

    def load_state_dict(self, state_dict):
        if hasattr(self.model, "module") and isinstance(self.model.module, nn.Module):
            self.model.module.load_state_dict(state_dict["model"])
        else:
            self.model.load_state_dict(state_dict["model"])
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.scaler.load_state_dict(state_dict["scaler"])
        self.step = state_dict["step"]


    def save_to_checkpoint(self,filename):
        save_basename = f"{filename}.pt"
        save_name = f"{self.model_dir}/{save_basename}"
        #link_name = f"{self.model_dir}/{filename}.pt"
        torch.save(self.state_dict(), save_name)

    def restore_from_checkpoint(self,filename = "weights"):
        try:
            checkpoint = torch.load(f"{self.model_dir}/{filename}.pt")
            self.load_state_dict(checkpoint)
            return True
        except FileNotFoundError:
            return False
    
    
    def train(self,max_steps = None):
        print(len(self.trainset),len(self.trainset))
        mus = glob.glob("/home/santi/datasets/musdb_test/*/*/mixture.wav")
        mus=random.sample(mus,4)
        mus.append("/home/santi/datasets/musdb_test/raw_audio/Carlos Gonzalez - A Place For Us/mixture.wav")
        vector_medians= []

        sum = summary(self.model,input_size = (512,128))

        while True:
             with tqdm(total=len(self.trainset), desc=f"Epoch {self.step // len(self.trainset)}", leave=False) as pbar:
                epoch_loss = 0
                for features in self.trainset:
                    if max_steps is not None and self.step >= max_steps: ##Esto sirve para acabar cuando max_steps es None o cuando self.step se pasa de max_steps. 
                        return
                         
                    features = _nested_map(features, lambda x: x.to(self.device) if isinstance(x, torch.Tensor) else x)     

                    loss, grad = self.train_step (features)

                    if self.step % len(self.trainset) == 0:
                        self.save_to_checkpoint("weights")    
                    self.step += 1

                    pbar.update()
                    pbar.set_postfix(
                        {"loss": loss.item(),
                        "grad": grad.item(),
                        "step": self.step})
                    epoch_loss += loss.item()
                print("Train loss:", epoch_loss / len(self.trainset))

                train_loss = epoch_loss / len(self.trainset)
                
                with open("results2/train_loss.txt", "a") as file:
                    file.write(str(train_loss) + "\n")

                validation_loss = []
                for features in self.testset:
                    features = _nested_map(features, lambda x: x.to(self.device) if isinstance(x, torch.Tensor) else x)
                    loss = self.validation_step(features)
                    validation_loss.append(loss.item())
                print("Validation loss:", np.mean(validation_loss))

                if (self.step // len(self.trainset))%10==0:
                    vector_medians_val = self.validation_inference(mus,vector_medians)
                    # vector_medians_train = self.validation_inference(trainset,vector_medians)
                
    def train_step (self,features):

        # for param in self.model.parameters():
        #     param.grad = None

        mixture = features["mix"].to(self.device)
        target = features["vocals"].to(self.device)


        training_data = mixture
        target_obj = target
        predicted = self.model(training_data).to(self.device)
        loss = self.loss_fn(target_obj, predicted.squeeze(1))

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        self.grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.params.max_grad_norm or 1e9)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return loss, self.grad_norm
    

    def validation_step(self,features):

        # for param in self.model.parameters():
        #     param.grad = None

        mixture = features["mix"].to(self.device)
        target = features["vocals"].to(self.device)

        training_data = mixture
        target_obj = target
        predicted = self.model(training_data).to(self.device)
        loss = self.loss_fn(target_obj, predicted.squeeze(1))

        return loss
    
    def validation_inference(self,mus,vector_medians):
        print('Validating')

        entire_med_sdr_voc = []
        c=0


        for count,filename in tqdm(enumerate(mus)):
            print(filename, 'Filename')
            mixture = load_and_resample(filename)
            vocals = load_and_resample(filename.replace("mixture.wav", "vocals.wav"))

            output_voice = []
            trim_low = 0
            duration = int(trim_low + (15*22050)) // self.params.n_hop * self.params.n_hop
            sec = math.ceil(mixture.shape[0] / 22050)

            for trim in tqdm(np.arange(math.ceil(sec / 15))):
                trim_high = trim_low + duration
                if trim_high > mixture.shape[0]:
                    #print("last one")
                    if (mixture.shape[0] // self.params.n_hop * self.params.n_hop) - trim_low > self.params.n_hop:
                        trim_high = mixture.shape[0] // self.params.n_hop * self.params.n_hop
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
                    mix_mag, mix_phase = compute_stft(mixture_analyse[None], self.params)
                    #print(mix_mag.shape,'mix_magnitude')
                    new_len = prev_power_of_2(mix_mag.shape[2])
                    #mix_mag=torch.transpose(mix_mag,1,2)
                    mix_mag = mix_mag[:, :, :new_len]
                    mix_phase = mix_phase[:, :, :new_len]
                    mix_mag=mix_mag.to("cuda")
                    mix_phase=mix_phase.to("cuda")
                    
                    #mix_mag=mix_mag.unsqueeze(1) #TODO: he añadido un unsqueeze porque sino peta el predict. 
                    diff_res = self.model(mix_mag)

                    output_signal = diff_res[:, :mix_mag.shape[1], :]
                    #print(output_signal.shape,'shape stft')
                    #print(mix_phase.shape,'shape de la fase')
                    #save_stft_image(mix_mag,f"stfts3/{c}.png")
                    #save_stft_image(output_signal,f"stfts2/{c}.png")
                    output_signal = compute_signal_from_stft2(output_signal, mix_phase, self.params).to("cpu").detach().numpy()
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
            c=c+1
            epoch = self.step // len(self.trainset)
            soundfile.write(f"audios2/audio{epoch}_{c}.wav", output_voice, 22050)
            estimates = np.array([output_voice])[..., None]

            scores = museval.evaluate(
                np.array([voc_ref])[..., None], estimates, win=22050, hop=22050)
            
            voc_sdr = scores[0][0]
            voc_sdr = np.round(np.median(voc_sdr[~np.isnan(voc_sdr)]), 3)

            print("VOCALS ==> SDR:", voc_sdr) #, " SIR:", voc_sir, " SAR:", voc_sar)
            entire_med_sdr_voc.append(voc_sdr)

        median_sdr_voc=np.median(entire_med_sdr_voc)
        print('All median SDR for vocals:',median_sdr_voc)
        
        previous_maximum=max(self.vector_medians)
        print(previous_maximum,'previous maximum')

        if median_sdr_voc>previous_maximum:
            self.best_SDR=median_sdr_voc
            #model=self.model_dir
            self.save_to_checkpoint("weights_bestSDR")
            self.best_epoch=self.step//len(self.trainset)
            print(f'I have saved a new model with the best SDR. It corresponds to epoch {self.best_epoch} with {self.best_SDR} of SDR')
        
        print(f'The best model corresponds to epoch {self.best_epoch}')
        self.vector_medians.append(median_sdr_voc)
        print(self.vector_medians)

        with open('./results2/SDR.txt', 'w') as file:
            file.write(str(self.vector_medians))

        return vector_medians

def train(args, params):
    trainset, testset = from_path(args.data_dir, params)  ## Create trainset and testset from the dataset class.
    print(trainset)
    model = UNET().to("cuda")#UNetV0(dim=2, in_channels=8,out_channels=1, channels=[1,2,4,8,16,32,64], factors=[2]*7, items=[2]*7).to("cuda")#UNetModel().to("cuda")   
    
    _train_impl(model, trainset, testset, args, params)


def _train_impl(model, trainset, testset, args, params):
    torch.backends.cudnn.benchmark = True ## For faster execution times. 
    opt = torch.optim.Adam(model.parameters(), lr=params.learning_rate) ## Stochastic Gradient Descent

    
    learner = Learner(args.model_dir, model, trainset, testset, opt,params,fp16=args.fp16)
    learner.restore_from_checkpoint()
    learner.train(max_steps=args.max_steps)  ## Call to the train method of the Learner to start training process. 
