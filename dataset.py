import numpy as np
import os
import random
import torch
import torch.nn.functional as F
import torchaudio

from glob import glob
from torch.utils.data.distributed import DistributedSampler
#from params import params


class Dataset(torch.utils.data.Dataset):
    def __init__(self,path, subset = "train"):
        self.all_files = glob(os.path.join(path,"*mixture.wav"))
        self.mix_files = []
        self.bass_files = []

        self.validation_files = ["20", "40", "90", "93"] ## Set of validation files of the MUSDB dataset that do inference while training. 
        self.subset = subset

        
        if self.subset == "train":  ## Esta parte sirve para después crear un objeto de la clase Unconditional Dataset con el subset="train" para obtener los datos de train en esa variable.
            self.mix_files = self.all_files.copy()
            print(len(self.mix_files),'all mix files')
            
            for track in self.validation_files: ## No acabo de entender bien por qué se recorren los validation files. 
                self.mix_files = [x for x in self.mix_files if track + '_' not in x]
                self.bass_files = [x.replace("mixture.wav", "bass.wav") for x in self.mix_files]
                self.accomp_files = [x.replace("mixture.wav", "accompaniment.wav") for x in self.mix_files]
            print(len(self.mix_files),'all mix files2')

        else:
            for track in self.validation_files:
                self.mix_files += [x for x in glob(os.path.join(path, '*mixture.wav')) \
                    if (track + '_' in x) and ('_' + track + '_' not in track)]
                self.mix_files = [x for x in self.mix_files if 'silence' not in x]
                self.bass_files = [x.replace("mixture.wav", "bass.wav") for x in self.mix_files]
                self.accomp_files = [x.replace("mixture.wav", "accompaniment.wav") for x in self.mix_files]
                
        print('Mix files',(len(self.mix_files)))
        print('BASS FILES',(len(self.bass_files)))
    def __len__(self):
        return len(self.mix_files)

    def __getitem__(self, idx):
        mix_filename = self.mix_files[idx]
        bass_filename = self.bass_files[idx]
        accomp_filename = self.accomp_files[idx]
        mix, _ = torchaudio.load(mix_filename)
        bass, _ = torchaudio.load(bass_filename)
        accomp, _ = torchaudio.load(accomp_filename)
        return {
            "mix": mix,
            "bass": bass,
            "accomp": accomp,
        }
    

class Collator:
    def __init__(self, params,validation=False):
        self.params = params
        self.validation=validation

    def collate(self, minibatch):
        samples_per_frame = self.params.n_hop
        mix_list=[]
        bass_list=[]
        acc_list=[]
        for record in minibatch:
            start = random.randint(0, record["mix"].shape[-1] - self.params.audio_len)
            end = start + self.params.audio_len
            if self.validation==False:
                record["mix"] = record["mix"].squeeze(0)[start:end]
                record["bass"] = record["bass"].squeeze(0)[start:end]
                record["accomp"] = record["accomp"].squeeze(0)[start:end]
            else:
                record["mix"] = record["mix"].squeeze(0)
                record["bass"] = record["bass"].squeeze(0)
                record["accomp"] = record["accomp"].squeeze(0)

            transform = torchaudio.transforms.Spectrogram(
                n_fft=self.params.n_fft,
                win_length=self.params.n_win,
                hop_length=self.params.n_hop,
                window_fn=torch.hann_window
            )

            mix_spec = self.check_shape_2d(
                self.check_shape_2d(
                    torch.abs(transform(record["mix"])), 0), 1)
            voc_spec = self.check_shape_2d(
                self.check_shape_2d(
                    torch.abs(transform(record["bass"])), 0), 1)
            acc_spec = self.check_shape_2d(
                self.check_shape_2d(
                    torch.abs(transform(record["accomp"])), 0), 1)
            mix_list.append(mix_spec)
            bass_list.append(voc_spec)
            acc_list.append(acc_spec) 
            
        mix_train = torch.stack([mix for mix in mix_list])
        vocal_train = torch.stack([bass for bass in bass_list])
        accomp_train = torch.stack([acc for acc in acc_list])

        return {
                "mix": mix_train,
                "bass": vocal_train,
                "accomp": accomp_train,
                "conditioning": None,
            }
        

    def check_shape_2d(self, data, dim):
        n = data.shape[dim]
        if n % 2 != 0:
            n = data.shape[dim] - 1
        if dim==0:
            return data[:n, :]
        if dim==1:
            return data[:, :n]


def from_path(data_dirs, params, is_distributed=False):
    
    train_dataset = Dataset(data_dirs, "train")
    test_dataset = Dataset(data_dirs, "test")
    
    return (
        torch.utils.data.DataLoader(
            train_dataset,
            batch_size=params.batch_size,
            collate_fn=Collator(params,validation=False).collate,
            shuffle=not is_distributed,
            num_workers=os.cpu_count(),
            sampler=None,
            pin_memory=True,
            drop_last=True),
        torch.utils.data.DataLoader(
            test_dataset,
            batch_size=1,
            collate_fn=Collator(params,validation=False).collate,
            shuffle=not is_distributed,
            num_workers=os.cpu_count(),
            sampler=None,
            pin_memory=True,
            drop_last=True),
    )