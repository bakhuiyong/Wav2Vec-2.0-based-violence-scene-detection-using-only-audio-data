from fairseq import models
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq import (
    checkpoint_utils,
    distributed_utils,
    options,
    quantization_utils,
    tasks,
    utils,
)

import os
import soundfile as sf
from fairseq.data.audio.raw_audio_dataset import * 
import torch.utils.data as data
from torch.utils.data.sampler import WeightedRandomSampler
import librosa
import numpy as np
import random

class SpeechCommandsDataset(RawAudioDataset):
    
    #sil=0.1, np=0.5, nl=0.7, sp=0.5, mp=0.5
    def __init__(self, CLASSES, mode='train', root=''):
        super(SpeechCommandsDataset, self).__init__(
            sample_rate=16000,
            pad=False
        ) 
        self.CLASSES = CLASSES
        self.mode = mode
        self.root = root
        self.mode_root = os.path.join(root,self.mode)
        self.sample_rate = 16000
        self.data = list()
        self.prep_dataset()
        self.y1 = 0
        self.x1 = 0

        if self.mode=='training':
            self.shift_prob = 0.5 
            self.mask_prob = 0.5 
            self.mask_len = 0.1   
        
            
    def prep_dataset(self): 
        
        self.id = 0
        for c in self.CLASSES: 
            for root, dir, files in os.walk(os.path.join(os.getcwd(),self.mode_root,c)):
                for file in files:
                    f_path, cmd = os.path.join(root, file), c 
                    self.data.append((f_path, cmd, self.id))
                    self.id += 1
                    #print(f"{self.mode} data number: {len(self.data)}")
    
    def __getitem__(self, idx): 
        f_path, cmd, id = self.data[idx]
        wav, curr_sample_rate = sf.read(f_path)
        try :
                self.x1, self.y1 = wav.shape
        except:
                self.x1 = wav.shape

        if curr_sample_rate!=self.sample_rate: 
            wav, curr_sample_rate = librosa.resample(wav, curr_sample_rate, self.sample_rate), self.sample_rate
            
        if self.y1==2:
            wav = librosa.to_mono(wav.transpose(1,0)) 
        self.y1 = 0
        wav_len = len(wav)
        if wav_len < self.sample_rate:
            pad_size = self.sample_rate - wav_len
            wav = np.pad(wav, (round(pad_size/2)+1,round(pad_size/2)+1), 'constant', constant_values=0)

        wav_len = len(wav)
        mid = int(len(wav)/2)
        cut_off = int(self.sample_rate/2)
        wav = wav[mid-cut_off:mid+cut_off] 
        

        if self.mode=='training': 
            if random.random()<self.mask_prob:
                t = int(self.mask_len*self.sample_rate)
                t0 = random.randint(0, self.sample_rate - t)
                wav[t0:t+t0] = 0
        
        feats = torch.from_numpy(wav).float()
        y = self.CLASSES.index(cmd)
        return {"id": id, "target": y, "source": feats}
    
    def __len__(self):
        return len(self.data)

