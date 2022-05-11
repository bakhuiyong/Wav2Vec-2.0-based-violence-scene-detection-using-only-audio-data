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

import torch.nn as nn
import torch.nn.functional as F
class Classfication_Model(nn.Module):
    def __init__(self, task, n_class=2, encoder_hidden_dim=768, cfg=None, state_dict=None):
        super(Classfication_Model, self).__init__()
        self.n_class = n_class
        assert not cfg is None
        assert not state_dict is None
        
        self.w2v_encoder = task.build_model(cfg.model)
        self.w2v_encoder.load_state_dict(state_dict)
        
        
        self.hidden_dim = 48 
        self.num_layers = 2
        self.output_dim = 2
        self.dropout =  0.1


        self.lstm = nn.LSTM(720,self.hidden_dim, self.num_layers,batch_first=True,dropout=self.dropout) 
        self.linear = nn.Linear(5376, self.output_dim)

        out_channels = 112
        
        self.decoder = nn.Sequential(
            nn.Conv1d(49, out_channels, 25, dilation=2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, 1)
            
        ) 
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):

        output = self.w2v_encoder(**x, features_only=True)
        output= output['x']
        b,t,c = output.shape
        
        output = self.decoder(output)
        out, hidden = self.lstm(output)
        out = out.reshape(-1,out.size(1)*out.size(2))
        
        out = self.linear(out)
        out = F.log_softmax(out, dim=1)

        return out

    @staticmethod
    def get_param_size(model):
        params = 0
        for p in model.parameters():
            tmp = 1
            for x in p.size():
                tmp *= x
            params += tmp
        return params