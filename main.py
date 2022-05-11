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



import argparse, textwrap
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import sys
import os
import numpy as np

import torch.utils.data as data
from model import Classfication_Model
from data_loader import SpeechCommandsDataset
from train import train

def main(args):

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    torch.cuda.set_device(0)
    args = parser.parse_args()

    save_path = os.path.join('checkpoint',args.name)
    os.makedirs(save_path, exist_ok=True)
    state_dict = torch.load(args.pt)
    cfg = convert_namespace_to_omegaconf(state_dict['args'])

    task = tasks.setup_task(cfg.task)
    w2v_encoder = task.build_model(cfg.model)

    logging.info(w2v_encoder)
    model_parameters = filter(lambda p: p.requires_grad, w2v_encoder.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    logging.info(params)

    CLASSES = 'vioence, nonvioence'.split(', ')

    logging.info(f"classes: {CLASSES}")
    criterion = torch.nn.CrossEntropyLoss().cuda()

    ##### Model creating
    
    model = Classfication_Model(task,n_class=len(CLASSES), cfg=cfg, state_dict=state_dict['model']).cuda()
    optimizer = torch.optim.Adam([
        {'params': model.w2v_encoder.parameters(), 'lr': 1e-5},
        #{'params': model.conv.parameters(), 'lr': 5e-4},
        {'params': model.decoder.parameters(), 'lr': 5e-4},
    ], weight_decay=1e-5)


    ############ dataloader

    batch_size = 2

    train_dataset = SpeechCommandsDataset(CLASSES, root=args.dataset, mode='training')
    validation_dataset = SpeechCommandsDataset(CLASSES, root=args.dataset, mode='validation')
    test_dataset = SpeechCommandsDataset(CLASSES, root=args.dataset, mode='test')

    def _collate_fn(samples):
    
        sub_samples = [s for s in samples if s["source"] is not None] 
        if len(sub_samples) == 0:
            return {}
        batch = validation_dataset.collater(samples) 
        batch['target'] = torch.LongTensor([s["target"] for s in sub_samples])
        
        return batch


    train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True,
                                collate_fn=_collate_fn, num_workers=4)
    validation_dataloader = data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, pin_memory=True,
                                collate_fn=_collate_fn, num_workers=4)
    test_dataloader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True,
                                collate_fn=_collate_fn, num_workers=4)



    ##### train
    epochs = 100
    save_epoch = 10

    train(model, criterion, optimizer, train_dataloader, validation_dataloader, epochs, save_epoch, save_path)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('pt', type = str, help='pretrained model path')
    parser.add_argument('dataset', type = str, help='dataset root directory')
    parser.add_argument('name', type = str, help='save model name')
    args = parser.parse_args()

    main(args)

# python main.py wav2vec_small.pt media2015 save