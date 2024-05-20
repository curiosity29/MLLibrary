
import numpy as np
import matplotlib.pyplot as plt
# import tensorflow.keras.backend as K
# from skimage.segmentation import mark_boundaries
import os, glob, sys, importlib
from functools import partial
from tqdm import tqdm
from datetime import datetime

import torch.nn as nn
import torch
from torch.autograd import Variable
from torchsummary import summary
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from .u2net_loss import get_loss
import torch.nn as nn
import time
from . import U2Net
# import U2Net

class Trainer():
    def __init__(self, dataloader, checkpoint_manager, train_length = 100, 
                 device = "cuda", model_args = {}, tb_writer = None,
                 metrics = {"loss": np.inf}, augmenter = None, model = None,
                 augmentation = None, metrics_calculator = {},
                clip_grad = False, loss_name = "bce", loss_args = {}, 
                optimizer_args = {},
                lr_scheduler_args = {}, 
                do_measure_time = False,
                pretrain_path = None, ## restart with this checkpoint
                verbose = 0):
        self.checkpoint_manager = checkpoint_manager
        self.train_length = train_length
        self.loss = get_loss(base_loss = loss_name, loss_args = loss_args, device = device)
        self.dataloader = dataloader
        self.epoch = 0
        self.train_step_index = 0
        self.clip_grad = clip_grad
        self.model_args = model_args.copy()
        self.device = device
        self.tb_writer = tb_writer
        self.metrics = metrics.copy()
        self.metrics_calculator = metrics_calculator.copy()
        self.optimizer_args = optimizer_args.copy()
        self.lr_scheduler_args = lr_scheduler_args.copy()
        self.running_metrics = metrics.copy()
        self.verbose = verbose
        self.do_measure_time = do_measure_time
        
        self.model = model
        if augmentation is None:
            self.augmentation = lambda *x: x
        else:
            self.augmentation = augmentation

        # self.current_time = time.now()
        self.subprocess_name = ["save", "load", "create data", "augment", "predict", "tensorboard"]
        self.subprocess_time = dict(zip(self.subprocess_name, [0.] * len(self.subprocess_name)))
        
        
        self.__build__()

        
        if pretrain_path is not None:
            self.load_checkpoint(pretrain_path)

        
    def __build__(self):
        if self.model is None:
            self.model = U2Net.get_u2net(**self.model_args)

        self.model.to(self.device)
        optimizer = optim.Adam(self.model.parameters(), **self.optimizer_args)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **self.lr_scheduler_args)
        
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.load_status()

    def measure_time(func):
        def wrapper(*args, **kwargs):
            self = args[0]
            if self.do_measure_time:
                start_time = time.time()
                result = func(*args, **kwargs)
                end_time = time.time()
                elapsed_time = end_time - start_time
                print(f"Function '{func.__name__}' took {elapsed_time:.6f} seconds to execute.")
                return result
            else:
                return func(*args, **kwargs)
        return wrapper
    
    # @measure_time
    def save_status(self):
        checkpoint_dict = self.checkpoint_manager.get_save_paths_new(index = self.train_step_index, metrics = self.running_metrics)
        save_dict = {
                    'train_epoch': self.epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'lr_state_dict': self.lr_scheduler.state_dict(),
                    'train_step_index': self.train_step_index,
                    }
        save_dict.update(self.metrics)
        for checkpoint_path in checkpoint_dict.values():
            torch.save(save_dict, checkpoint_path)

    def load_checkpoint(self, checkpoint_path):
        model_checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(model_checkpoint['model_state_dict'])
        self.optimizer = optim.Adam(self.model.parameters(), **self.optimizer_args)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, **self.lr_scheduler_args)
        # return self.model
    # @measure_time
    def load_status(self):
        checkpoint_path = self.checkpoint_manager.get_save_paths()["last_period"]
        if checkpoint_path is None or not os.path.exists(checkpoint_path):
            return
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        # self.lr_scheduler.load_state_dict(checkpoint["lr_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['train_epoch']
        self.train_step_index = checkpoint['train_step_index']
        # val_step = checkpoint['val_step']
        


    # @measure_time
    def train_step(self, data, train_step_index):

        ### 
        self.model.train()
        # inputs, labels, usable = data
        inputs, labels = data
        inputs, labels = self.augmentation(inputs, labels)
        # if any(usable < 0.):
        #     continue

        inputs = inputs.type(torch.FloatTensor)
        labels = labels.type(torch.FloatTensor)

        # wrap them in Variable
        if self.device == "cuda":
            if torch.cuda.is_available():
                inputs_v, labels_v = Variable(inputs.cuda(), requires_grad=False), Variable(labels.cuda(),
                                                                                        requires_grad=False)
        else:
            inputs_v, labels_v = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)

        # y zero the parameter gradients
        self.optimizer.zero_grad()

        d0, d1, d2, d3, d4, d5, d6 = self.model(inputs_v)
        loss2, loss = self.loss(d0, d1, d2, d3, d4, d5, d6, labels_v)

        loss.backward()
        if self.clip_grad:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm = 'inf', norm_type = 1.)
        self.optimizer.step()
        self.lr_scheduler.step()

        # # print statistics
        loss_value = loss.data.item()
        loss_tar =  loss2.data.item()
        # running_loss += loss_value
        # running_tar_loss += loss_tar
        self.metrics["loss"] = loss_value

        with torch.no_grad():
            for metric_name, calculator in self.metrics_calculator.items():
                    self.metrics[metric_name] = calculator(labels_v, d0)
                
        # del temporary outputs and loss
        del d0, d1, d2, d3, d4, d5, d6, loss2, loss


    # def train_step_maker(self):
        
    def log_activation_map(self):


        return 

    def verbose_print(self, message, level = 0):
        if level <= self.verbose:
            print(message)
        
    # @measure_time
    def train(self):
        # self.verbose_print(f"start training (step {self.train_step_index})", level = 2)
        
        self.running_metrics = dict(zip(self.metrics.keys(), [0.]* len(self.metrics.keys())))

        for idx, data in enumerate(self.dataloader):
            self.train_step(data, self.train_step_index)
            self.train_step_index += 1
            
            for key in self.running_metrics.keys():
                self.running_metrics[key] += self.metrics[key]
            
            if self.tb_writer is not None:
                for metric_name, value in self.metrics.items():
                    self.tb_writer.add_scalars(f'Training/{metric_name}',
                            {metric_name: value},
                            self.train_step_index)
                self.tb_writer.add_scalars('learning_rate',
                    {'learning_rate' : self.optimizer.param_groups[0]["lr"] },
                self.train_step_index)
            if idx >= self.train_length - 1:
                break
                
        for key in self.running_metrics.keys():
            self.running_metrics[key] = self.running_metrics[key]/ idx #self.train_length

        ## log weight
        if self.tb_writer is not None:
            for name, param in self.model.named_parameters():
                if "weight" in name:  # Log only weight parameters
                    self.tb_writer.add_histogram(name, param, global_step=self.train_step_index)

        self.epoch += 1
        
        self.verbose_print(f"start training (step {self.train_step_index})", level = 2)


    def execute(self, num_loop: int = 1):
        for _ in range(num_loop):
            self.train()
        self.save_status()

    