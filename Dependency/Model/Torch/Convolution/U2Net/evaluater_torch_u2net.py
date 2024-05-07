
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
from pytorch_lightning.callbacks import LearningRateMonitor
import torch.nn as nn

from .u2net_loss import muti_bce_loss_fusion
from . import U2Net


class Evaluater():
    def __init__(self, dataloader, checkpoint_manager_in, checkpoint_manager_out, metrics = {}, metrics_calculator = {}, tb_writer = None, model = None,
                eval_length = 20, fig_drawer = None,
                 device = "cuda", writer = None, model_args = {}):

        self.eval_length = eval_length
        self.checkpoint_manager_in = checkpoint_manager_in
        self.checkpoint_manager_out = checkpoint_manager_out
        self.checkpoint_folder = checkpoint_manager_out.checkpoint_folder
        self.checkpoint_step = os.path.join(self.checkpoint_folder, "eval_step.pth")
        self.dataloader = dataloader
        self.dataIterator = iter(dataloader)
        self.model_args = model_args.copy()
        self.device = device
        self.tb_writer = tb_writer
        self.eval_step_index = 0
        self.epoch = 0
        self.metrics = metrics.copy() # dict of name: value
        self.metrics_calculator = metrics_calculator.copy() # dict of name: func
        self.running_metrics = metrics.copy()
        
        self.fig_drawer = fig_drawer
        if model is None:
            self.__build__()
        else:
            self.model = model
        self.load_status()

    
    def __build__(self):
        model = U2Net.get_u2net(**self.model_args)
        model.to(self.device)

        self.model = model
    
    def save_status(self):
        checkpoint_dict = self.checkpoint_manager_out.get_save_paths_new(index = self.eval_step_index, metrics = self.running_metrics)
        for checkpoint_path in checkpoint_dict.values():
            save_dict = {
                        'model_state_dict': self.model.state_dict(),
                        'eval_epoch': self.epoch,
                        'eval_step_index': self.eval_step_index,
                        }
            save_dict.update(self.metrics)
            torch.save(save_dict, checkpoint_path)
        
        torch.save({
                    'eval_epoch': self.epoch,
                    'eval_step_index': self.eval_step_index,
                    }, self.checkpoint_step)

    def load_status(self):
        checkpoint_path = self.checkpoint_manager_in.get_save_paths()["last_period"]
        if checkpoint_path is None or not os.path.exists(checkpoint_path):
            return
        model_checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(model_checkpoint['model_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if not os.path.exists(self.checkpoint_step):
            torch.save({
                    'eval_epoch': self.epoch,
                    'eval_step_index': self.eval_step_index,
                    }, self.checkpoint_step)
        try:
            epoch = model_checkpoint['eval_epoch']
            eval_step_index = model_checkpoint['eval_step_index']
        except:
            step_checkpoint = torch.load(self.checkpoint_step)
            self.epoch = step_checkpoint['eval_epoch']
            self.eval_step_index = step_checkpoint['eval_step_index']
        # val_step = checkpoint['val_step']

    def infer(self, inputs):
        inputs = inputs.type(torch.FloatTensor)
        labels = labels.type(torch.FloatTensor)
        
        # wrap them in Variable
        if self.device == "cuda":
            if torch.cuda.is_available():
                inputs_v, labels_v = Variable(inputs.cuda(), requires_grad=False), Variable(labels.cuda(),
                                                                                            requires_grad=False)
            else:
                inputs_v, labels_v = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)

        d0, d1, d2, d3, d4, d5, d6 = self.model(inputs_v)
        return d0, d1, d2, d3, d4, d5, d6
        
    def eval_step(self, data, eval_step_index, plot_image = False):

        self.model.eval()
        with torch.no_grad():
            # inputs, labels, usable = data
            inputs, labels = data
            # if any(usable) < 0.:
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

            d0, d1, d2, d3, d4, d5, d6 = self.model(inputs_v)
            loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v)
            for metric_name, calculator in self.metrics_calculator.items():
                self.metrics[metric_name] = calculator(labels_v, d0)
            # running_vloss += loss.data.item()
            loss_value = loss.data.item()
            self.metrics["loss"] = loss_value
            if plot_image and self.tb_writer is not None and self.fig_drawer is not None:
                self.tb_writer.add_figure('Predictions',
                    self.fig_drawer(inputs[0].cpu().detach().numpy(), labels[0].cpu().detach().numpy(), d0[0].cpu().detach().numpy()),
                    global_step=self.epoch, close = True)

            del d0, d1, d2, d3, d4, d5, d6, loss2, loss
            
        # return loss_value

    def test_infer(self):
        image_index = np.random.randint(0, min(self.eval_length, len(self.dataloader)))
        data = next(self.dataIterator)

        inputs, labels = data
        # if any(usable) < 0.:
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

        d0, d1, d2, d3, d4, d5, d6 = self.model(inputs_v)
        loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v)
        metrics = {}
        for metric_name, calculator in self.metrics_calculator.items():
            metrics[metric_name] = calculator(labels_v, d0)
        loss_value = loss.data.item()
        metrics["loss"] = loss_value
        # running_vloss += loss.data.item()

        figure = self.fig_drawer(inputs[0].cpu().detach().numpy(), labels[0].cpu().detach().numpy(), d0[0].cpu().detach().numpy())
        
        return inputs, labels, d0, metrics, figure

            
    def eval(self):
        self.running_metrics = dict(zip(self.metrics.keys(), [0.]* len(self.metrics.keys())))
        plot_image_index = np.random.randint(0, min(self.eval_length, len(self.dataloader)))
        for idx, data in enumerate(self.dataloader):
            self.eval_step_index += 1
            plot_image = (plot_image_index == idx)
            self.eval_step(data, self.eval_step_index, plot_image = plot_image)
            if self.tb_writer is not None:
                for metric_name, value in self.metrics.items():
                    self.tb_writer.add_scalars(f'Validation/{metric_name}',
                        {metric_name: value},
                        self.eval_step_index)
                    
            for key in self.running_metrics.keys():
                self.running_metrics[key] += self.metrics[key]
            
            if idx >= self.eval_length - 1:
                break
                
        for key in self.running_metrics.keys():
            self.running_metrics[key] = self.running_metrics[key]/self.eval_length
        # self.metrics["loss"] = avg_vloss 
        self.epoch += 1



            
        self.save_status()

    def execute(self):
        self.eval()