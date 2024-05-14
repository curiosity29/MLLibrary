from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
import configparser
import json
from Dependency.Utility.Dataset.Dataset import FinalDataset
from Dependency.Utility.Dataset.Extractor import RawDataset
# from DependenData import RawDataset, FinalDataset

from Dependency.Model.Torch.Convolution.U2Net.trainer_torch_u2net import Trainer
from Dependency.Model.Torch.Convolution.U2Net.evaluater_torch_u2net import Evaluater

# from Dependency.Model.Torch.Convolution.Simple.trainer_ff import Trainer
# from Dependency.Model.Torch.Convolution.Simple.evaluater_ff import Evaluater

from Dependency.Utility.Process.process import Process_command
from Dependency.Utility.Callbacks.checkpoint_manager import Checkpoint_manager
from Dependency.Utility.Inference.Inferer import Inferer

from Dependency.CD_Sen2.data import get_image_preprocess, get_label_preprocess ## clip by quantile and one_hot
from Dependency.CD_Sen2.Visual import plot_prediction as fig_drawer
from Dependency.CD_Sen2.Runner import get_metrics_calculator, get_loss_calculator

from Dependency.CD_Sen2.Inference import get_predictor
from Dependency.CD_Sen2.augmenter import get_augmentation
class Manager():
    def __init__(self, configs_file, restart_raw_data = False, restart_train_data = False, restart_eval_data = False, restart_checkpoint = False, restart_runner = False,
                 model = None,
                setup = False, backup = True):
        self.configs_file = configs_file
        self.model = model

        
        self.reload(
            restart_raw_data = restart_raw_data, 
            restart_train_data = restart_train_data, 
            restart_eval_data = restart_eval_data,
            restart_checkpoint = restart_checkpoint, 
            restart_runner = restart_runner,
            do_backup = backup,
        )
        self.setup = setup
        
    def reload(self, restart_raw_data = False, restart_train_data = False, restart_eval_data = False, restart_checkpoint = False, restart_runner = False,
              do_backup = True):

        with open(self.configs_file) as src:
            configs = json.load(src)



        ### Raw data
        raw_data_configs = configs["Raw_data"]
        raw_image_folder = raw_data_configs["image_folder"]
        raw_label_folder = raw_data_configs["label_folder"]
        raw_data_status_file = raw_data_configs["status_file"]
        raw_data_meta_file = raw_data_configs["meta_data_file"]
        raw_data_preprocess_mode = raw_data_configs["preprocess_mode"]
        raw_data_preprocess_stats_file = raw_data_configs["preprocess_stats_file"]
        raw_data_dynamic_preprocess = (raw_data_preprocess_mode == "dynamic")
        raw_data_step_divide = raw_data_configs["step_divide"]

        ### Data
        data_configs = configs["Data"]
        train_image_folder = data_configs["train_image_folder"]
        train_label_folder = data_configs["train_label_folder"]
        eval_image_folder = data_configs["eval_image_folder"]
        eval_label_folder = data_configs["eval_label_folder"]
        split_ratio = data_configs["split_ratio"]
        label_filter_threshold = data_configs["filter_threshold"]
        
        command_file = data_configs["command_file"]
        train_data_status_file = data_configs["train_status_file"]
        eval_data_status_file = data_configs["eval_status_file"]

        ### Runner
        runner_configs = configs["Runner"]
        train_length = runner_configs["train_length"]
        eval_length = runner_configs["eval_length"]
        
        num_workers = runner_configs["num_workers"]
        batch_size_train = runner_configs["batch_size_train"]
        batch_size_eval = runner_configs["batch_size_eval"]
        clip_grads = runner_configs["clip_grads"]
        loss_name = runner_configs["loss_name"]
        
        device =  runner_configs["device"]

        ### Model
        model_configs = configs["Model"]
        window_size = model_configs["window_size"]
        n_channel = model_configs["n_channel"]
        n_class = model_configs["n_class"]
        # mid_ch = model_configs["mid_ch"]
        loss_args = model_configs["loss_args"]
        optimizer_args = model_configs["optimizer_args"]
        lr_scheduler_args = model_configs["lr_scheduler_args"]
        model_version = model_configs["version"]
        sigmoid_head = model_configs["sigmoid_head"]
        model_args = dict(
            in_ch = n_channel,
            out_ch = n_class,
            # mid_ch = mid_ch,
            version = model_version,
            sigmoid_head = sigmoid_head,
        )

        ## Data augmentation

        augment_configs = configs["Augmentation"]
        aug_cloud_prob = augment_configs["cloud_prob"]
        aug_noise_std = augment_configs["noise"]
        aug_flip = augment_configs["flip"]



        ### Train_checkpoint
        checkpoint_configs = configs["Train_checkpoint"]
        train_checkpoint_status_file = checkpoint_configs["status_file"]
        train_checkpoint_period = checkpoint_configs["period"]
        train_metrics = checkpoint_configs["metrics"]
        train_metrics_init = checkpoint_configs["metrics_init"]
        train_metrics_target = checkpoint_configs["metrics_target"]
        train_metrics_target = dict(zip(train_metrics, train_metrics_target))
        train_metrics = dict(zip(train_metrics, train_metrics_init))
        
        train_checkpoint_extension = checkpoint_configs["checkpoint_extension"]
        train_time_format = checkpoint_configs["time_format"]
        train_max_to_keep = checkpoint_configs["max_to_keep"]
        train_checkpoint_root_folder = checkpoint_configs["root_folder"]

        ### Eval_checkpoint
        checkpoint_configs = configs["Eval_checkpoint"]
        eval_checkpoint_status_file = checkpoint_configs["status_file"]
        eval_checkpoint_period = checkpoint_configs["period"]
        eval_metrics = checkpoint_configs["metrics"]
        eval_metrics_init = checkpoint_configs["metrics_init"]
        eval_metrics_target = checkpoint_configs["metrics_target"]
        eval_metrics_target = dict(zip(eval_metrics, eval_metrics_target))
        eval_metrics = dict(zip(eval_metrics, eval_metrics_init))


        
        eval_checkpoint_extension = checkpoint_configs["checkpoint_extension"]
        eval_time_format = checkpoint_configs["time_format"]
        eval_max_to_keep = checkpoint_configs["max_to_keep"]
        eval_checkpoint_root_folder = checkpoint_configs["root_folder"]

        ### Inference
        infer_configs = configs["Inference"]
        infer_mode = infer_configs["mode"]

        ## func config

        image_preprocess = get_image_preprocess(stats_file = raw_data_preprocess_stats_file, mode = raw_data_preprocess_mode) # channel last preprocess
        label_preprocess = get_label_preprocess(n_class = n_class)
        metrics_calculator = get_metrics_calculator(n_class = n_class)
        # loss_calculator = get_loss_calculator()
        loss_calculator = {}
        augmentation = get_augmentation(
            cloud_prob = aug_cloud_prob,
            noise_std = aug_noise_std,
            flip = aug_flip,
        )

        ### Tensorboard
        tb_configs = configs["Tensorboard"]
        logdir = tb_configs["logdir"]
        # train_scalar = tb_configs["train_scalar"]
        # eval_scalar = tb_configs["eval_scalar"]
        tb_writer = SummaryWriter(os.path.join(logdir, "scalar"))
        
        raw_dataset = RawDataset(
            input_image_folder = raw_image_folder, 
            input_label_folder = raw_label_folder, 
            output_image_folders = [train_image_folder, eval_image_folder], 
            output_label_folders = [train_label_folder, eval_label_folder],
            ratio_list = split_ratio,
            window_size = window_size, 
            channel_last = False, 
            status_file = raw_data_status_file, 
            command_file = command_file,
            item_meta_file = raw_data_meta_file,
            restart_all = restart_raw_data,
            image_preprocess = image_preprocess,
            label_preprocess = label_preprocess,
            dynamic_preprocess = raw_data_dynamic_preprocess,
            preprocess_info_file = raw_data_preprocess_stats_file,
            auto = False,
            filter_threshold = label_filter_threshold,
            step_divide = raw_data_step_divide,
        )
        
        train_dataset = FinalDataset(
            name = "Train_data",
            image_folder = train_image_folder, 
            label_folder = train_label_folder, 
            window_size = window_size, 
            channel_last = False, 
            status_file = train_data_status_file, 
            command_file = command_file,
            command_section = "Train_data",
            restart_all = restart_train_data,
        )
        train_dataloader = DataLoader(train_dataset, batch_size = batch_size_train, shuffle = False, num_workers = num_workers)
        
        
        eval_dataset = FinalDataset(
            name = "Eval_data",
            image_folder = eval_image_folder, 
            label_folder = eval_label_folder, 
            window_size = window_size, 
            channel_last = False, 
            status_file = eval_data_status_file, 
            command_file = command_file,
            command_section = "Eval_data",
            restart_all = restart_eval_data,
        )
        eval_dataloader = DataLoader(eval_dataset, batch_size = batch_size_eval, shuffle = False, num_workers = num_workers)
        
        process = Process_command(command_file = command_file, section_name = "Process")



        train_checkpoint_manager = Checkpoint_manager(
            name = "train",
            status_file = train_checkpoint_status_file, 
            period = train_checkpoint_period,
            metrics = train_metrics, 
            metrics_target = train_metrics_target,
            checkpoint_extension = train_checkpoint_extension, 
            time_format = train_time_format, 
            max_to_keep = train_max_to_keep,
            clean_file = restart_checkpoint,
            root_folder = train_checkpoint_root_folder,
            backup = do_backup,
        )


        eval_checkpoint_manager = Checkpoint_manager(
            name = "eval",
            status_file = eval_checkpoint_status_file, 
            period = eval_checkpoint_period,
            metrics = eval_metrics, 
            metrics_target = eval_metrics_target,
            checkpoint_extension = eval_checkpoint_extension, 
            time_format = eval_time_format, 
            max_to_keep = eval_max_to_keep,
            clean_file = restart_checkpoint,
            root_folder = eval_checkpoint_root_folder,
            backup = do_backup,
        )

        trainer = Trainer(
        checkpoint_manager = train_checkpoint_manager,
        dataloader = train_dataloader, 
        train_length = train_length,
        model_args = model_args,
        device = device,
        clip_grad = clip_grads,
        metrics = train_metrics,
        tb_writer = tb_writer,
        model = self.model,
        augmentation = augmentation,
        metrics_calculator = loss_calculator,
        loss_name = loss_name,
        loss_args = loss_args,
        optimizer_args = optimizer_args,
        lr_scheduler_args = lr_scheduler_args,
        verbose = 0,
        )
    
        evaluater = Evaluater(
        checkpoint_manager_in = train_checkpoint_manager,
        checkpoint_manager_out = eval_checkpoint_manager,
        dataloader = eval_dataloader, 
        eval_length = eval_length,
        model_args = model_args,
        device = device,
        metrics = eval_metrics,
        tb_writer = tb_writer,
        fig_drawer = fig_drawer,
        metrics_calculator = metrics_calculator,
        model = trainer.model,
        loss_name = loss_name,
        loss_args = loss_args,
        verbose = 0,
        )

        predictor = get_predictor(model = evaluater.model, mode = infer_mode)
        
        inferer = Inferer(
            image_folder = raw_image_folder, 
            prediction_folder = './Predictions', 
            predictor = predictor, 
            preprocess = "auto",
            window_size = window_size, 
            input_dim = n_channel, 
            predict_dim = n_class, 
            output_type = "float32", 
            batch_size = 1, 
            step_divide = 2, 
            kernel = None
        )
        

        
        self.command_file = command_file

        self.raw_dataset = raw_dataset
        self.train_dataset = train_dataset
        self.train_dataloader = train_dataloader
        self.eval_dataset = eval_dataset
        self.eval_dataloader = eval_dataloader
        self.process = process

        self.trainer = trainer
        self.evaluater = evaluater

        self.inferer = inferer

    def execute(self):
        with open(self.command_file) as src:
            command = json.load(src)
        match command["Manager"]["mode"]:
            case "reload":
                restart_raw_data = command["Raw_data"]["restart"]
                restart_train_data = command["Train_data"]["restart"]
                restart_eval_data = command["Eval_data"]["restart"]
                restart_checkpoint = command["Checkpoint"]["restart"]
                restart_runner = command["Runner"]["restart"]
                self.reload(
                    restart_raw_data = restart_raw_data, 
                    restart_train_data = restart_train_data,
                    restart_eval_data = restart_eval_data,
                    restart_checkpoint = restart_checkpoint,
                    restart_runner = restart_runner,
                )
                command["Raw_data"]["restart"] = False
                command["Train_data"]["restart"] = False
                command["Eval_data"]["restart"] = False
                command["Checkpoint"]["restart"] = False
                command["Runner"]["restart"] = False
                command["Manager"]["mode"] = False
                
                with open(self.command_file, "w") as dest:
                    json.dump(command, dest, indent = 4)
            # case ""
        
    
