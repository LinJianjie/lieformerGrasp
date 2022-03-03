import sys
import time
from abc import abstractmethod
from pathlib import Path

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[2]
sys.path.append(str(root))
from pytorch_utils.components.Logger import *
import torch.nn as nn
import torch.optim as optim
import torch
import os
from collections import OrderedDict


class TorchLightModule(nn.Module):
    def __init__(self, config, checkpoint_name="check_points.pth",
                 best_name="best_name.pth",
                 checkpoint_path=".",
                 logger_file_name="log"):
        """
        :rtype: object
        """
        super(TorchLightModule, self).__init__()
        self.device = None
        self.model_optimizer = None
        self.checkpoint_name = checkpoint_name
        self.checkpoint_path = checkpoint_path
        self.best_name = best_name
        self.config = config
        self.model = None
        self.time_str = time.strftime("%Y%m%d_%H%M%S")
        logger_file_name = logger_file_name + "_" + self.time_str + ".log"
        self.Logger = Logger(filename=logger_file_name)
        self.state = None

    def forward(self, *args):
        pass

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3, betas=(0.9, 0.999))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.config["epochs"],
                                                               eta_min=1e-3)
        opt_scheduler = OrderedDict({"opt": optimizer, "scheduler": scheduler})
        self.model_optimizer = opt_scheduler
        return opt_scheduler

    def toCuda(self, device):
        self.device = device
        if torch.cuda.device_count() > 1:
            print("====> use data parallel")
            self.model = nn.DataParallel(self.model)
            self.model.to(device)
        else:
            print("====> use only one cuda")
            self.model.to(device)

    def save_checkpoint(self, best_model_name):
        best_model = self.time_str + "_" + best_model_name
        print("--> best_model: ", best_model)
        self.best_name = best_model
        save_path = os.path.join(self.checkpoint_path, best_model)
        torch.save(self.state, save_path)

    @abstractmethod
    def train_one_epoch(self, train_loader, at_epoch, n_epochs, batch_size):
        pass

    # @abstractmethod
    # def train_step(self, start_epoch, n_epochs, train_loader, test_loader, best_loss=0.0, batch_size=8,
    #                best_model_name="best_model.pth"):
    #     pass
    @abstractmethod
    def compute_step(self, *args):
        pass

    @abstractmethod
    def evaluation_step(self, test_loader, check_point_name=None):
        pass

    @abstractmethod
    def backward_model(self, *args):
        pass

    @abstractmethod
    def backward_model_update(self, loss):
        pass

    @staticmethod
    def configure_loss_function():
        pass

    def count_parameters(self):
        number_of_parameter = sum(p.numel() for p in self.parameters() if p.requires_grad)
        # self.Logger.INFO("the number of parameter: %.2f M", number_of_parameter / 1e6)
        print("the number of parameter: %5.2f M" % (number_of_parameter / 1e6))

    def state_to_save(self, state=None):
        if state is None:
            self.state = {
                "model_state": self.model.state_dict(),
                "optimizer_state": self.model_optimizer["opt"].state_dict()
            }
        else:
            self.state = state

    def load_checkpoint(self, filename="checkpoint"):
        if os.path.isfile(filename):
            print("==> Loading from checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)
            if checkpoint["model_state"] is not None:
                self.model.load_state_dict(checkpoint["model_state"])
            if checkpoint["optimizer_state"] is not None:
                self.model_optimizer["opt"].load_state_dict(checkpoint["optimizer_state"])
            print("==> Done")
        else:
            print("==> Checkpoint '{}' not found".format(filename))
