import sys
from pathlib import Path

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[2]
sys.path.append(str(root))
from pytorch_utils.components.ioUtils import *
from pytorch_utils.components.pytorchLightModule import TorchLightModule
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='infoGan based point completion network  ')
    parser.add_argument('--num_points', type=int, default=2048, help="number of point cloud")
    parser.add_argument('--NN_config', type=str, default="infoGan.yaml", help="yaml file for Neural network")
    parser.add_argument('--batch_size', type=int, default=8, help="batch size")
    parser.add_argument('--num_works', type=int, default=16, metavar='num_workers', help='num of workers')
    parser.add_argument('--shuffle', type=lambda x: not (str(x).lower() == 'false'), default=True, help='Use shuffle')
    parser.add_argument('--drop_last', type=lambda x: not (str(x).lower() == 'false'), default=True,
                        help='Use drop_last')
    parser.add_argument("--lr", type=float, default=1e-3, help="lr")
    parser.add_argument("--cuda", type=str, default="0", help="which cuda used")
    parser.add_argument('--gpu', nargs='+', type=int, help='list of gpu ids to be used')
    parser.add_argument("--epochs", type=int, default=100, help="maximum epochs")
    parser.add_argument("--checkpoint_name", type=str, default="ckpt", help="checkpoint names")
    parser.add_argument('--load_checkPoints', type=lambda x: not (str(x).lower() == 'false'), default=False,
                        help='load_checkpoints')
    parser.add_argument('--best_name', type=str, default=False, help='best name')
    parser.add_argument('--log_filename', type=str, default="log", help="create_a_logfilename")
    parser.add_argument('--train', type=lambda x: not (str(x).lower() == 'false'), default=True,
                        help='train')
    parser.add_argument('--evaluate', type=lambda x: not (str(x).lower() == 'false'), default=False,
                        help='evaluate')

    return parser.parse_args()


class MainTrainer(object):
    def __init__(self, yaml_config):
        self.log_dir = None
        self.check_point_dir = None
        self.args = None
        self.train_loader = None
        self.evaluation_model = None
        self.train_model = None
        self.test_loader = None
        self.yaml_config = yaml_config
        self.create_check_points()
        self.create_log()

    def from_argparse_args(self, args=None):
        if args is not None:
            self.args = args
        else:
            self.args = parse_args()

    def get_config(self):
        return self.yaml_config.config

    def load_data(self, train_dataset: Dataset = None, test_dataset: Dataset = None):
        if train_dataset is not None:
            self.train_loader = DataLoader(dataset=train_dataset,
                                           num_workers=self.args.num_works,
                                           batch_size=self.args.batch_size,
                                           shuffle=self.args.shuffle,
                                           drop_last=self.args.drop_last,
                                           pin_memory=True)
        if test_dataset is not None:
            self.test_loader = DataLoader(dataset=test_dataset,
                                          num_workers=self.args.num_works,
                                          batch_size=1,
                                          shuffle=False,
                                          drop_last=self.args.drop_last,
                                          pin_memory=True)

    def create_log(self):
        self.log_dir = make_dirs_log()

    def create_check_points(self):
        self.check_point_dir = make_dirs_checkout()

    @property
    def model(self):
        return self.train_model

    def set_model(self, model: TorchLightModule, use_cuda=True):
        self.train_model = model
        if use_cuda:
            cuda_index = "cuda"
            device = torch.device(cuda_index if (torch.cuda.is_available()) else "cpu")
        else:
            device = torch.device("cpu")
        self.train_model.toCuda(device=device)

    def load_check_points(self, check_point_path):
        self.train_model.load_checkpoint(check_point_path)

    def fit(self):
        if self.train_loader is None and self.test_loader is None:
            print("No need to train")
        if self.train_loader is not None and self.test_loader is None:
            self.train_model.state_to_save()
            train_loss = 100000
            for at_epoch in range(0, self.args.epochs):
                _train_loss = self.train_model.train_one_epoch(train_loader=self.train_loader,
                                                               at_epoch=at_epoch,
                                                               n_epochs=self.args.epochs,
                                                               batch_size=self.args.batch_size)

                self.train_model.Logger.INFO('Train %d/%d, train_loss: %.6f, the best_train_loss: %.6f\n',
                                             at_epoch, self.args.epochs, _train_loss, train_loss)
                if _train_loss < train_loss:
                    self.train_model.state_to_save()
                    train_loss = _train_loss
            self.train_model.save_checkpoint(best_model_name=self.args.best_name)
        if self.train_loader is None and self.test_loader is not None:
            self.train_model.evaluation_step(test_loader=self.test_loader, check_point_name=None)
        if self.train_loader is not None and self.test_loader is not None:
            self.train_model.state_to_save()
            evaluation_loss = 100000
            for at_epoch in range(self.args.start_epoch, self.args.epochs):
                train_loss = self.train_model.train_one_epoch(train_loader=self.train_loader,
                                                              at_epoch=at_epoch,
                                                              n_epochs=self.args.epochs,
                                                              batch_size=self.args.batch_size)
                self.train_model.Logger.INFO('Train %d/%d, train_loss: %.6f\n', at_epoch, self.args.epochs,
                                             train_loss)

                _evaluation_loss = self.train_model.evaluation_step(test_loader=self.test_loader, check_point_name=None)
                self.train_model.Logger.INFO('Train %d/%d, evaluation_loss: %.6f\n', at_epoch, self.args.epochs,
                                             _evaluation_loss)
                if _evaluation_loss < evaluation_loss:
                    self.train_model.state_to_save()
                    evaluation_loss = _evaluation_loss
            self.train_model.save_checkpoint(best_model_name=self.args.best_name)

    def evaluate(self, val_data: DataLoader = None):
        if val_data is None:
            self.train_model.evaluation_step(test_loader=self.test_loader, check_point_name=None)
        else:
            self.train_model.evaluation_step(test_loader=val_data, check_point_name=None)

    def predict(self, test_loader: DataLoader = None, evaluation_prefix: str = None):
        if test_loader is None:
            if evaluation_prefix is None:
                self.train_model.predict(test_loader=self.test_loader)
            else:
                self.train_model.predict(test_loader=self.test_loader, evaluation_prefix=evaluation_prefix)

        else:
            if evaluation_prefix is None:
                self.train_model.predict(test_loader=test_loader)
            else:
                self.train_model.predict(test_loader=test_loader, evaluation_prefix=evaluation_prefix)
