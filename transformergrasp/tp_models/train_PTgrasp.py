import os
import sys
from pathlib import Path

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))
from pytorch_utils.components.mainTrainer import MainTrainer
from pytorch_utils.components.nvidia_grasp_dataset import NvidiaGraspDataSet
from pytorch_utils.components.yaml_utils import YAMLConfig
from tp_models.PTnet_grasp import PTNetGraspModel as GraspModel

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='infoGan based point completion network  ')
    parser.add_argument('--num_points', type=int, default=2048, help="number of point cloud")
    parser.add_argument('--NN_config', type=str, default="infoGan.yaml", help="yaml file for Neural network")
    parser.add_argument('--batch_size', type=int, default=1, help="batch size")
    parser.add_argument('--num_works', type=int, default=16, metavar='num_workers', help='num of workers')
    parser.add_argument('--shuffle', type=lambda x: not (str(x).lower() == 'false'), default=True, help='Use shuffle')
    parser.add_argument('--drop_last', type=lambda x: not (str(x).lower() == 'false'), default=True,
                        help='Use drop_last')
    parser.add_argument("--lr", type=float, default=1e-3, help="lr")
    parser.add_argument("--cuda", type=str, default="0", help="which cuda used")
    parser.add_argument('--gpu', nargs='+', type=int, help='list of gpu ids to be used')
    parser.add_argument("--epochs", type=int, default=100, help="maximum epochs")
    parser.add_argument("--start_epoch", type=int, default=100, help="maximum epochs")
    parser.add_argument("--checkpoint_name", type=str, default="ckpt", help="checkpoint names")
    parser.add_argument("--yaml_file", type=str, default="ckpt", help="checkpoint names")
    parser.add_argument('--load_checkPoints', type=lambda x: not (str(x).lower() == 'false'), default=False,
                        help='load_checkpoints')
    parser.add_argument('--best_name', type=str, default=False, help='best name')
    parser.add_argument('--log_filename', type=str, default="log", help="create_a_logfilename")
    parser.add_argument('--train', type=lambda x: not (str(x).lower() == 'false'), default=True,
                        help='train')
    parser.add_argument('--evaluate', type=lambda x: not (str(x).lower() == 'false'), default=False,
                        help='evaluate')
    parser.add_argument('--evaluate_prefix', type=str, default="xxx", help="checkpoint names")

    return parser.parse_args()


if __name__ == '__main__':
    args_ = parse_args()
    yaml_file = args_.yaml_file
    yaml_config = YAMLConfig(yaml_file)
    trainer = MainTrainer(yaml_config=yaml_config.config)
    trainer.from_argparse_args(args=args_)
    print("===> define the Model")
    train_model = GraspModel(config=yaml_config.config,
                             checkpoint_name=args_.checkpoint_name,
                             best_name=args_.best_name,
                             logger_file_name=os.path.join(trainer.log_dir, args_.log_filename),
                             checkpoint_path=trainer.check_point_dir)
    train_model.count_parameters()

    if args_.train:
        print("===> create Nvidiagrasp data")
        train_dataset = NvidiaGraspDataSet(train=True, normalized=True)
        print("===> load data")
        trainer.load_data(train_dataset=train_dataset)
        print("===> set model")
        trainer.set_model(train_model)
        print("===> start to train the model")
        trainer.fit()
    if args_.evaluate:
        print("===> create test dataset")
        test_dataset = NvidiaGraspDataSet(train=False, normalized=True, ycb=True)
        print("===> load test dataset")
        trainer.load_data(test_dataset=test_dataset)
        if args_.load_checkPoints:
            print("===> load checkpoints")
            print("===> set model")
            trainer.set_model(train_model)
            print("===> load checkpoints")
            trainer.load_check_points(check_point_path=args_.checkpoint_name)
        print("===> start to predict")
        if args_.load_checkPoints:
            trainer.predict(evaluation_prefix=args_.evaluate_prefix)
        else:
            trainer.predict()
