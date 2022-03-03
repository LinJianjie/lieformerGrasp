import sys

from pathlib import Path

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[2]
sys.path.append(str(root))
import logging


class Logger:
    def __init__(self, logname="log", level="INFO", use_console=True, use_file=True, filename="log.log"):
        self.formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
        if level == "INFO":
            self.level = logging.INFO
        if level == "DEBUG":
            self.level = logging.DEBUG
        if level == "WARNING":
            self.level = logging.warning

        self.logger = logging.getLogger(logname)
        self.logger.setLevel(logging.INFO)
        # self._train_loss = []
        # self._evaluation_loss = []
        # self._test_loss = []
        if use_console:
            self.__add_console()

        if use_file:
            # self.filename = "log/"+filename
            self.filename = filename
            self.__add_file()

    def __add_console(self):
        console = logging.StreamHandler()
        console.setLevel(self.level)
        console.setFormatter(self.formatter)
        self.logger.addHandler(console)

    def __add_file(self):
        file = logging.FileHandler(filename=self.filename, mode='w')
        file.setLevel(self.level)
        file.setFormatter(self.formatter)
        self.logger.addHandler(file)

    def INFO(self, msg, *args, **kwargs):
        self.logger.info(msg, *args, **kwargs)

    # @property
    # def train_loss(self):
    #     return sum(self._train_loss) / len(self._train_loss)
    #
    # @train_loss.setter
    # def train_loss(self, value):
    #     self._train_loss.append(value)
    #
    # @property
    # def evaluation_loss(self):
    #     return sum(self._evaluation_loss) / len(self._evaluation_loss)
    #
    # @evaluation_loss.setter
    # def evaluation_loss(self, value):
    #     self._evaluation_loss.append(value)
    #
    # @property
    # def test_loss(self):
    #     return sum(self._test_loss) / len(self._test_loss)
    #
    # @test_loss.setter
    # def test_loss(self, value):
    #     self._test_loss.append(value)
    #
    # def metric_train_loss(self):
    #     self._train_loss = []
    #
    # def metric_evaluation_loss(self):
    #     self._evaluation_loss = []
    #
    # def metric_test_loss(self):
    #     self._test_loss = []
