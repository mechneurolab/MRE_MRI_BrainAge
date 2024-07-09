from typing import Any, Dict, Optional, Tuple

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split, TensorDataset
from torchvision.datasets import MNIST
from src.data.mre_brain_datautils import data_splitting, get_bin_centers, distribute_target
import pandas as pd
import numpy as np

class MREBrainDataModule(LightningDataModule):
    """LightningDataModule for MRE Brain maps data set

    A DataModule implements 5 key methods:

        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html
    """

    def __init__(
        self,
        data_dir: str = "data/",
        splits_dir: str = "splits/",
        train_val_test_split: Tuple[int, int, int] = (166, 56, 57),
        map_type: str = "Stiffness",
        batch_size: int = 4,
        distributed_target: bool = False,
        bin_range: Tuple[int, int] = (0,100),
        bin_step: int = 1,
        num_workers: int = 4,
        pin_memory: bool = False,

    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.n_maps = len(self.hparams.map_type.split('-'))
        # data transformations
        # self.transforms = transforms.Compose(
            # [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        # )

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        self.data_dict: Optional[Dataset] = None
    # @property
    # def num_classes(self):
        # return 10

    def prepare_data(self):
        """Download data if needed.

        Do not use it to assign state (self.x = y).
        """
        pass
        # MNIST(self.hparams.data_dir, train=True, download=True)
        # MNIST(self.hparams.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load splits and datasets
        if not self.data_train and not self.data_val and not self.data_test:
            train_id = pd.read_csv(self.hparams.splits_dir+'train_split.csv', delimiter=',', header=None).to_numpy().squeeze()
            val_id = pd.read_csv(self.hparams.splits_dir+'val_split.csv', delimiter=',', header=None).to_numpy().squeeze()
            test_id = pd.read_csv(self.hparams.splits_dir+'test_split.csv', delimiter=',', header=None).to_numpy().squeeze()
            ids = (train_id, val_id, test_id)
            self.data_dict = data_splitting(self.hparams.map_type, ids, self.hparams.data_dir)
            # Training set 
            train_maps = torch.Tensor( np.array(self.data_dict['train_data'][:self.n_maps]).swapaxes(0,1) )
            train_categorical = torch.Tensor(self.data_dict['train_data'][-1])
            train_ages = torch.Tensor(self.data_dict['train_target']).reshape(-1,1)
            # Validation set 
            val_maps = torch.Tensor( np.array(self.data_dict['val_data'][:self.n_maps]).swapaxes(0,1) )
            val_categorical = torch.Tensor(self.data_dict['val_data'][-1])
            val_ages = torch.Tensor(self.data_dict['val_target']).reshape(-1,1)
            # Test set
            test_maps = torch.Tensor( np.array(self.data_dict['test_data'][:self.n_maps]).swapaxes(0,1) )
            test_categorical = torch.Tensor(self.data_dict['test_data'][-1])
            test_ages = torch.Tensor(self.data_dict['test_target']).reshape(-1,1)
            if self.hparams.distributed_target:
                # ages = np.concatenate((self.data_dict['train_target'],self.data_dict['val_target'],self.data_dict['test_target']))
                # bin_range = [np.min(ages),np.max(ages)]
                self.bin_centers = get_bin_centers(self.hparams.bin_range, self.hparams.bin_step)
                train_ages_distribution = torch.Tensor(distribute_target(self.data_dict['train_target'], self.bin_centers))
                val_ages_distribution = torch.Tensor(distribute_target(self.data_dict['val_target'], self.bin_centers))
                test_ages_distribution = torch.Tensor(distribute_target(self.data_dict['test_target'], self.bin_centers))
                self.data_train = TensorDataset(train_maps, train_categorical, train_ages, train_ages_distribution)
                self.data_val = TensorDataset(val_maps, val_categorical, val_ages, val_ages_distribution)
                self.data_test = TensorDataset(test_maps, test_categorical, test_ages, test_ages_distribution)
            else:
                self.data_train = TensorDataset(train_maps, train_categorical, train_ages)
                self.data_val = TensorDataset(val_maps, val_categorical, val_ages)
                self.data_test = TensorDataset(test_maps, test_categorical, test_ages)
        
        # load only if not loaded already
            # trainset = MNIST(self.hparams.data_dir, train=True, transform=self.transforms)
            # testset = MNIST(self.hparams.data_dir, train=False, transform=self.transforms)
            # dataset = ConcatDataset(datasets=[trainset, testset])
            # self.data_train, self.data_val, self.data_test = random_split(
            #     dataset=dataset,
            #     lengths=self.hparams.train_val_test_split,
            #     generator=torch.Generator().manual_seed(42),
            # )

    def train_dataloader(self, shuffle_opt=True):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=shuffle_opt,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


if __name__ == "__main__":
    _ = MREBrainDataModule()
