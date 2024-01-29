from typing import Union, List, Optional

import pandas as pd
import os
import logging
import random

import pytorch_lightning as pl
from torchvision.transforms import RandomCrop
import torch
import torchaudio
from torchaudio.transforms import Resample
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset, default_collate, WeightedRandomSampler
import datasets

class AudioDataModule(pl.LightningDataModule):
    """ DataModule containing waveforms and optionally labels """
    def __init__(self,
                 datasets: List[Dataset],
                 batch_size: int, 
                 num_workers: int,
                 prefetch_factor: int,
                 crop_length: int,
                 shuffle: bool,
                 pin_memory: bool,
                 train_val_test_split: List[float],
                 return_labels: Optional[bool] = False,
                 seed: Optional[int] = 12345,
                 stratify: Optional[bool] = False,
                 verbose: Optional[bool] = False,
                ):
        super().__init__()
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.crop_length = crop_length
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.return_labels = return_labels
        self.seed = seed
        self.stratify = stratify
        self.verbose = verbose
        
        datasplit = self._process_datasets(datasets, train_val_test_split)
        self.train_dataset, self.val_dataset, self.test_dataset = datasplit

        if self.verbose:
            logging.info(f"batch_size={self.batch_size}, num_workers={self.num_workers}, prefetch_factor={self.prefetch_factor}, stratify={self.stratify}")

    
    def _process_datasets(self, datasets: List[Dataset], train_val_test_split: List[float]):
        split = lambda seq: random_split(seq,
                                         train_val_test_split,
                                         generator=torch.Generator().manual_seed(self.seed),
                                        )
        
        if self.stratify:
            try:
                classes_list = [ds.classes for ds in datasets]
            except AttributeError as e:
                raise AttributeError("if stratify is used, datasets need to have classes attribute of type pd.Series holding the class label")
            
            classes = pd.concat(classes_list, ignore_index=True)
            self.class_split = split(classes)
            
            
        dataset = ConcatDataset(datasets)
        datasplit = split(dataset)
        return datasplit

    
    def get_collate(self, eval: Optional[bool] = False):
        def collate(batch: List[torch.Tensor]):
            if not self.return_labels:
                cropped_batch = [self.crop(waveform, random_crop=not eval) for waveform in batch]
            else: 
                cropped_batch = [(self.crop(waveform, random_crop=not eval), label) for waveform, label in batch]
                
            return default_collate(cropped_batch)
        return collate

    
    def get_sampler(self, eval: Optional[bool] = False):
        if not self.stratify:
            return None
        
        if eval:
            i = 1
            length: int = len(self.val_dataset) 
        else:
            i = 0
            length: int = len(self.train_dataset) 
            
        classes: pd.Series = pd.Series(iter(self.class_split[i]))
        classes.fillna("Unknown", inplace=True)
        
        class_counts = classes.value_counts()
        num_classes = class_counts.size
        
        weights = classes.map(lambda c: 1/(class_counts[c] * num_classes))

        sampler = WeightedRandomSampler(weights=weights, num_samples=length)
        return sampler

        
    def crop(self, waveform: torch.Tensor, random_crop: Optional[bool] = True):
        channels,  total_length = waveform.shape
        
        if random_crop:
            start = random.randint(0, max(total_length - self.crop_length, 0))
        else:
            start = 0
        
        # Crop up to start
        x = waveform[:, start:]
        length_from_start = x.shape[-1]
        
        # Pad to end if not large enough, else crop end
        if length_from_start < self.crop_length:
            padding_length = self.crop_length - length_from_start
            padding = torch.zeros(channels, padding_length).to(x)
            return torch.cat([waveform, padding], dim=-1)
        else:
            return x[:, :self.crop_length]
    
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle if not self.stratify else None,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
            collate_fn=self.get_collate(),
            sampler=self.get_sampler(),
        )
        
    def val_dataloader(self) -> DataLoader:
        eval = True
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False if not self.stratify else None,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
            collate_fn=self.get_collate(eval),
            sampler=self.get_sampler(eval),
        )
        
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False if not self.stratify else None,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
            collate_fn=self.get_collate(eval=True),
            sampler=self.get_sampler(eval),
        )


def read_audio_file(file, target_channels: int, target_sr: int, resample: Optional[Resample] = None):
    waveform, sr = torchaudio.load(file)
    channels = waveform.shape[0]

    # apply resampler and check samplingrates
    if resample:
        assert sr == resample.orig_freq, f"The audio at {file} has a sample rate of {sr} while a sample rate of {target_sr} is expected by resample func."
        assert target_sr == resample.new_freq, f"Resample func outputs sr={resample.new_freq} but sr={target_sr} was expected"
        waveform = resample(waveform)
    else:
        assert sr == target_sr, f"The audio at {file} has a sample rate of {sr} while a sample rate of {target_sr} is expected."

    # check and fix channel counts
    if target_channels == channels:
        return waveform

    if target_channels == 1:
        # channels  is > 1
        # to mono
        return waveform.mean(dim=0, keepdim=True)
        
    if target_channels == 2:
        if channels > 2:
            # to mono
            waveform = waveform.mean(dim=0, keepdim=True)
            logging.warn(f"Audiofile with {channels} channels converted to mono and back to 2 channels to obtain expected channel count. File at {file} ")
        # broadcast to stereo
        return torch.cat([waveform, waveform])

    # none of the cases that could be handled was met
    raise AssertionError(f"Could not interpolate target_channels={target_channels} and found_channels={channels}.")
     
    
class XenoCantoAudioDataset(Dataset):
    def __init__(self, 
                sample_rate: int,
                channels: int,
                data_dir: str,
                metadata_dir: str,
                max_duration: Optional[float] = 300, #in seconds
                return_labels: Optional[bool] = False,
                label_col_name: Optional[str] = "label", 
                ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.data_dir = data_dir
        self.max_duration = max_duration
        self.metadata_dir = metadata_dir
        self.return_labels = return_labels
        self.label_col_name = label_col_name
        
        self.metadata = self._load_metadata()
        self.files: pd.Series = self.metadata["file"].map(lambda x: os.path.join(self.data_dir, x))

        self.classes: pd.Series = self.metadata["sci_name"] #may be used for stratification 
    
        if self.return_labels:
            self.labels: pd.Series = self.metadata[self.label_col_name]

    
    def _load_metadata(self):
        metadata = datasets.load_from_disk(self.metadata_dir)
        metadata = metadata.filter(lambda rec: rec["duration"] <= self.max_duration)
        return metadata.with_format("pandas")
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        file = self.files.iloc[idx]
        waveform = read_audio_file(file, target_channels=self.channels, target_sr=self.sample_rate)
        if not self.return_labels:
            return waveform
        else:
            label = self.labels.iloc[idx]
            return waveform, label


class BergmanAudioDataset(Dataset):
    def __init__(self, 
                sample_rate: int,
                channels: int,
                metadata_file: str,
                max_duration: Optional[float] = 300, #in seconds
                return_labels: Optional[bool] = False,
                ):  
        self.sample_rate = sample_rate
        self.data_sample_rate = 44100
        self.channels = channels
        self.metadata_file = metadata_file
        self.max_duration = max_duration
        self.return_labels = return_labels

        self.resample = Resample(self.data_sample_rate, self.sample_rate)
        
        metadata = pd.read_csv(metadata_file)
        metadata = metadata[metadata["duration"] <= self.max_duration]
        
        self.files: pd.Series = metadata["file"]
        self.classes: pd.Series = metadata["name_sci"] #may be used for stratification 
        if self.return_labels:
            self.labels: pd.Series = metadata["label"]

        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        file = self.files.iloc[idx]
        waveform = read_audio_file(file, target_channels=self.channels, target_sr=self.sample_rate, resample=self.resample)
        if not self.return_labels:
            return waveform
        else:
            label = self.labels.iloc[idx]
            return waveform, label
