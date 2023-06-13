import os

import json

import torch
import torch.utils.data as data
from imblearn import over_sampling

from . import utils

CLASSES = [
    "happiness",
    "sadness",
    "neutral",
    "anger",
    "surprise",
    "disgust",
    "fear"
]
script_dir = os.path.dirname(__file__)


class DFEW(data.Dataset):
    def __init__(
            self,
            root_path: str,
            transforms: callable = None,
            target_transform: callable = None,
            load_transform: callable = None,
            split: str = None,
            fold: int = None
    ):
        if split not in ['train', 'test']:
            raise ValueError
        if fold not in list(range(1, 6)):
            raise ValueError
        self.root_path = root_path
        self.transforms = transforms
        self.annotation_path = os.path.join(*[
            self.root_path,
            'DFEW_set_{}_{}.txt'.format(fold, split)
        ])
        self.target_transform = target_transform
        self.load_transform = load_transform
        self.data = self._make_dataset(
            self.annotation_path
        )

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        self._data = data
        self.labels = [x['label'] for x in self.data]
        self.indices = list(range(0, len(self.data)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (clip, target, video_idx)
        """
        sample = self.data[index]
        video_name = sample['video']
        label = sample['label']
        description = sample['descr']

        video_path = os.path.join(*[self.root_path, 'DFEW_Frame_Face', video_name])

        video = utils.load_frames(video_path, time_transform=self.load_transform)
        if self.transforms is not None:
            video = self.transforms(video)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return video, label, description

    def _make_dataset(self, annotation_path: str) -> list:
        annotations = utils.load_annotation(annotation_path, encoding='UTF-8', separator=' ')
        dataset = []
        for idx, row in enumerate(annotations):
            row = [el.replace('\n', '') for el in row]
            video_path = row[0]
            num_frames = row[1]
            label = int(row[2])

            video_info = video_path.split('/')
            video_idx = video_info[-1]

            description = ' '
            sample = {
                'video': video_idx,
                'descr': description,
                'label': label
            }
            dataset.append(sample)
        return dataset

    def resample(self):
        sampler = over_sampling.RandomOverSampler()
        idx = torch.arange(len(self.data)).reshape(-1, 1)
        y = torch.tensor([sample['label'] for sample in self.data]).reshape(-1, 1)
        idx, _ = sampler.fit_resample(idx, y)
        idx = idx.reshape(-1)
        data = [self.data[i] for i in idx]
        self.data = data
